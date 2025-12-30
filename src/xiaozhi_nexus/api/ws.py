from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from xiaozhi_nexus.audio.opus import OpusDecoder, OpusEncoder
from xiaozhi_nexus.runtime.session import StreamSession
from xiaozhi_nexus.inferencers.stream_asr import OpenAIRealtimeASRInferencer
from xiaozhi_nexus.inferencers.chat import OpenAIChatInferencer
from xiaozhi_nexus.inferencers.tts import OpenAITTSInferencer
from xiaozhi_nexus.config import get_config

router = APIRouter()


def _create_chat_inferencer() -> OpenAIChatInferencer | None:
    """
    从全局配置创建 Chat 推理器

    Returns:
        OpenAIChatInferencer 实例，如果未配置 API Key 则返回 None
    """
    cfg = get_config()

    api_key = cfg.openai.api_key
    if not api_key:
        return None

    return OpenAIChatInferencer(
        base_url=cfg.openai.base_url,
        api_key=api_key,
        model=cfg.openai.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        max_history=cfg.llm.max_history,
        system_prompt=cfg.system.prompt,
        verify_ssl=cfg.openai.verify_ssl,
    )


def _create_tts_inferencer(sample_rate: int) -> OpenAITTSInferencer:
    """从全局配置创建 TTS 推理器"""
    cfg = get_config()

    # TTS 配置回退到 OpenAI 配置
    base_url = cfg.tts.base_url or cfg.openai.base_url
    api_key = cfg.tts.api_key or cfg.openai.api_key

    return OpenAITTSInferencer(
        base_url=base_url,
        api_key=api_key,
        model=cfg.tts.model,
        voice=cfg.tts.voice,
        response_format=cfg.tts.response_format,
        output_sample_rate=sample_rate,
        chunk_duration_ms=cfg.tts.chunk_duration_ms,
        verify_ssl=cfg.tts.verify_ssl,
    )


def _create_asr_inferencer(sample_rate: int) -> OpenAIRealtimeASRInferencer:
    """从全局配置创建 ASR 推理器"""
    cfg = get_config()

    # ASR 配置回退到 OpenAI 配置
    base_url = cfg.asr.base_url or cfg.openai.base_url
    api_key = cfg.asr.api_key or cfg.openai.api_key

    return OpenAIRealtimeASRInferencer(
        base_url=base_url,
        api_key=api_key,
        model=cfg.asr.model,
        sample_rate=sample_rate,
        chunk_duration_ms=cfg.asr.chunk_duration_ms,
        verify_ssl=cfg.asr.verify_ssl,
    )


@dataclass(frozen=True)
class AudioParams:
    format: str
    sample_rate: int
    channels: int
    frame_duration: int

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * (self.frame_duration / 1000))


def _parse_audio_params(payload: dict[str, Any]) -> AudioParams:
    audio = payload.get("audio_params") or {}
    return AudioParams(
        format=str(audio.get("format") or "opus"),
        sample_rate=int(audio.get("sample_rate") or 16000),
        channels=int(audio.get("channels") or 1),
        frame_duration=int(audio.get("frame_duration") or 20),
    )


OutgoingKind = Literal["json", "bytes"]


@dataclass(frozen=True)
class Outgoing:
    kind: OutgoingKind
    payload: Any


@router.websocket("/ws")
@router.websocket("/xiaozhi/v1")
@router.websocket("/xiaozhi/v1/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    loop = asyncio.get_running_loop()
    outgoing: asyncio.Queue[Outgoing] = asyncio.Queue(maxsize=2000)

    listening = False
    audio_params: AudioParams | None = None
    decoder: OpusDecoder | None = None
    encoder = OpusEncoder(sample_rate=24000, channels=1, frame_duration_ms=20)

    session: StreamSession | None = None

    async def sender_loop() -> None:
        while True:
            try:
                msg = await outgoing.get()
                if msg.kind == "json":
                    await websocket.send_text(
                        json.dumps(msg.payload, ensure_ascii=False)
                    )
                else:
                    await websocket.send_bytes(msg.payload)
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                break
            except Exception:
                break

    sender_task = asyncio.create_task(sender_loop())

    def _enqueue(msg: Outgoing) -> None:
        try:
            outgoing.put_nowait(msg)
        except asyncio.QueueFull:
            logging.warning("Outgoing queue full (2000), audio packets may be dropped!")

    def publish_json(payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(_enqueue, Outgoing(kind="json", payload=payload))

    def publish_bytes(payload: bytes) -> None:
        loop.call_soon_threadsafe(_enqueue, Outgoing(kind="bytes", payload=payload))

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                raise WebSocketDisconnect(code=1000)

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                typ = payload.get("type")
                if typ == "hello":
                    audio_params = _parse_audio_params(payload)
                    decoder = OpusDecoder(
                        sample_rate=audio_params.sample_rate,
                        channels=audio_params.channels,
                        frame_size=audio_params.frame_size,
                    )
                    publish_json(
                        {"type": "hello", "transport": "websocket", "version": 1}
                    )
                    continue

                if typ == "listen":
                    state = payload.get("state")
                    if state == "start":
                        listening = True
                        # 如果已有 session 且正在运行，触发中断（用户打断）
                        if session is not None:
                            session.interrupt()
                        else:
                            # 创建新的 session
                            if not decoder or not audio_params:
                                continue
                            cfg = get_config()
                            session = StreamSession(
                                publish_json=publish_json,
                                publish_bytes=publish_bytes,
                                asr_inferencer=_create_asr_inferencer(
                                    audio_params.sample_rate
                                ),
                                chat_inferencer=_create_chat_inferencer(),
                                tts=_create_tts_inferencer(encoder.sample_rate),
                                encoder=encoder,
                                allow_interrupt=cfg.system.allow_interrupt,
                                audio_send_delay_ms=cfg.tts.audio_send_delay_ms,
                                tts_split_by_punctuation=cfg.tts.split_by_punctuation,
                            )
                            session.start()
                        continue
                    if state == "stop":
                        listening = False
                        if session:
                            session.stop()
                            session = None
                        continue

                continue

            if "bytes" in message and message["bytes"] is not None:
                if not listening or not decoder or not session:
                    continue
                try:
                    pcm = decoder.decode_to_float32(message["bytes"])
                except Exception:
                    continue
                session.push_audio(pcm)

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            session.stop()
        sender_task.cancel()
        try:
            await sender_task
        except (asyncio.CancelledError, Exception):
            pass
