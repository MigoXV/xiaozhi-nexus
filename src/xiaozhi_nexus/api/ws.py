from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from xiaozhi_nexus.audio.opus import OpusDecoder, OpusEncoder
from xiaozhi_nexus.runtime.session import StreamSession
from xiaozhi_nexus.stubs.asr import StreamIASRnferencer
from xiaozhi_nexus.stubs.tts import SineWaveTTS

router = APIRouter()


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
            pass

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
                        if session is None:
                            if not decoder or not audio_params:
                                continue
                            session = StreamSession(
                                publish_json=publish_json,
                                publish_bytes=publish_bytes,
                                inferencer=StreamIASRnferencer(
                                    sample_rate=audio_params.sample_rate
                                ),
                                tts=SineWaveTTS(sample_rate=encoder.sample_rate),
                                encoder=encoder,
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
