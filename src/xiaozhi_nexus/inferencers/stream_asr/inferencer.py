from __future__ import annotations

import ssl
import base64
import asyncio
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Iterator, AsyncIterator, Optional

import numpy as np
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


@dataclass
class OpenAIRealtimeASRInferencer:
    """
    基于 OpenAI Realtime API 的流式 ASR 推理器

    接口设计参考 stubs/asr.py:
    - input: iterator of PCM float32 (mono)
    - output: iterator of incremental transcripts

    支持同步 (__call__) 和异步 (astream) 两种调用方式
    """

    # OpenAI 配置
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-realtime-preview"

    # 音频配置
    sample_rate: int = 16000
    chunk_duration_ms: int = 100  # 每个块的时长（毫秒）

    # SSL 配置
    verify_ssl: bool = True

    # 内部状态
    _client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)
    _ssl_context: Optional[ssl.SSLContext] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """初始化 OpenAI 客户端和 SSL 上下文"""
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        if not self.verify_ssl:
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE

    @property
    def chunk_size(self) -> int:
        """每个块的采样点数"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    def _float32_to_pcm16(self, audio_float32: np.ndarray) -> bytes:
        """将 float32 音频数据转换为 PCM16 字节"""
        # float32 范围 [-1.0, 1.0] 转换为 int16 范围 [-32768, 32767]
        audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _pcm16_to_base64(self, pcm16_data: bytes) -> str:
        """将 PCM16 字节转换为 base64 编码"""
        return base64.b64encode(pcm16_data).decode("utf-8")

    async def _send_audio_stream(
        self,
        connection: AsyncRealtimeConnection,
        audio_iter: AsyncIterator[np.ndarray],
    ):
        """异步发送音频流到 OpenAI Realtime API"""
        # 启动流式转录
        await connection.send({"type": "response.create", "response": {}})

        async for chunk in audio_iter:
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue

            # 转换为 PCM16 并编码
            pcm16_data = self._float32_to_pcm16(chunk)
            audio_base64 = self._pcm16_to_base64(pcm16_data)

            await connection.send({
                "type": "input_audio_buffer.append",
                "audio": audio_base64,
            })

            # 小延迟模拟实时音频流，避免发送过快
            await asyncio.sleep(0.01)

        # 发送结束标记
        await connection.send({"type": "input_audio_buffer.commit"})

    async def _receive_transcripts(
        self,
        connection: AsyncRealtimeConnection,
    ) -> AsyncIterator[str]:
        """异步接收转录结果"""
        async for event in connection:
            event_type = event.type

            if event_type == "response.audio_transcript.delta":
                # 每次返回独立的文本块
                yield event.delta

            elif event_type == "response.audio_transcript.done":
                # 转录完成
                break

            elif event_type == "response.text.delta":
                # 文本响应（备用）
                yield event.delta

            elif event_type == "response.text.done":
                break

            elif event_type == "response.done":
                break

            elif event_type == "error":
                raise RuntimeError(f"OpenAI Realtime API error: {event}")

    async def astream(self, audio_iter: AsyncIterator[np.ndarray]) -> AsyncIterator[str]:
        """
        异步流式 ASR 推理

        Args:
            audio_iter: 异步音频数据迭代器，每个元素为 float32 numpy 数组

        Yields:
            增量转录文本（累积形式）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        websocket_options = {}
        if self._ssl_context is not None:
            websocket_options["ssl"] = self._ssl_context

        async with self._client.beta.realtime.connect(
            model=self.model,
            websocket_connection_options=websocket_options,
        ) as connection:
            # 创建发送任务
            send_task = asyncio.create_task(
                self._send_audio_stream(connection, audio_iter)
            )

            try:
                # 接收并 yield 转录结果
                async for transcript in self._receive_transcripts(connection):
                    yield transcript
            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

    def __call__(self, audio_iter: Iterator[np.ndarray]) -> Iterator[str]:
        """
        同步流式 ASR 推理（兼容 stubs/asr.py 接口）

        Args:
            audio_iter: 同步音频数据迭代器，每个元素为 float32 numpy 数组

        Yields:
            增量转录文本（累积形式）
        """
        # 使用队列在线程间传递结果
        result_queue: Queue[str | Exception | None] = Queue()

        async def _async_wrapper():
            """异步包装器，将同步迭代器转换为异步"""
            async def async_audio_iter() -> AsyncIterator[np.ndarray]:
                for chunk in audio_iter:
                    yield chunk
                    await asyncio.sleep(0)  # 让出控制权

            try:
                async for transcript in self.astream(async_audio_iter()):
                    result_queue.put(transcript)
            except Exception as e:
                result_queue.put(e)
            finally:
                result_queue.put(None)  # 结束标记

        def _run_async():
            """在新线程中运行异步代码"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_async_wrapper())
            finally:
                loop.close()

        # 启动异步处理线程
        thread = threading.Thread(target=_run_async, daemon=True)
        thread.start()

        # 从队列中读取结果
        while True:
            try:
                result = result_queue.get(timeout=60000)  # 60秒超时
                if result is None:
                    break
                if isinstance(result, Exception):
                    raise result
                yield result
            except Empty:
                raise TimeoutError("ASR inference timeout")

        thread.join(timeout=5)


@dataclass
class OpenAIRealtimeASRInferencerAsync:
    """
    纯异步版本的 OpenAI Realtime ASR 推理器

    适用于异步应用场景，提供更好的性能
    """

    # OpenAI 配置
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-realtime-preview"

    # 音频配置
    sample_rate: int = 16000

    # SSL 配置
    verify_ssl: bool = True

    # 内部状态
    _client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)
    _ssl_context: Optional[ssl.SSLContext] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        if not self.verify_ssl:
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE

    def _float32_to_pcm16_base64(self, audio_float32: np.ndarray) -> str:
        """将 float32 音频转换为 base64 编码的 PCM16"""
        audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode("utf-8")

    async def transcribe(
        self,
        audio_iter: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[str]:
        """
        流式转录音频

        Args:
            audio_iter: 异步音频迭代器，float32 格式

        Yields:
            转录文本块（每次返回独立的文本块）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        websocket_options = {}
        if self._ssl_context is not None:
            websocket_options["ssl"] = self._ssl_context

        async with self._client.beta.realtime.connect(
            model=self.model,
            websocket_connection_options=websocket_options,
        ) as connection:
            # 启动转录
            await connection.send({"type": "response.create", "response": {}})

            audio_done = asyncio.Event()

            async def send_audio():
                async for chunk in audio_iter:
                    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        continue
                    audio_base64 = self._float32_to_pcm16_base64(chunk)
                    await connection.send({
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64,
                    })
                    await asyncio.sleep(0.01)
                await connection.send({"type": "input_audio_buffer.commit"})
                audio_done.set()

            send_task = asyncio.create_task(send_audio())

            try:
                async for event in connection:
                    event_type = event.type

                    if event_type == "response.audio_transcript.delta":
                        # 每次返回独立的文本块
                        yield event.delta

                    elif event_type == "response.audio_transcript.done":
                        break

                    elif event_type == "response.text.delta":
                        yield event.delta

                    elif event_type == "response.text.done":
                        break

                    elif event_type == "response.done":
                        break

                    elif event_type == "error":
                        raise RuntimeError(f"OpenAI Realtime API error: {event}")

            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass


# 便捷别名
StreamASRInferencer = OpenAIRealtimeASRInferencer
AsyncStreamASRInferencer = OpenAIRealtimeASRInferencerAsync
