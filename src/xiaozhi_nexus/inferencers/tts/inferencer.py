from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import httpx
from openai import OpenAI

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false


@dataclass
class OpenAITTSInferencer:
    """
    基于 OpenAI Audio Speech API 的 TTS 推理器

    接口设计:
    - input: str (要合成的文本)
    - output: Iterator[np.ndarray] (float32 PCM 音频片段)
    """

    # OpenAI 配置
    base_url: str = "http://localhost:10001/v1"
    api_key: str = "no-key"
    model: str = "fnlp/MOSS-TTSD-v0.5"
    voice: str = "fnlp/MOSS-TTSD-v0.5:anna"
    response_format: str = "wav"

    # 音频输出配置
    output_sample_rate: Optional[int] = None
    chunk_duration_ms: int = 100

    # SSL 配置
    verify_ssl: bool = True

    # 内部状态
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        http_client = None
        if not self.verify_ssl:
            http_client = httpx.Client(verify=False)

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
        )

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """
        同步 TTS 推理

        Args:
            text: 要合成的文本

        Yields:
            float32 PCM 片段（单声道）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        if not text:
            return

        if self.response_format.lower() != "wav":
            raise ValueError("Only wav response_format is supported")

        with self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=str(text),
            response_format=self.response_format,
        ) as response:
            wav_bytes = b"".join(response.iter_bytes())

        pcm, sample_rate = self._decode_wav_bytes(wav_bytes)

        target_rate = self.output_sample_rate or sample_rate
        if target_rate != sample_rate:
            pcm = self._resample_audio(pcm, sample_rate, target_rate)
            sample_rate = target_rate

        for chunk in self._chunk_audio(pcm, sample_rate):
            yield chunk

    def _decode_wav_bytes(self, data: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        pcm = self._pcm_bytes_to_float32(frames, sample_width)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        return pcm.astype(np.float32), int(sample_rate)

    def _pcm_bytes_to_float32(self, data: bytes, sample_width: int) -> np.ndarray:
        if sample_width == 1:
            pcm_u8 = np.frombuffer(data, dtype=np.uint8)
            return (pcm_u8.astype(np.float32) - 128.0) / 128.0
        if sample_width == 2:
            pcm_i16 = np.frombuffer(data, dtype="<i2")
            return pcm_i16.astype(np.float32) / 32768.0
        if sample_width == 3:
            raw = np.frombuffer(data, dtype=np.uint8)
            if raw.size % 3:
                raw = raw[: raw.size - (raw.size % 3)]
            raw = raw.reshape(-1, 3)
            pcm_i32 = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            pcm_i32 = (pcm_i32 ^ sign_bit) - sign_bit
            return pcm_i32.astype(np.float32) / float(1 << 23)
        if sample_width == 4:
            pcm_i32 = np.frombuffer(data, dtype="<i4")
            return pcm_i32.astype(np.float32) / float(1 << 31)
        raise ValueError(f"Unsupported sample width: {sample_width}")

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        if audio.size == 0:
            return audio.astype(np.float32)
        import librosa

        resampled = librosa.resample(
            audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
        )
        return resampled.astype(np.float32)

    def _chunk_audio(self, audio: np.ndarray, sample_rate: int) -> Iterator[np.ndarray]:
        if audio.size == 0:
            return
        chunk_size = int(sample_rate * (self.chunk_duration_ms / 1000.0))
        if chunk_size <= 0:
            yield audio.astype(np.float32)
            return
        for idx in range(0, int(audio.shape[0]), chunk_size):
            yield audio[idx : idx + chunk_size].astype(np.float32)


TTSInferencer = OpenAITTSInferencer
