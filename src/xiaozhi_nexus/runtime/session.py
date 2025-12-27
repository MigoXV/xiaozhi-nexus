from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from xiaozhi_nexus.audio.opus import OpusEncoder
from xiaozhi_nexus.inferencers.stream_asr import OpenAIRealtimeASRInferencer
from xiaozhi_nexus.inferencers.chat import OpenAIChatInferencer
from xiaozhi_nexus.inferencers.tts import OpenAITTSInferencer


@dataclass
class StreamSession:
    """
    流式会话管理器

    工作流程: 语音输入 → ASR → LLM 对话 → TTS → 音频输出

    支持用户打断：当用户开始新的语音输入时，会中断当前的 LLM 生成和 TTS 播放
    """

    publish_json: Callable[[dict], None]
    publish_bytes: Callable[[bytes], None]
    asr_inferencer: OpenAIRealtimeASRInferencer
    tts: OpenAITTSInferencer
    encoder: OpusEncoder
    chat_inferencer: Optional[OpenAIChatInferencer] = None
    input_maxsize: int = 200

    # 内部状态
    _audio_q: queue.Queue[np.ndarray | None] = field(init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _running: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _interrupted: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self._audio_q = queue.Queue(maxsize=self.input_maxsize)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._interrupted.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._interrupted.set()
        while True:
            try:
                self._audio_q.put(None, timeout=0.05)
                break
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def interrupt(self) -> None:
        """
        中断当前的 LLM 生成和 TTS 播放

        用于用户打断场景：当检测到新的语音输入时调用
        """
        self._interrupted.set()

    def clear_interrupt(self) -> None:
        """清除中断标志，准备处理新的输入"""
        self._interrupted.clear()

    def push_audio(self, pcm_f32: np.ndarray) -> None:
        if not self._running.is_set():
            return
        try:
            self._audio_q.put_nowait(np.asarray(pcm_f32, dtype=np.float32))
        except queue.Full:
            pass

    def _audio_iter(self):
        while self._running.is_set():
            item = self._audio_q.get()
            if item is None:
                break
            yield item

    def _is_interrupted(self) -> bool:
        """检查是否被中断"""
        return self._interrupted.is_set() or not self._running.is_set()

    def _worker(self) -> None:
        for user_text in self.asr_inferencer(self._audio_iter()):
            if self._is_interrupted():
                self.clear_interrupt()
                continue

            # 1. 发送 ASR 识别结果
            self.publish_json({"type": "stt", "text": user_text})

            # 2. 调用 LLM 生成响应
            if self.chat_inferencer:
                response_text = self._process_llm(user_text)
                if response_text is None:
                    # 被中断，跳过 TTS
                    self.clear_interrupt()
                    continue
            else:
                # 无 LLM 时直接回显用户输入（用于测试）
                response_text = user_text

            # 3. TTS 合成并发送音频
            if self._is_interrupted():
                self.clear_interrupt()
                continue

            self._process_tts(response_text)

            # 4. 发送情绪状态（可扩展）
            self.publish_json({"type": "llm", "emotion": "neutral"})

    def _process_llm(self, user_text: str) -> Optional[str]:
        """
        处理 LLM 推理，支持流式输出和中断

        Returns:
            完整的 LLM 响应文本，如果被中断则返回 None
        """
        if not self.chat_inferencer:
            return user_text

        self.publish_json({"type": "llm", "state": "start"})

        full_response = ""
        try:
            for chunk in self.chat_inferencer(user_text):
                if self._is_interrupted():
                    self.publish_json({"type": "llm", "state": "stop", "interrupted": True})
                    return None

                full_response += chunk
                # 流式发送 LLM 响应文本
                self.publish_json({"type": "llm", "text": chunk})

            self.publish_json({"type": "llm", "state": "stop"})
            return full_response

        except Exception as e:
            self.publish_json({"type": "llm", "state": "error", "error": str(e)})
            return None

    def _process_tts(self, text: str) -> None:
        """
        处理 TTS 合成，支持中断

        Args:
            text: 要合成的文本
        """
        self.publish_json({"type": "tts", "state": "start", "text": text})

        try:
            for pcm in self.tts.synthesize(text):
                if self._is_interrupted():
                    self.publish_json({"type": "tts", "state": "stop", "interrupted": True})
                    return

                for packet in self.encoder.encode_pcm_float32(pcm):
                    if self._is_interrupted():
                        self.publish_json({"type": "tts", "state": "stop", "interrupted": True})
                        return
                    self.publish_bytes(packet)

            self.publish_json({"type": "tts", "state": "stop"})

        except Exception as e:
            self.publish_json({"type": "tts", "state": "error", "error": str(e)})
