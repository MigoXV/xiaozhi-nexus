"""
配置模式定义

使用 dataclass 定义类型安全的配置结构，与 OmegaConf 配合使用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAIConfig:
    """OpenAI API 配置"""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    verify_ssl: bool = True


@dataclass
class LLMConfig:
    """LLM 生成参数配置"""

    temperature: float = 0.7
    max_tokens: int = 1024
    max_history: int = 20


@dataclass
class TTSConfig:
    """TTS 配置"""

    base_url: Optional[str] = None  # None 表示回退到 OpenAI 配置
    api_key: Optional[str] = None  # None 表示回退到 OpenAI 配置
    model: str = "fnlp/MOSS-TTSD-v0.5"
    voice: str = "fnlp/MOSS-TTSD-v0.5:anna"
    response_format: str = "wav"
    output_sample_rate: Optional[int] = None
    chunk_duration_ms: int = 100
    verify_ssl: bool = True

    # 音频包发送延时（毫秒），用于控制发送速度接近实时播放，0 表示不延时
    audio_send_delay_ms: float = 15.0

    # 是否按标点符号分段进行 TTS 合成（分段可以加快首包响应，但可能影响语音连贯性）
    split_by_punctuation: bool = True


@dataclass
class ASRConfig:
    """ASR 配置"""

    base_url: Optional[str] = None  # None 表示回退到 OpenAI 配置
    api_key: Optional[str] = None  # None 表示回退到 OpenAI 配置
    model: str = "gpt-4o-realtime-preview"
    sample_rate: int = 16000
    chunk_duration_ms: int = 100
    verify_ssl: bool = True


@dataclass
class SystemConfig:
    """系统配置"""

    # system prompt 可以是字符串，也可以是文件路径
    prompt: str = "你是小智，一个有帮助的语音助手。请用简洁友好的方式回答用户的问题。"
    prompt_file: Optional[str] = None  # 优先级高于 prompt

    # Opus 库路径
    opus_lib: Optional[str] = None

    # 是否允许用户打断（当用户开始新的语音输入时，中断当前的 LLM 生成和 TTS 播放）
    allow_interrupt: bool = True


@dataclass
class ServerConfig:
    """服务器配置"""

    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class AppConfig:
    """应用程序顶层配置"""

    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
