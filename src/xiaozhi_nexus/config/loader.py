"""
配置加载器

使用 OmegaConf 加载和验证 YAML 配置文件
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig

from .schema import AppConfig


# 全局配置单例
_config: Optional[AppConfig] = None

# 配置文件路径环境变量
CONFIG_ENV_VAR = "XIAOZHI_NEXUS_CONFIG"


def get_config_path() -> Optional[Path]:
    """
    获取配置文件路径

    优先级:
    1. 环境变量 XIAOZHI_NEXUS_CONFIG
    2. 当前目录下的 config.yaml
    3. None (使用默认配置)
    """
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    return None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，如果为 None 则自动查找

    Returns:
        验证后的 AppConfig 实例

    Raises:
        FileNotFoundError: 配置文件不存在
        omegaconf.errors.ConfigAttributeError: 配置验证失败
    """
    global _config

    # 创建结构化配置的 schema
    schema: DictConfig = OmegaConf.structured(AppConfig)

    if config_path is None:
        config_path = get_config_path()

    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载用户配置
        user_config = OmegaConf.load(config_path)

        # 合并配置 (用户配置覆盖默认值)
        merged = OmegaConf.merge(schema, user_config)
    else:
        # 使用默认配置
        merged = schema

    # 转换为 dataclass 实例并验证
    _config = OmegaConf.to_object(merged)  # type: ignore[assignment]

    # 后处理：加载 system prompt 文件
    if _config.system.prompt_file:
        prompt_path = Path(_config.system.prompt_file)
        if prompt_path.exists() and prompt_path.is_file():
            _config.system.prompt = prompt_path.read_text(encoding="utf-8").strip()

    # 后处理：设置 opus 库路径环境变量
    if _config.system.opus_lib:
        os.environ["XIAOZHI_OPUS_LIB"] = _config.system.opus_lib

    return _config


def get_config() -> AppConfig:
    """
    获取当前配置

    Returns:
        当前 AppConfig 实例

    Raises:
        RuntimeError: 配置未初始化
    """
    if _config is None:
        raise RuntimeError(
            "配置未初始化，请先调用 load_config() 或通过命令行启动"
        )
    return _config


def validate_config(config: AppConfig) -> list[str]:
    """
    验证配置完整性

    Args:
        config: 要验证的配置

    Returns:
        错误消息列表，如果为空则验证通过
    """
    errors: list[str] = []

    # 验证 OpenAI API Key
    if not config.openai.api_key:
        errors.append("openai.api_key 未配置")

    # 验证 TTS 配置 (如果没有单独配置，检查是否能回退到 OpenAI)
    if config.tts.api_key is None and not config.openai.api_key:
        errors.append("tts.api_key 未配置且无法回退到 openai.api_key")

    # 验证 ASR 配置 (如果没有单独配置，检查是否能回退到 OpenAI)
    if config.asr.api_key is None and not config.openai.api_key:
        errors.append("asr.api_key 未配置且无法回退到 openai.api_key")

    # 验证 system prompt 文件路径
    if config.system.prompt_file:
        prompt_path = Path(config.system.prompt_file)
        if not prompt_path.exists():
            errors.append(f"system.prompt_file 不存在: {config.system.prompt_file}")

    return errors


def reset_config() -> None:
    """重置配置（主要用于测试）"""
    global _config
    _config = None
