"""
配置管理模块

提供基于 YAML 和 OmegaConf 的配置管理
"""

from .schema import (
    AppConfig,
    OpenAIConfig,
    LLMConfig,
    TTSConfig,
    ASRConfig,
    SystemConfig,
    ServerConfig,
)
from .loader import (
    load_config,
    get_config,
    validate_config,
    reset_config,
    get_config_path,
    CONFIG_ENV_VAR,
)

__all__ = [
    # Schema
    "AppConfig",
    "OpenAIConfig",
    "LLMConfig",
    "TTSConfig",
    "ASRConfig",
    "SystemConfig",
    "ServerConfig",
    # Loader
    "load_config",
    "get_config",
    "validate_config",
    "reset_config",
    "get_config_path",
    "CONFIG_ENV_VAR",
]
