"""
Typer 命令行应用入口
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import logging
import typer

from xiaozhi_nexus.config import (
    load_config,
    validate_config,
    CONFIG_ENV_VAR,
)

app = typer.Typer(
    name="xiaozhi-nexus",
    help="小智语音助手后端服务",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _load_and_validate_config(config_path: Optional[Path]) -> None:
    """加载并验证配置"""
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as e:
        typer.secho(f"错误: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"配置加载失败: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # 验证配置
    errors = validate_config(cfg)
    if errors:
        typer.secho("配置验证失败:", fg=typer.colors.RED, err=True)
        for error in errors:
            typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar=CONFIG_ENV_VAR,
        help=f"配置文件路径 (YAML)，也可通过环境变量 {CONFIG_ENV_VAR} 指定",
        exists=False,  # 我们自己处理文件不存在的情况
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-h",
        help="绑定主机地址 (覆盖配置文件中的 server.host)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="绑定端口 (覆盖配置文件中的 server.port)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="开发模式：代码变更时自动重载",
    ),
) -> None:
    """启动 xiaozhi-nexus WebSocket 服务"""
    import uvicorn

    # 加载并验证配置
    _load_and_validate_config(config)

    # 从已加载的配置获取服务器设置
    from xiaozhi_nexus.config import get_config

    cfg = get_config()

    # 命令行参数覆盖配置文件
    bind_host = host if host is not None else cfg.server.host
    bind_port = port if port is not None else cfg.server.port

    typer.secho(
        f"启动服务: http://{bind_host}:{bind_port}",
        fg=typer.colors.GREEN,
    )

    uvicorn.run(
        "xiaozhi_nexus.api.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="info",
    )


# @app.command()
# def validate(
#     config: Optional[Path] = typer.Option(
#         None,
#         "--config",
#         "-c",
#         envvar=CONFIG_ENV_VAR,
#         help=f"配置文件路径 (YAML)，也可通过环境变量 {CONFIG_ENV_VAR} 指定",
#     ),
# ) -> None:
#     """验证配置文件"""
#     try:
#         cfg = load_config(config)
#     except FileNotFoundError as e:
#         typer.secho(f"错误: {e}", fg=typer.colors.RED, err=True)
#         raise typer.Exit(1)
#     except Exception as e:
#         typer.secho(f"配置加载失败: {e}", fg=typer.colors.RED, err=True)
#         raise typer.Exit(1)

#     errors = validate_config(cfg)
#     if errors:
#         typer.secho("配置验证失败:", fg=typer.colors.RED, err=True)
#         for error in errors:
#             typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
#         raise typer.Exit(1)

#     typer.secho("✓ 配置验证通过", fg=typer.colors.GREEN)

#     # 显示配置摘要
#     typer.echo("\n配置摘要:")
#     typer.echo(f"  OpenAI Base URL: {cfg.openai.base_url}")
#     typer.echo(f"  OpenAI Model: {cfg.openai.model}")
#     typer.echo(f"  TTS Model: {cfg.tts.model}")
#     typer.echo(f"  ASR Model: {cfg.asr.model}")
#     typer.echo(f"  Server: {cfg.server.host}:{cfg.server.port}")


def main() -> None:
    """命令行入口"""
    app()


if __name__ == "__main__":
    main()
