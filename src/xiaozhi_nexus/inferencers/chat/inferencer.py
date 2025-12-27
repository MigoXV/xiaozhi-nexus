from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, AsyncIterator, Optional, List, Dict

import httpx
from openai import OpenAI, AsyncOpenAI

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false


@dataclass
class OpenAIChatInferencer:
    """
    基于 OpenAI Chat Completions API 的 LLM 推理器

    接口设计:
    - input: str (用户消息)
    - output: Iterator[str] (流式响应文本块)

    支持同步 (__call__) 和异步 (astream) 两种调用方式
    """

    # OpenAI 配置
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"

    # 生成配置
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful assistant."

    # 对话历史配置
    max_history: int = 20  # 最大历史消息数量（不含 system prompt）

    # SSL 配置
    verify_ssl: bool = True

    # 内部状态
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)
    _async_client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)
    _messages: List[Dict[str, str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化 OpenAI 客户端"""
        http_client = None
        async_http_client = None

        if not self.verify_ssl:
            http_client = httpx.Client(verify=False)
            async_http_client = httpx.AsyncClient(verify=False)

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
        )

        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=async_http_client,
        )

        # 初始化对话历史
        self._messages = []

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """构建完整的消息列表（包含 system prompt 和历史）"""
        messages: List[Dict[str, str]] = []

        # 添加 system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史消息
        messages.extend(self._messages)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _truncate_history(self) -> None:
        """截断历史消息，保持在 max_history 限制内"""
        if len(self._messages) > self.max_history:
            # 保留最近的消息
            self._messages = self._messages[-self.max_history:]

    def reset(self) -> None:
        """重置对话历史"""
        self._messages = []

    def __call__(self, text: str) -> Iterator[str]:
        """
        同步流式 LLM 推理

        Args:
            text: 用户输入文本

        Yields:
            响应文本块（增量形式）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        messages = self._build_messages(text)

        # 调用 OpenAI Chat Completions API（流式）
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        # 收集完整响应用于历史记录
        full_response = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # 更新对话历史
        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()

    async def astream(self, text: str) -> AsyncIterator[str]:
        """
        异步流式 LLM 推理

        Args:
            text: 用户输入文本

        Yields:
            响应文本块（增量形式）
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        messages = self._build_messages(text)

        # 调用 OpenAI Chat Completions API（异步流式）
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        # 收集完整响应用于历史记录
        full_response = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # 更新对话历史
        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()


@dataclass
class OpenAIChatInferencerAsync:
    """
    纯异步版本的 OpenAI Chat 推理器

    适用于异步应用场景，提供更好的性能
    """

    # OpenAI 配置
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"

    # 生成配置
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful assistant."

    # 对话历史配置
    max_history: int = 20

    # SSL 配置
    verify_ssl: bool = True

    # 内部状态
    _client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)
    _messages: List[Dict[str, str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化异步 OpenAI 客户端"""
        async_http_client = None

        if not self.verify_ssl:
            async_http_client = httpx.AsyncClient(verify=False)

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=async_http_client,
        )

        self._messages = []

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """构建完整的消息列表"""
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_input})

        return messages

    def _truncate_history(self) -> None:
        """截断历史消息"""
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

    def reset(self) -> None:
        """重置对话历史"""
        self._messages = []

    async def __call__(self, text: str) -> AsyncIterator[str]:
        """
        异步流式 LLM 推理

        Args:
            text: 用户输入文本

        Yields:
            响应文本块（增量形式）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        messages = self._build_messages(text)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        full_response = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()


# 便捷别名
ChatInferencer = OpenAIChatInferencer
AsyncChatInferencer = OpenAIChatInferencerAsync
