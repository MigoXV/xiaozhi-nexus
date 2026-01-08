from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

import httpx
from openai import AsyncOpenAI, OpenAI

from xiaozhi_nexus.mcp.get_date import get_date
from xiaozhi_nexus.mcp.tools_list import tools_list

from ..utils import format_text_for_log
from .text_normalization import normalize_full_text, stream_text_normalization

logger = logging.getLogger(__name__)


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

    # 工具扩展
    tools_provider: Optional[Callable[[], List[Any]]] = None
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]] = None

    # 内部状态
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)
    _async_client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)
    _messages: List[Dict[str, str]] = field(
        default_factory=list, init=False, repr=False
    )

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
        messages.append({"role": "user", "content": user_input + " /no_think"})

        return messages

    def _truncate_history(self) -> None:
        """截断历史消息，保持在 max_history 限制内"""
        if len(self._messages) > self.max_history:
            # 保留最近的消息
            self._messages = self._messages[-self.max_history :]

    def reset(self) -> None:
        """重置对话历史"""
        self._messages = []

    def _execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if self.tool_executor:
            try:
                result = self.tool_executor(tool_name, arguments)
                if result is not None:
                    return result
            except Exception as exc:
                return {"error": f"Tool execution failed: {exc}"}

        if tool_name == "get_date":
            time_zone, timestamp = get_date()
            return {
                "timezone": time_zone,
                "timestamp": timestamp,
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _get_tools(self) -> List[Any]:
        if self.tools_provider:
            try:
                tools = self.tools_provider()
                if tools is not None:
                    return tools if tools else []
            except Exception as exc:
                logger.warning("Tools provider failed: %s", exc)
                return tools_list
        return tools_list

    def _get_tool_required_map(self, tools: List[Any]) -> Dict[str, List[str]]:
        required_map: Dict[str, List[str]] = {}
        for tool in tools:
            name = None
            params = None
            if isinstance(tool, dict):
                func = tool.get("function")
                if isinstance(func, dict):
                    name = func.get("name")
                    params = func.get("parameters")
                else:
                    name = tool.get("name")
                    params = tool.get("parameters")
            else:
                func = getattr(tool, "function", None)
                if func is not None:
                    name = getattr(func, "name", None)
                    params = getattr(func, "parameters", None)
                else:
                    name = getattr(tool, "name", None)
                    params = getattr(tool, "parameters", None)
            if not name or not isinstance(params, dict):
                continue
            required = params.get("required")
            if isinstance(required, list) and all(
                isinstance(item, str) for item in required
            ):
                required_map[str(name)] = required
        return required_map

    def _apply_required_defaults(
        self, tool_name: str, tool_args: Any, tools: List[Any]
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = tool_args if isinstance(tool_args, dict) else {}
        required = self._get_tool_required_map(tools).get(tool_name)
        if not required:
            return args
        for field in required:
            if field not in args:
                args[field] = ""
        return args

    def stream_chat(self, text: str) -> Iterator[str]:
        """
        同步流式 LLM 推理（支持工具调用）

        Args:
            text: 用户输入文本

        Yields:
            响应文本块（增量形式）
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        input_preview, input_len = format_text_for_log(text)
        logger.info(
            "Chat input text_len=%d text=%s",
            input_len,
            input_preview,
        )

        messages = self._build_messages(text)

        # 调用 OpenAI Chat Completions API（流式）
        tools = self._get_tools()
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,  # type: ignore
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            request_kwargs["tools"] = tools

        response_itr = self._client.chat.completions.create(**request_kwargs)

        # 收集完整响应和工具调用
        content_chunks = []
        tool_calls = []

        for chunk in response_itr:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta is None:
                continue

            # 收集内容
            if delta.content:
                content_chunks.append(delta.content)

            # 收集工具调用
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # 确保有足够的空间
                    while len(tool_calls) <= tool_call.index:
                        tool_calls.append(
                            {
                                "id": None,
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                            }
                        )

                    target = tool_calls[tool_call.index]
                    if tool_call.id:
                        target["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        target["function"]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        target["function"]["arguments"] += tool_call.function.arguments

        # 处理内容输出
        full_response = ""
        if content_chunks:
            content_text = "".join(content_chunks)
            content_itr = stream_text_normalization(iter([content_text]))
            for content in content_itr:
                full_response += content
                yield content

        # 如果有工具调用，执行工具并重新请求
        if tool_calls:
            logger.info(f"Tool calls detected: {tool_calls}")
            tools_for_args = tools or self._get_tools()

            # 将助手的消息（包含工具调用）添加到历史
            assistant_message = {
                "role": "assistant",
                "content": full_response if full_response else None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)

            # 执行每个工具调用
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_args = self._apply_required_defaults(
                    tool_name, tool_args, tools_for_args
                )
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                result = self._execute_tool(tool_name, tool_args)

                # 添加工具结果到消息
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            # 重新请求获取最终响应
            tools = self._get_tools()
            request_kwargs = {
                "model": self.model,
                "messages": messages,  # type: ignore
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            if tools:
                request_kwargs["tools"] = tools
            response_itr = self._client.chat.completions.create(**request_kwargs)

            # 收集第二次响应
            full_response = ""
            content_itr = (
                chunk.choices[0].delta.content
                for chunk in response_itr
                if chunk.choices and chunk.choices[0].delta.content
            )
            content_itr = stream_text_normalization(content_itr)
            for content in content_itr:
                full_response += content
                yield content

        output_preview, output_len = format_text_for_log(full_response)
        logger.info(
            "Chat output text_len=%d text=%s",
            output_len,
            output_preview,
        )

        # 更新对话历史
        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()

    def chat(self, text: str) -> str:
        """
        同步完整 LLM 推理

        Args:
            text: 用户输入文本

        Returns:
            完整响应文本
        """
        if self._client is None:
            raise RuntimeError("Client not initialized")

        input_preview, input_len = format_text_for_log(text)
        logger.info(
            "Chat input text_len=%d text=%s",
            input_len,
            input_preview,
        )

        messages = self._build_messages(text)

        # 调用 OpenAI Chat Completions API（非流式）
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        full_response = response.choices[0].message.content
        full_response = normalize_full_text(full_response)

        output_preview, output_len = format_text_for_log(full_response)
        logger.info(
            "Chat output text_len=%d text=%s",
            output_len,
            output_preview,
        )

        # 更新对话历史
        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()

        return full_response

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

        input_preview, input_len = format_text_for_log(text)
        logger.info(
            "Chat input text_len=%d text=%s",
            input_len,
            input_preview,
        )

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

        output_preview, output_len = format_text_for_log(full_response)
        logger.info(
            "Chat output text_len=%d text=%s",
            output_len,
            output_preview,
        )

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
    _messages: List[Dict[str, str]] = field(
        default_factory=list, init=False, repr=False
    )

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
            self._messages = self._messages[-self.max_history :]

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

        input_preview, input_len = format_text_for_log(text)
        logger.info(
            "Chat input text_len=%d text=%s",
            input_len,
            input_preview,
        )

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

        output_preview, output_len = format_text_for_log(full_response)
        logger.info(
            "Chat output text_len=%d text=%s",
            output_len,
            output_preview,
        )

        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": full_response})
        self._truncate_history()


# 便捷别名
ChatInferencer = OpenAIChatInferencer
AsyncChatInferencer = OpenAIChatInferencerAsync
