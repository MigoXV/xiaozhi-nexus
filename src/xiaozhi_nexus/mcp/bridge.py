from __future__ import annotations

import asyncio
import itertools
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai.types.chat.chat_completion_function_tool import (
    ChatCompletionFunctionTool,
)
from openai.types.shared.function_definition import FunctionDefinition

logger = logging.getLogger(__name__)

_TOOL_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_tool_name(name: str, fallback: str) -> str:
    cleaned = _TOOL_NAME_PATTERN.sub("_", name).strip("_")
    if not cleaned:
        cleaned = fallback
    if len(cleaned) > 64:
        cleaned = cleaned[:64]
    return cleaned


def _unique_tool_name(name: str, existing: set[str], fallback: str) -> str:
    candidate = _sanitize_tool_name(name, fallback=fallback)
    if candidate not in existing:
        return candidate
    suffix = 1
    while True:
        candidate = _sanitize_tool_name(f"{name}_{suffix}", fallback=fallback)
        if candidate not in existing:
            return candidate
        suffix += 1


def _build_tool_definition(
    name: str, description: str | None, parameters: Dict[str, Any] | None
) -> ChatCompletionFunctionTool:
    schema = (
        parameters
        if isinstance(parameters, dict)
        else {"type": "object", "properties": {}}
    )
    if isinstance(schema, dict) and "type" not in schema:
        schema = {"type": "object", **schema}
    if isinstance(schema, dict):
        required = schema.get("required")
        if not isinstance(required, list):
            schema = {**schema, "required": []}
    return ChatCompletionFunctionTool(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=description or "",
            parameters=schema,
        ),
    )


class McpError(RuntimeError):
    def __init__(self, payload: Any) -> None:
        message = "MCP error"
        if isinstance(payload, dict):
            message = str(payload.get("message") or message)
        super().__init__(message)
        self.payload = payload


class McpClientBridge:
    def __init__(
        self,
        send_json: Callable[[dict[str, Any]], None],
        loop: asyncio.AbstractEventLoop,
        session_id: str = "",
        request_timeout_s: float = 20.0,
    ) -> None:
        self._send_json = send_json
        self._loop = loop
        self._session_id = session_id
        self._request_timeout_s = request_timeout_s
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = itertools.count(1)
        self._lock = threading.Lock()
        self._tools: list[ChatCompletionFunctionTool] = []
        self._tool_name_map: dict[str, str] = {}
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] | None = None

    def handle_message(self, message: dict[str, Any]) -> None:
        rpc = message.get("payload")
        if not isinstance(rpc, dict):
            return
        if "id" not in rpc:
            return
        request_id = rpc.get("id")
        future = self._pending.get(request_id)
        if future is None and isinstance(request_id, str):
            try:
                future = self._pending.get(int(request_id))
            except ValueError:
                future = None
        if future is None or future.done():
            return
        if "error" in rpc and rpc["error"] is not None:
            future.set_exception(McpError(rpc["error"]))
            return
        if "result" in rpc:
            future.set_result(rpc["result"])

    async def initialize(self, capabilities: dict[str, Any] | None = None) -> None:
        result = await self._request(
            "initialize", {"capabilities": capabilities or {}}
        )
        if isinstance(result, dict):
            self.protocol_version = str(result.get("protocolVersion") or "")
            self.server_info = result.get("serverInfo")

    async def refresh_tools(self) -> None:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = await self._request("tools/list", params)
            if isinstance(result, dict):
                tool_items = result.get("tools")
                if isinstance(tool_items, list):
                    tools.extend(tool_items)
                cursor = (
                    result.get("nextCursor")
                    or result.get("next_cursor")
                    or result.get("cursor")
                )
            if not cursor:
                break

        self._update_tools(tools)

    async def initialize_and_refresh_tools(
        self, capabilities: dict[str, Any] | None = None
    ) -> None:
        try:
            await self.initialize(capabilities)
        except Exception as exc:
            logger.warning("MCP initialize failed: %s", exc)
            return

        for attempt in range(3):
            try:
                await self.refresh_tools()
                return
            except Exception as exc:
                logger.warning("MCP tools/list failed (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(1)

    def get_tools(self) -> List[ChatCompletionFunctionTool]:
        with self._lock:
            return list(self._tools)

    def has_tool(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._tool_name_map

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self.call_tool(tool_name, arguments), self._loop
        )
        return future.result(timeout=self._request_timeout_s + 1.0)

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        with self._lock:
            remote_name = self._tool_name_map.get(tool_name)
        if not remote_name:
            return {"error": f"Unknown MCP tool: {tool_name}"}
        result = await self._request(
            "tools/call", {"name": remote_name, "arguments": arguments}
        )
        if isinstance(result, dict):
            return result
        return {"result": result}

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = next(self._next_id)
        future: asyncio.Future[Any] = self._loop.create_future()
        self._pending[request_id] = future
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self._send_json(
            {
                "session_id": self._session_id,
                "type": "mcp",
                "payload": payload,
            }
        )
        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout_s)
        finally:
            self._pending.pop(request_id, None)

    def _update_tools(self, tools: list[dict[str, Any]]) -> None:
        tool_defs: list[ChatCompletionFunctionTool] = []
        tool_map: dict[str, str] = {}
        existing_names: set[str] = set()
        raw_names: list[str] = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            raw_names.append(name)
            description = tool.get("description")
            parameters = (
                tool.get("inputSchema")
                or tool.get("input_schema")
                or tool.get("parameters")
            )
            tool_name = _unique_tool_name(name, existing_names, fallback="mcp_tool")
            existing_names.add(tool_name)
            tool_defs.append(_build_tool_definition(tool_name, description, parameters))
            tool_map[tool_name] = name

        with self._lock:
            self._tools = tool_defs
            self._tool_name_map = tool_map
        if raw_names:
            logger.info("MCP tools received: %s", ", ".join(raw_names))


@dataclass(frozen=True)
class IotToolTarget:
    device: str
    method: str


class IoTRegistry:
    def __init__(
        self,
        send_json: Callable[[dict[str, Any]], None],
        loop: asyncio.AbstractEventLoop,
        session_id: str = "",
        response_timeout_s: float = 5.0,
    ) -> None:
        self._send_json = send_json
        self._loop = loop
        self._session_id = session_id
        self._response_timeout_s = response_timeout_s
        self._lock = threading.Lock()
        self._descriptors: dict[str, dict[str, Any]] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._tools: list[ChatCompletionFunctionTool] = []
        self._tool_map: dict[str, IotToolTarget] = {}
        self._state_waiters: dict[str, list[asyncio.Future[Any]]] = {}

    def handle_update(self, message: dict[str, Any]) -> None:
        if message.get("update") is False:
            return
        descriptors = message.get("descriptors")
        states = message.get("states")
        update_tools = False
        descriptor_names: list[str] = []
        state_names: list[str] = []

        with self._lock:
            if isinstance(descriptors, list):
                for descriptor in descriptors:
                    if not isinstance(descriptor, dict):
                        continue
                    name = str(descriptor.get("name") or "").strip()
                    if not name:
                        continue
                    self._descriptors[name] = descriptor
                    update_tools = True
                    descriptor_names.append(name)

            if isinstance(states, list):
                for state in states:
                    if not isinstance(state, dict):
                        continue
                    name = str(state.get("name") or "").strip()
                    if not name:
                        continue
                    self._states[name] = state.get("state") or {}
                    state_names.append(name)

        if update_tools:
            self._rebuild_tools()
        if descriptor_names:
            logger.info("IoT descriptors received: %s", ", ".join(descriptor_names))
        if state_names:
            logger.info("IoT states received: %s", ", ".join(state_names))

        if isinstance(states, list):
            for state in states:
                if not isinstance(state, dict):
                    continue
                name = str(state.get("name") or "").strip()
                if not name:
                    continue
                self._notify_waiters(name, state.get("state") or {})

    def get_tools(self) -> List[ChatCompletionFunctionTool]:
        with self._lock:
            return list(self._tools)

    def has_tool(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._tool_map

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self.call_tool(tool_name, arguments), self._loop
        )
        return future.result(timeout=self._response_timeout_s + 1.0)

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        with self._lock:
            target = self._tool_map.get(tool_name)
        if not target:
            return {"error": f"Unknown IoT tool: {tool_name}"}

        payload = {
            "type": "iot",
            "commands": [
                {
                    "name": target.device,
                    "method": target.method,
                    "parameters": arguments,
                }
            ],
        }
        if self._session_id:
            payload["session_id"] = self._session_id
        self._send_json(payload)

        state = await self._wait_for_state(target.device)
        response: dict[str, Any] = {
            "device": target.device,
            "method": target.method,
            "status": "sent",
        }
        if state is not None:
            response["status"] = "updated"
            response["state"] = state
        else:
            response["status"] = "timeout"
            response["state"] = self.get_state(target.device)
        return response

    def get_state(self, device_name: str) -> dict[str, Any] | None:
        with self._lock:
            return self._states.get(device_name)

    async def _wait_for_state(self, device_name: str) -> dict[str, Any] | None:
        future = self._loop.create_future()
        self._state_waiters.setdefault(device_name, []).append(future)
        try:
            result = await asyncio.wait_for(
                future, timeout=self._response_timeout_s
            )
            if isinstance(result, dict):
                return result
            return None
        except asyncio.TimeoutError:
            return None
        finally:
            waiters = self._state_waiters.get(device_name, [])
            if future in waiters:
                waiters.remove(future)
            if not waiters:
                self._state_waiters.pop(device_name, None)

    def _notify_waiters(self, device_name: str, state: dict[str, Any]) -> None:
        waiters = self._state_waiters.get(device_name, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(state)
        if waiters:
            self._state_waiters.pop(device_name, None)

    def _rebuild_tools(self) -> None:
        tool_defs: list[ChatCompletionFunctionTool] = []
        tool_map: dict[str, IotToolTarget] = {}
        existing_names: set[str] = set()

        with self._lock:
            descriptors = list(self._descriptors.values())

        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            device_name = str(descriptor.get("name") or "").strip()
            if not device_name:
                continue
            methods = descriptor.get("methods") or {}
            if isinstance(methods, dict):
                method_items = methods.items()
            elif isinstance(methods, list):
                method_items = [(method, {}) for method in methods]
            else:
                method_items = []

            for method_name, method_info in method_items:
                method_name_str = str(method_name or "").strip()
                if not method_name_str:
                    continue
                description = ""
                parameters: dict[str, Any] | None = None
                if isinstance(method_info, dict):
                    description = str(method_info.get("description") or "")
                    parameters = (
                        method_info.get("parameters")
                        or method_info.get("inputSchema")
                        or method_info.get("input_schema")
                        or method_info.get("schema")
                    )
                    if parameters is None and isinstance(method_info.get("properties"), dict):
                        parameters = {
                            "type": "object",
                            "properties": method_info.get("properties"),
                            "required": method_info.get("required") or [],
                        }
                tool_key = f"iot_{device_name}_{method_name_str}"
                tool_name = _unique_tool_name(tool_key, existing_names, fallback="iot_tool")
                existing_names.add(tool_name)
                if not description:
                    description = f"IoT command {device_name} {method_name_str}"
                tool_defs.append(_build_tool_definition(tool_name, description, parameters))
                tool_map[tool_name] = IotToolTarget(
                    device=device_name, method=method_name_str
                )

        with self._lock:
            self._tools = tool_defs
            self._tool_map = tool_map


class ClientToolRouter:
    def __init__(
        self,
        mcp_bridge: McpClientBridge | None = None,
        iot_registry: IoTRegistry | None = None,
    ) -> None:
        self._mcp_bridge = mcp_bridge
        self._iot_registry = iot_registry

    def attach_mcp(self, bridge: McpClientBridge) -> None:
        self._mcp_bridge = bridge

    def attach_iot(self, registry: IoTRegistry) -> None:
        self._iot_registry = registry

    def get_tools(
        self, base_tools: List[ChatCompletionFunctionTool] | None
    ) -> List[ChatCompletionFunctionTool]:
        tools = list(base_tools or [])
        if self._mcp_bridge:
            tools.extend(self._mcp_bridge.get_tools())
        if self._iot_registry:
            tools.extend(self._iot_registry.get_tools())
        return tools

    def execute(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self._mcp_bridge and self._mcp_bridge.has_tool(tool_name):
            try:
                return self._mcp_bridge.call_tool_sync(tool_name, arguments)
            except Exception as exc:
                return {"error": f"MCP tool call failed: {exc}"}
        if self._iot_registry and self._iot_registry.has_tool(tool_name):
            try:
                return self._iot_registry.call_tool_sync(tool_name, arguments)
            except Exception as exc:
                return {"error": f"IoT command failed: {exc}"}
        return None
