# 服务端支持 MCP 与 IoT 的实现计划书

## 目标与范围

- 目标：服务端通过 WebSocket 与客户端建立连接后，能够发现并调用客户端本地 MCP 工具，同时能发现并控制 IoT 设备。
- 范围：仅描述服务端侧需要实现的功能与交互流程；客户端实现已在现有项目中完成。
- 传输：以 WebSocket 为主。

## 关键交互概览（WebSocket）

1. **连接建立与 hello 握手**
   - 客户端发出 hello，包含 `features: { mcp: true }` 和 `transport: "websocket"`。
   - 服务端回复 hello（`type: "hello"`, `transport: "websocket"`），完成握手。
2. **MCP 会话初始化**
   - 服务端发送 MCP JSON-RPC `initialize` 请求，携带 capabilities（可选 vision 服务信息）。
   - 客户端回复 `initialize` 结果，包含协议版本、capabilities 与 serverInfo。
3. **MCP 工具发现**
   - 服务端发送 `tools/list` 请求。
   - 客户端返回工具清单（支持分页 cursor）。
4. **IoT 设备发现与状态同步**
   - 客户端主动上报 IoT descriptors 与 states。
   - 服务端缓存 descriptors/states，并允许后续控制与状态更新。
5. **服务端发起控制**
   - MCP：服务端发 `tools/call` 调用具体工具。
   - IoT：服务端发 `type: "iot"` 命令列表，客户端执行后回传最新状态。

## 功能清单（服务端必须实现）

### 1) WebSocket 通信与会话管理

- 建立 WS 连接并处理 hello 握手。
- 维护连接状态、重连策略与超时处理。
- 记录设备标识（Device-Id/Client-Id）以区分多设备连接。
- 处理服务端内部会话与客户端连接的映射。

### 2) MCP 协议支持（JSON-RPC 2.0）

- **initialize**
  - 请求格式：`{ jsonrpc:"2.0", id, method:"initialize", params:{ capabilities } }`
  - capabilities 可包含 vision 服务的 `url`/`token`（用于本地摄像头工具）。
  - 需要处理客户端返回的 `protocolVersion` 与 `serverInfo` 并记录。
- **tools/list**
  - 请求格式：`{ jsonrpc:"2.0", id, method:"tools/list", params:{ cursor? } }`
  - 需支持分页 cursor，循环拉取完整工具列表并缓存。
- **tools/call**
  - 请求格式：`{ jsonrpc:"2.0", id, method:"tools/call", params:{ name, arguments } }`
  - 需处理正常结果与错误返回，统一透传或转换为服务端内部结构。

### 3) MCP 消息封装（外层 envelope）

所有 MCP JSON-RPC 消息需包装为：
```json
{
  "session_id": "<session-id>",
  "type": "mcp",
  "payload": { ...JSON-RPC... }
}
```
备注：WebSocket 模式下 `session_id` 可能为空字符串，可由服务端忽略或自行维护。

### 4) IoT 上报与命令

- **上报格式（客户端 -> 服务端）**
  - descriptors：
    ```json
    { "type":"iot", "update": true, "descriptors": [ { "name": "...", "properties": {...}, "methods": {...} } ] }
    ```
  - states：
    ```json
    { "type":"iot", "update": true, "states": [ { "name":"...", "state": { ... } } ] }
    ```
  - descriptors 会“逐条发送”，服务端需支持多条累积。
- **控制格式（服务端 -> 客户端）**
  ```json
  {
    "type":"iot",
    "commands":[
      { "name":"Lamp", "method":"TurnOn", "parameters": {} }
    ]
  }
  ```
  - 可附带 `session_id` 字段以保持一致性。

### 5) 缓存与路由

- MCP 工具清单缓存：按设备 ID 维护工具列表，供后续路由调用。
- IoT 设备描述与状态缓存：支持全量与增量更新。
- 将“服务端业务请求”路由为 MCP 调用或 IoT 命令。

### 6) 错误处理与观测性

- JSON-RPC 错误统一处理：记录 `id` 与 `error.message`。
- 失败重试策略：对 `tools/list` 可重试；对 `tools/call` 需上报失败。
- 日志与追踪：记录每次 MCP/IoT 调用的请求/响应摘要。


## 里程碑与交付

1. **M1：握手与基础会话**
   - WS 连接、hello 回应、基础设备注册。
2. **M2：MCP 初始化与工具发现**
   - `initialize`、`tools/list` 完成与缓存。
3. **M3：MCP 工具调用**
   - `tools/call` 请求-响应链路稳定。
4. **M4：IoT 上报与控制**
   - descriptors/states 接收与命令下发。


## 验收标准

- WS 连接后，服务端能获取 MCP 工具列表并成功调用至少一个工具。
- IoT 设备描述与状态能被服务端完整缓存，命令下发可成功执行并收到状态更新。
- 断线重连后，工具与设备缓存能恢复或自动重新拉取。
*** End Patch} خرج to=functions.apply_patch commentary code  天天中彩票软件 code
