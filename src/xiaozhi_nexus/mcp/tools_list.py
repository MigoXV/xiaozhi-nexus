from openai.types.chat.chat_completion_function_tool import ChatCompletionFunctionTool
from openai.types.shared.function_definition import FunctionDefinition

weather = ChatCompletionFunctionTool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="获取一个地点的天气，用户应该先提供一个地点。",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名，例如：北京，上海，广州",
                }
            },
            "required": ["location"],
        },
    ),
)

date = ChatCompletionFunctionTool(
    type="function",
    function=FunctionDefinition(
        name="get_date",
        description="获取当前的日期和时间。",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，例如：Asia/Shanghai，America/New_York",
                },
                "timestamp": {
                    "type": "string",
                    "description": "可选的时间戳，格式为ISO 8601，例如：2023-10-01T12:00:00Z",
                },
            },
            "required": [],
        },
    ),
)

tools_list = [date]
