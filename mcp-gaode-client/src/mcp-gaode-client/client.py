import json
import asyncio
import re
import sys
from typing import Optional
from contextlib import AsyncExitStack
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


load_dotenv()
ds_api_key = os.getenv("DS_API_KEY")
ds_base_url = os.getenv("DS_BASE_URL")

def format_tools_for_llm(tool) -> str:
    """对tool进行格式化
    Returns:
        格式化之后的tool描述
    """
    args_desc = []
    if "properties" in tool.inputSchema:
        for param_name, param_info in tool.inputSchema["properties"].items():
            arg_desc = (
                f"- {param_name}: {param_info.get('description', 'No description')}"
            )
            if param_name in tool.inputSchema.get("required", []):
                arg_desc += " (required)"
            args_desc.append(arg_desc)

    return f"Tool: {tool.name}\nDescription: {tool.description}\nArguments:\n{chr(10).join(args_desc)}"


class Client:
    def __init__(self):
        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
        self.is_connected = False

        # 重点1：模型客户端，可自行适配不同模型和其对应的base_url、apikey和模型名称
        self.client = AsyncOpenAI(
            base_url=ds_base_url,
            api_key=ds_api_key,
        )
        self.model = "deepseek-chat"
        self.messages = []

    # server连接函数
    async def connect_server(self, server_config):
        async with self._lock:
            # 重点2：提取servers_config.json配置文件中基于sse模式mcp server的远程连接url
            url = server_config["mcpServers"]["amap-maps"]["url"]
            print(f"尝试连接到: {url}")

            self._exit_stack = AsyncExitStack()
            sse_cm = sse_client(url)
            streams = await self._exit_stack.enter_async_context(sse_cm)
            print("SSE 流已获取。")

            session_cm = ClientSession(streams[0], streams[1])
            self.session = await self._exit_stack.enter_async_context(session_cm)
            print("ClientSession 已创建。")

            await self.session.initialize()
            print("Session 已初始化。")

            # 获取并存储mcp server的工具列表
            response = await self.session.list_tools()
            self.tools = {tool.name: tool for tool in response.tools}
            print(f"成功获取 {len(self.tools)} 个工具:")
            for name, tool in self.tools.items():
                print(f"  - {name}: {tool.description[:50]}...")  # 打印部分描述

            print("连接成功并准备就绪。")

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools

        tools_description = "\n".join([format_tools_for_llm(tool) for tool in tools])
        # 定义系统提示
        system_prompt = (
            "你是一个有用的助手，可以使用以下工具：\n\n"
            f"{tools_description}\n"
            "请根据用户的问题选择合适的工具。如果不需要使用任何工具，请直接回复。\n\n"
            "重要提示：当你需要使用工具时，必须仅使用以下确切的JSON对象格式回复，不要包含其他任何内容：\n"
            "{\n"
            '    "tool": "工具名称",\n'
            '    "arguments": {\n'
            '        "参数名称": "参数值"\n'
            "    }\n"
            "}\n\n"
            "不允许使用 ```json 标记\n"
            "在收到工具响应后：\n"
            "1. 将原始数据转换为自然、对话式的回复\n"
            "2. 保持回复简洁但信息丰富\n"
            "3. 专注于最相关的信息\n"
            "4. 使用用户问题中的适当上下文\n"
            "5. 避免简单地重复原始数据\n\n"
            "请仅使用上面明确定义的工具。"
        )
        self.messages.append({"role": "system", "content": system_prompt})

    async def disconnect(self):
        """关闭 Session 和连接。"""
        async with self._lock:
            await self._exit_stack.aclose()

    async def chat(self, prompt, role="user"):
        """与LLM进行交互"""
        self.messages.append({"role": role, "content": prompt})

        # 初始化 LLM API 调用
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        llm_response = response.choices[0].message.content
        return llm_response

    # mcp server的工具调用函数
    async def execute_tool(self, llm_response: str):
        import json
        try:
            pattern = r"```json\n(.*?)\n?```"
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                llm_response = match.group(1)
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                # result = await self.session.call_tool(tool_name, tool_args)
                response = await self.session.list_tools()
                tools = response.tools

                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        print(f"[提示]：正在调用工具 {tool_call['tool']}")
                        result = await self.session.call_tool(
                            tool_call["tool"], tool_call["arguments"]
                        )

                        if isinstance(result, dict) and "progress" in result:
                            progress = result["progress"]
                            total = result["total"]
                            percentage = (progress / total) * 100
                            print(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                        # print(f"[执行结果]: {result}")
                        return f"Tool execution result: {result}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print(error_msg)
                        return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("MCP 客户端启动")
        print("输入 /bye 退出")

        while True:
            prompt = input(">>> ").strip()
            if "/bye" in prompt.lower():
                break

            response = await self.chat(prompt)
            self.messages.append({"role": "assistant", "content": response})

            result = await self.execute_tool(response)
            while result != response:
                response = await self.chat(result, "system")
                self.messages.append(
                    {"role": "assistant", "content": response}
                )
                result = await self.execute_tool(response)
            print(response)


# 加载servers_config.json配置文件函数
def load_server_config(config_file):
    with open(config_file) as f:
        return json.load(f)


async def main():
    try:
        server_config = load_server_config("servers_config.json")
        client = Client()
        await client.connect_server(server_config)
        await client.chat_loop()
    except Exception as e:
        print(f"主程序发生错误: {type(e).__name__}: {e}")
    finally:
        # 无论如何，最后都要尝试断开连接并清理资源
        print("\n正在关闭客户端...")
        await client.disconnect()
        print("客户端已关闭。")


if __name__ == '__main__':
    # 我要去新郑机场出差，请你查询附近5km的酒店，为我安排行程
    asyncio.run(main())