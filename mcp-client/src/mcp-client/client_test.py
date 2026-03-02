#Python 内置的异步编程库
import asyncio
#用于管理 MCP 客户端会话（但目前我们先不连接 MCP 服务器）。
from mcp import ClientSession
#自动管理资源，确保程序退出时正确关闭 MCP 连接。
from contextlib import AsyncExitStack
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

#定义 MCPClient 类
class MCPClient:
    def __init__(self):
        """初始化客户端"""
        self.session = None    #存储与 MCP 服务器的会话对象.暂时不连接 MCP 服务器，后续可以修改来真正连接。
        #创建资源管理器,管理 MCP 客户端的资源，确保程序退出时可以正确释放资源。
        self.exit_stack = AsyncExitStack()
        # 读取 OpenAI API Key
        self.openai_api_key = os.getenv("DS_API_KEY")
        # 读取 BASE YRL
        self.base_url = os.getenv("DS_BASE_URL")
        # 读取 model
        self.model = os.getenv("DS_MODEL")
        if self.openai_api_key is None:
            raise ValueError("❌ 未找到 API Key，请在 .env 文件中设置 API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

    # 存储与 MCP 服务器的会话对象.暂时不连接 MCP 服务器，后续可以修改来真正连接。
    # 创建资源管理器,管理 MCP 客户端的资源，确保程序退出时可以正确释放资源。
    # async def connect_to_mock_server(self):
    #     """模拟 MCP 服务器的连接（暂不连接真实服务器）"""
    #     print("✅ MCP 客户端已初始化，但未连接到服务器")

    # 发送用户输入到大模型
    async def process_query(self, query: str) -> str:
        """调用 OpenAI API 处理用户查询"""
        messages = [{"role": "system", "content": "你是一个智能助手，帮助用户回答问题。"},
                    {"role": "user", "content": query}]
        """
        lambda是匿名函数，是一种简洁的函数写法。
                        如果不用lambda，就需要：
            def make_completion_func(client, model, messages):
                def create_completion():
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                )
                return create_completion
            completion_func = make_completion_func(self.client, self.model, messages) 
    
            response = await asyncio.get_event_loop().run_in_executor(
            None,
            completion_func
        )
        这里，lambda后面直接是“：”，代表这个匿名函数无参数。
        """
        try:
            # 调用 OpenAI API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ 发生错误：{e}")



    # 多轮对话：这是客户端的核心功能，允许用户通过命令行与 AI 模型交互。
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！输入 'quit' 退出")
        while True:
            try:
                user_input = input("用户：")
                if user_input.lower() == "quit":
                    print("已退出")
                    break
                response = await self.process_query(user_input)
                print(f"\n🤖 AI: {response}")
            except Exception as e:
                print(f"❌ 发生错误：{e}")
    #清理资源：关闭所有通过 AsyncExitStack 管理的异步资源（比如会话、文件句柄等），
    #确保程序退出时不会留下未释放的资源和正确关闭 MCP 连接（尽管目前没有真正的连接）。
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
# 主函数：创建 MCPClient 实例。
async def main():
    # 创建一个MCPClient 客户端实例
    client = MCPClient()
    try:
        # 运行交互式聊天循环
        await client.chat_loop()
    finally:
        # 在退出时清理资源
        await client.cleanup()
if __name__ == "__main__":
    asyncio.run(main())

"""之后在终端中的UV创建的mcp-client虚拟环境中输入：uv run client.py运行这个Python文件"""







