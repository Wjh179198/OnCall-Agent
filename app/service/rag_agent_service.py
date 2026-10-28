"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""
import asyncio
import os
import sys
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Any, Dict, AsyncGenerator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.chat_models.base import init_chat_model
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.utils.logger import logger
from app.utils.config_handler_util import rag_conf
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

# psycopg 的异步模式不支持 Windows 默认的 ProactorEventLoop，
# 必须在事件循环创建之前切换到 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 读取配置信息
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

DB_URL = "postgresql://langchain_user:123456@127.0.0.1:5432/langchain_db?sslmode=disable"


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent服务 - 使用LangChain构建"""

    def __init__(self, streaming: bool = True):
        """
        初始化Agent服务

        Args:
            streaming: 是否启用流式输出
        """

        self.model_name = rag_conf["chat_model_name"]
        self.model_provider = rag_conf["chat_model_provider"]
        self.system_prompt = self._build_system_prompt()

        self.model = init_chat_model(
            model=self.model_name,
            model_provider=self.model_provider,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            streaming=streaming, # 启用流式输出
            extra_body={
                "thinking": {
                    "type": rag_conf["chat_thinking_type"]
                },
            }
        )

        # 定义基础工具
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # MCP客户端
        self.mcp_tools: list = []

        # PostgreSQL 检查点(用于会话管理)
        # AsyncPostgresSaver 必须在运行中的事件循环里创建，因此延迟到 _initialize_agent 中初始化
        self.checkpointer: AsyncPostgresSaver | None = None
        self._checkpointer_pool: AsyncConnectionPool | None = None

        # Agent初始化 (在异步方法中完成)
        self.agent = None
        self._agent_initialized = False
        self._init_lock = asyncio.Lock()

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    def _build_system_prompt(self):
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """

        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。
                
            工作原则:
                1. 理解用户需求，选择合适的工具来完成任务
                2. 当需要获取实时信息或专业知识时，主动使用相关工具
                3. 基于工具返回的结果提供准确、专业的回答
                4. 如果工具无法提供足够信息，请诚实地告知用户
                
                回答要求:
                - 保持友好、专业的语气
                - 回答简洁明了，重点突出
                - 基于事实，不编造信息
                - 如有不确定的地方，明确说明

                请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 checkpointer 和 MCP 工具）"""
        async with self._init_lock:
            if self._agent_initialized:
                return

            # todo 加载MCP工具

            all_tools = self.tools + self.mcp_tools

            # AsyncPostgresSaver 需要运行中的事件循环，只能在异步方法里创建
            await self._initialize_checkpointer()

            self.agent = create_agent(
                model=self.model,
                tools=all_tools,
                checkpointer=self.checkpointer,
            )

            self._agent_initialized = True

            if all_tools:
                tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
                logger.info(f"可用工具列表: {', '.join(tool_names)}")

    async def _initialize_checkpointer(self) -> None:
        """初始化 PostgreSQL checkpointer（连接池 + 自动建表）"""
        if self.checkpointer is not None:
            return

        if not DB_URL:
            raise RuntimeError(
                "未配置 PostgreSQL checkpointer，请在 .env 中设置 POSTGRES_URI，"
                "例如 postgresql://user:password@localhost:5432/rag_agent"
            )

        # 如果服务在已经运行的事件循环里才被导入，这里给出明确提示，
        # 避免 psycopg 连接池卡 30 秒后抛 PoolTimeout
        if sys.platform == "win32" and isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
            raise RuntimeError(
                "psycopg 异步模式不支持 Windows 的 ProactorEventLoop，"
                "请在应用启动前设置 asyncio.WindowsSelectorEventLoopPolicy()"
            )

        pool = AsyncConnectionPool(
            conninfo=DB_URL,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

        try:
            await pool.open()
            checkpointer = AsyncPostgresSaver(pool)
            # 首次运行时自动创建 checkpoint 相关表，已存在时只执行迁移
            await checkpointer.setup()
        except Exception:
            await pool.close()
            raise

        self._checkpointer_pool = pool
        self.checkpointer = checkpointer
        logger.info("PostgreSQL checkpointer 初始化完成")

    async def close(self) -> None:
        """关闭 PostgreSQL 连接池并释放 Agent 资源"""
        if self._checkpointer_pool is not None:
            await self._checkpointer_pool.close()
            self._checkpointer_pool = None

        self.checkpointer = None
        self.agent = None
        self._agent_initialized = False
        logger.info("RAG Agent 资源已释放")


    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{e}"
            )
            raise

    async def query_stream(
            self,
            question: str,
            session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content', None)
                    text_content = ""
                    if isinstance(content_blocks, str):
                        text_content = content_blocks
                    elif content_blocks and isinstance(content_blocks, list):
                        text_content = "".join(
                            context.get("text", "") for context in content_blocks
                            if isinstance(context, dict) and context.get("type") == "text"
                        )
                    if text_content:
                        yield {
                            "type": "content",
                            "data": text_content,
                            "node": node_name,
                        }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}"
            )
            yield {"type": "error", "data": e}


    async def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 PostgreSQL checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "..."}]
        """
        try:
            await self._initialize_agent()

            config = {"configurable": {"thread_id": session_id}}
            state = self.agent.aget_state(config)

            if state is None:
                return []

            messages = state.values.get("messages", [])

            history = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue
                history.append(
                    {
                        "role": role,
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            return history
        except Exception as e:
            logger.error(f"获取会话消息失败{e}")
            
            return []

    async def cleanup(self, session_id: str) -> bool:
        """
        清除指定session_id的会话历史

        Args:
            session_id: 会话id (即thread_id)

        Returns:
                是否成功
        """

        try:
            await self._initialize_agent()

            await self.checkpointer.adelete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清除会话 {session_id} 失败: {e}")
            return False


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
            



