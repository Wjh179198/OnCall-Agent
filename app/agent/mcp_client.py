"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""
import asyncio
from typing import Optional, Any, Union, Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent

from app.utils.logger import logger

# 全局 MCP 客户端（延迟初始化）
_mcp_client: Optional[MultiServerMCPClient] = None


DEFAULT_MCP_CLIENT = {
    "my_mcp_server": {
        "transport": "streamable-http",
        "url": "http://localhost:8003/mcp",
    },
    "my_monitor_server": {
        "transport": "streamable-http",
        "url": "http://localhost:8004/mcp",
    }
}

def format_exception_chain(exc: BaseException) -> str:
    """展开 ExceptionGroup / TaskGroup，便于日志定位真实子异常。"""
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions is not None:
        lines = [str(exc)]
        for i, sub in enumerate(sub_exceptions):
            lines.append(f"  [{i}] {format_exception_chain(sub)}")
        return "\n".join(lines)
    msg = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{msg}\n  caused by: {format_exception_chain(cause)}"
    return msg

async def load_mcp_tools_safe(
    client: MultiServerMCPClient,
) -> tuple[list[Union[BaseTool, Any]], str | None]:
    """加载 MCP 工具；失败时返回空列表与可读错误信息，不向上抛出。"""
    try:
        tools = await client.get_tools()
        logger.info(f"获取MCP工具列表成功len: {len(tools)}")
        return tools, None
    except Exception as e:
        logger.error(f"获取MCP工具列表失败: {e}")
        return [], format_exception_chain(e)


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """MCP 工具调用重试拦截器

    当工具调用失败时，使用指数退避策略自动重试。
    如果所有重试都失败，返回包含错误信息的结果而不是抛出异常。

    MCPToolCallRequest 结构：
    - name: str - 工具名称
    - args: dict[str, Any] - 工具参数
    - server_name: str - 服务器名称

    Args:
        request: MCP 工具调用请求
        handler: 实际的工具调用处理器
        max_retries: 最大重试次数（默认3次）
        delay: 初始延迟时间（秒，默认1秒）

    Returns:
        CallToolResult: 工具调用结果或错误信息
    """
    last_error = ""

    for attempt in range(max_retries):
        try:
            logger.info(
                f"调用 MCP 工具: {request.name} "
                f"(服务器: {request.server_name}, 第 {attempt + 1}/{max_retries} 次尝试)"
            )
            result = await handler(request)
            logger.info(f"MCP 工具 {request.name} 调用成功")
            return result

        except Exception as e:
            logger.warning(f"调用MCP工具: {request.name} "
                           f"服务器: {request.server_name}, 第 {attempt + 1}/{max_retries}次失败")
            logger.warning(f"开始第{attempt + 1}重新调用MCP工具: {request.name}")

            if attempt < max_retries - 1:
                last_error = format_exception_chain(e)
                wait_time = delay * (2 ** attempt)  # 指数退避
                logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)

        # 所有重试都失败，返回错误结果而不是抛出异常
        error_msg = f"工具 {request.name} 在 {max_retries} 次重试后仍然失败: {str(last_error)}"
        logger.error(error_msg)
        return CallToolResult(
            content=[TextContent(type="text", text=error_msg)],
            isError=True
        )


async def get_mcp_client(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带重试拦截器）

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）

    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client

    if force_new:
        client = _create_mcp_client(servers or DEFAULT_MCP_CLIENT, tool_interceptors)
        return client

    if _mcp_client is None:
        mcp_client = _create_mcp_client(servers or DEFAULT_MCP_CLIENT, tool_interceptors)

    return mcp_client


async def get_mcp_client_with_retry(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """
    获取或初始化带重试功能的 MCP 客户端

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    重试拦截器会自动添加到拦截器列表的开头

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在重试拦截器之后添加）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers=servers,
        tool_interceptors=interceptors,
        force_new=force_new
    )


def _create_mcp_client(
    servers: Dict[str, Dict[str, str]],
    tool_interceptors: Optional[List] = None
) -> MultiServerMCPClient:
    """
    创建 MCP 客户端实例

    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    kwargs: Dict[str, Any] = {}
    if tool_interceptors is not None:
        kwargs["tool_interceptors"] = tool_interceptors

    return MultiServerMCPClient(servers, **kwargs)


