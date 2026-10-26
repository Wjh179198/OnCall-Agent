"""知识检索工具 - 从向量数据库中检索相关信息"""
from typing import Tuple, List
from app.service.vector_search_service import SearchResult, vector_search_service
from langchain_core.tools import tool

from app.utils.logger import logger

@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[SearchResult]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[SearchResult]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query: {query}")

        query_result = vector_search_service.search_similar_documents(query)
        if not query_result or len(query_result) == 0:
            logger.info(f"未检索到相关内容 query: {query}")
            return "没有找到相关信息", []

        context = format_query_result(query_result)

        logger.info(f"检索到{len(query_result)}条相关内容")
        return context, query_result
    except Exception as e:
        logger.error(f"检索工具调用失败 {e}")
        return "", []


def format_query_result(query_result: List[SearchResult]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        query_result: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, part in enumerate(query_result, 1):
        metadata = part.metadata
        res_source = metadata.get("source", "未知来源")

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        formatted += f"\n来源: {res_source}"
        formatted += f"\n内容:\n{part.content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)


