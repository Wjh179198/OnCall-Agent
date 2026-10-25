"""向量检索服务模块"""
from typing import Dict, Any, List
from app.service.vector_store_manager_service import vector_store_manager_service
from app.utils.logger import logger
from app.utils.config_handler_util import rag_conf


class SearchResult:
    """搜索结果类"""

    def __init__(
            self,
            id: str,
            content: str,
            score: float,
            metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }

class VectorSearchService:
    """向量检索服务 - 负责从 milvus 中搜索相似向量"""

    def __init__(self):
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    @staticmethod
    def search_similar_documents(query: str, top_k: int = rag_conf["rag_top_k"]) -> List[SearchResult]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回最相似的K个结果

        Returns:
            List[SearchResult]: 搜索结果列表

        Raises:
            RuntimeError: 搜索失败时抛出
        """

        try:
            hits = vector_store_manager_service.similarity_search(query, top_k)
            query_result = []
            for hit in hits:
                entity = hit.get("entity") or {}
                result = SearchResult(
                    id=hit["id"],
                    content=entity.get("text", ""),
                    score=hit["distance"],
                    metadata={
                        "source": entity.get("source", ""),
                        "extension": entity.get("extension", ""),
                        "file_name": entity.get("file_name", ""),
                    }
                )
                query_result.append(result)

            logger.info(f"搜索完成, 找到 {len(query_result)} 个相似文档")
            return query_result
        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e

# 全局单例
vector_search_service = VectorSearchService()