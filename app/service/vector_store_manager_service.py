"""向量存储管理器 - 封装 milvus VectorStore 操作"""
import uuid
from typing import List

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.utils.logger import logger
from app.utils.config_handler_util import milvus_conf
from app.model.factory import embedding_model

class VectorStoreManagerService:
    """向量存储管理器"""
    def __init__(self):
        """初始化向量存储管理器"""

        # 建立milvus连接
        # 必须在 pymilvus / langchain_milvus 访问 Collection 之前建立连接，
        # 否则会出现 ConnectionNotExistException: should create connection first.
        self.vector_store = MilvusClient(
            uri=milvus_conf["milvus_url"],
            db_name=milvus_conf["db_name"],
            auto_id=False, # 不能自动生成id
        )

        # 嵌入式模型
        self.embedding_model = embedding_model

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        if not documents:
            logger.warning(f"导入milvus数据库时文档列表为空")
            return []
        try:
            # 向量化
            vectors = embedding_model.embed_documents(documents)

            # 为每个文档生成唯一 id（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # 组装数据
            data = [
                {
                    "id": ids[i],
                    "vector": vectors[i],
                    "text": documents[i].page_content,
                    "source": documents[i].metadata["source"],
                    "chunk_id": i
                }
                for i in range(len(vectors))
            ]

            # 数据添加到数据库中
            self.vector_store.upsert(
                collection_name=milvus_conf["db_name"],
                data=data,
            )

            return ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise


    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'
            delete_result = self.vector_store.delete(
                collection_name=milvus_conf["db_name"],
                expr=expr,
            )

            deleted_count = len(delete_result)

            logger.info(f"删除数据成功 count: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning("删除数据失败")
            raise


    def similarity_search(self, query: str, k: int = 3) -> List[dict]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """

        try:
            vector_query = self.embedding_model.embed_text(str(query))
            results = self.vector_store.search(
                collection_name=milvus_conf["db_name"],
                query=vector_query,
            )
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(results[0])}")
            return results[0]
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

# 全局单例
vector_store_manager_service = VectorStoreManagerService()