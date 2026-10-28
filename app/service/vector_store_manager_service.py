"""向量存储管理器 - 封装 Milvus 的写入、删除、检索操作"""
import uuid
from typing import List

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.core.factory import embedding_model
from app.service.milvus_collection_service import ensure_collection
from app.utils.config_handler_util import milvus_conf, rag_conf
from app.utils.logger import logger


class VectorStoreManagerService:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        # db_name 是 Milvus 的数据库名，collection_name 才是集合名
        self.vector_store = MilvusClient(
            uri=milvus_conf["milvus_url"],
            db_name=milvus_conf["db_name"],
        )
        self.collection_name = milvus_conf["collection_name"]

        # 嵌入式模型
        self.embedding_model = embedding_model

        # 首次使用时自动建表，保证 schema 与下面的读写字段一致
        ensure_collection(self.vector_store)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        if not documents:
            logger.warning("导入milvus数据库时文档列表为空")
            return []
        try:
            # 向量化
            chunks = [doc.page_content for doc in documents]
            vectors = self.embedding_model.embed_documents(chunks)

            # 用 (source, 序号, 内容) 生成确定性 id：
            # 同一个文件重复导入会覆盖同一批记录，而不是产生重复数据
            ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{doc.metadata.get('source', '')}|{i}|{doc.page_content}",
                    )
                )
                for i, doc in enumerate(documents)
            ]

            # 组装数据（字段与 collection schema 一致）
            data = [
                {
                    "id": ids[i],
                    "vector": vectors[i],
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "extension": doc.metadata.get("extension", ""),
                    "file_name": doc.metadata.get("file_name", ""),
                }
                for i, doc in enumerate(documents)
            ]

            # 数据添加到数据库中
            self.vector_store.upsert(
                collection_name=self.collection_name,
                data=data,
            )
            logger.info(f"写入 milvus 成功: {len(data)} 条 -> {self.collection_name}")

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
            # source 是顶层字段，与写入时保持一致；
            # 使用 filter_params 传参，避免 Windows 路径中的反斜杠/引号破坏表达式
            delete_result = self.vector_store.delete(
                collection_name=self.collection_name,
                filter="source == {path}",
                filter_params={"path": file_path},
            )

            deleted_count = delete_result.get("delete_count", 0)

            logger.info(f"删除数据成功 count: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除数据失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = rag_conf["rag_top_k"]) -> List[dict]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[dict]: Milvus 检索结果（每项包含 id / distance / entity）
        """
        try:
            vector_query = self.embedding_model.embed_query(str(query))
            results = self.vector_store.search(
                collection_name=self.collection_name,
                # MilvusClient.search 的 data 是"一批向量"，单条查询也必须包一层
                data=[vector_query],
                limit=k,
                output_fields=["text", "source", "extension", "file_name"],
            )
            hits = results[0] if results else []
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(hits)}")
            return hits
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            raise


# 全局单例
vector_store_manager_service = VectorStoreManagerService()
