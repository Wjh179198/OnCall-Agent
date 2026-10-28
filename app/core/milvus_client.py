"""Milvus 客户端工厂模块"""
from types import UnionType
from typing import Any

from pymilvus import MilvusClient, DataType, MilvusException
from app.utils.logger import logger
from app.utils.config_handler_util import milvus_conf
from app.service.milvus_collection_service import ensure_collection


DENSE_VECTOR_TYPES = {
    DataType.FLOAT_VECTOR,
    DataType.FLOAT16_VECTOR,
    DataType.BFLOAT16_VECTOR,
    DataType.INT8_VECTOR,
}

class MilvusClientManager:
    """Milvus 客户端管理器"""
    def __init__(self):
        self.client: MilvusClient | None = None

    def connect(self) -> MilvusClient | None:
        """Milvus客户端初始化"""

        if self.client:
            logger.info("Milvus客户端已连接")
            return self.client
        try:
            host = milvus_conf["milvus_url"]
            db_name = milvus_conf["db_name"]

            logger.info(f"milvus正在连接至 {host}:{db_name}")
            self.client: MilvusClient = MilvusClient(
                uri=host,
                db_name=db_name,
            )
            logger.info(f"milvus连接成功 {host}:{db_name}")

            collection_name = milvus_conf["collection_name"]
            # 检查是否创建了collection
            if not self.client.has_collection(collection_name):
                logger.info(f"collection '{collection_name}' 不存在，正在创建...")
                ensure_collection(self.client)
                logger.info(f"成功创建 collection '{collection_name}'")
            else:
                logger.info(f"collection '{collection_name}'已被创建")

                # 检查向量维度是否匹配
                collection_describe = self.client.describe_collection(collection_name)
                for field in collection_describe.get("fields", []):
                    if field.get("name") != "vector":
                        continue
                    if field.get("type") not in DENSE_VECTOR_TYPES:
                        logger.warning(f"vector不是稠密向量字段，实际类型: {field.get('type')}")
                    dim = field.get("params", {}).get("dim")
                    if dim is None or int(dim) != milvus_conf["vector_dim"]:
                        logger.warning("当前collection的向量维度不匹配 需要重新创建")
                        logger.info(f"正在删除旧collection {collection_name}")
                        self.client.drop_collection(collection_name)
                        logger.info("旧collection删除成功, 开始创建新的collection")
                        ensure_collection(self.client)
                    else:
                        logger.info("向量维度匹配!")

            return self.client

        except Exception as e:
            logger.error(f"Milvus客户端初始化失败 {e}")
            self.close()
            return None

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        if not self.client:
            logger.error("检测collection失败 由于未建立客户端连接")
            return False
        return self.client.has_collection(milvus_conf["collection_name"])

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self.client is None:
                return False

            # 尝试列出 connections
            self.client.list_collections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭 Milvus 客户端，释放连接资源（可重复调用）"""
        if self.client is None:
            return

        try:
            self.client.close()
            logger.info("Milvus 客户端已关闭")
        except Exception as e:
            logger.error(f"关闭 Milvus 客户端失败: {e}")
        finally:
            # 无论关闭是否成功都清空引用，后续 connect() 可以重新建立连接
            self.client = None

# 全局单例
milvus_manager = MilvusClientManager()