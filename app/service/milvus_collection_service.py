"""Milvus collection 初始化 - 保证目标 collection 存在且 schema 与读写字段一致"""
from pymilvus import DataType, MilvusClient

from app.utils.config_handler_util import milvus_conf
from app.utils.logger import logger


def ensure_collection(client: MilvusClient) -> None:
    """
    确保目标 collection 存在

    schema 与 VectorStoreManagerService 写入/查询/删除使用的字段保持一致：
    id / vector / text / source / extension / file_name

    Args:
        client: 已建立连接的 MilvusClient
    """
    collection_name = milvus_conf["collection_name"]

    if client.has_collection(collection_name):
        logger.info(f"collection 已存在: {collection_name}")
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=milvus_conf["id_max_length"],
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=milvus_conf["vector_dim"],
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=milvus_conf["text_max_length"],
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=milvus_conf["source_max_length"],
    )
    schema.add_field(
        field_name="extension",
        datatype=DataType.VARCHAR,
        max_length=milvus_conf["extension_max_length"],
    )
    schema.add_field(
        field_name="file_name",
        datatype=DataType.VARCHAR,
        max_length=milvus_conf["file_name_max_length"],
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"collection 创建完成: {collection_name}")
