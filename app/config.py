"""
配置管理类

使用 Pydantic Settings 实现类型安全的配置管理
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:

    # 应用配置
    app_name = os.getenv("APP_NAME")
    app_version: str = "1.0.0"
    debug: bool = False
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    # 模型API配置
    deepseek_api_key  = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL") # LLM模型
    silicon_api_key = os.getenv("SILICONFLOW_API_KEY")
    silicon_base_url = os.getenv("SILICON_BASE_URL") # 向量模型

    # milvus配置
    milvus_host = os.getenv("MILVUS_HOST")
    milvus_port = os.getenv("MILVUS_PORT")
    milvus_timeout = os.getenv("MILVUS_TIMEOUT") # 超时时间

    # RAG配置
    rag_top_k = os.getenv("RAG_TOP_K")
    rag_model = os.getenv("RAG_MODEL")

    # 文档切分配置
    chunk_max_size = os.getenv("CHUNK_MAX_SIZE")
    chunk_overlap = os.getenv("CHUNK_OVERLAP")


config = Settings()  # 全局单例
