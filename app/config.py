"""
配置管理类

使用 Pydantic Settings 实现类型安全的配置管理
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, # .env里大写 这里小写 自动匹配
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OnCallAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # 模型API配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "deepseek-flash" # LLM模型
    siliconflow_api_key: str = "" # 向量模型

    # milvus配置
    milvus_host: str = "134.175.135.230"
    milvus_port: int = 19530
    milvus_timeout: int = 10000 # 超时时间

    # RAG配置
    rag_top_k: int = 3
    rag_model: str = "deepseek-flash"

    # 文档切分配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100


config = Settings()  # 全局单例
