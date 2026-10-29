"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.utils.config_handler_util import app_conf
from loguru import logger
from app.api import chat, healthy, file, aiops
from app.core.milvus_client import milvus_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {app_conf["app_name"]} v{app_conf["app_version"]} 启动中...")
    logger.info(f"📝 环境: {'开发' if app_conf["app_debug"] else '生产'}")
    logger.info(f"🌐 监听地址: http://{app_conf["host"]}:{app_conf["port"]}")
    logger.info(f"📚 API 文档: http://{app_conf["host"]}:{app_conf["port"]}/docs")
    
    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")
    
    logger.info("=" * 60)
    
    yield
    
    # 关闭时执行
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {app_conf["app_name"]} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=app_conf["app_name"],
    version=app_conf["app_version"],
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(healthy.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件（使用绝对路径，避免依赖启动时的工作目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = static_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {app_conf["app_name"]} API",
        "version": app_conf["app_version"],
        "docs": "/docs"
    }


if __name__ == "__main__":
    import sys

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=app_conf["host"],
        port=app_conf["port"],
        reload=app_conf["app_debug"],
        log_level="info",
        # psycopg 异步模式要求 SelectorEventLoop，而 uvicorn>=0.36 在 Windows
        # 非 reload 模式下会强制使用 ProactorEventLoop（忽略 event loop policy），
        # 因此这里显式指定自定义的 loop 工厂
        loop="app.core.event_loop:selector_loop_factory" if sys.platform == "win32" else "auto",
    )
