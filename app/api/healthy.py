from typing import Any

from fastapi import APIRouter
from app.core.milvus_client import milvus_manager
from app.utils.config_handler_util import app_conf
from app.utils.logger import logger

router = APIRouter()


@router.get("/health")
async def health_check():

    """健康检查接口
    检查服务状态和数据库连接状态

    Returns:
        JSONResponse: 健康检查结果
    """
    health_data: dict[str, Any] = {
        "service": app_conf["app_name"],
        "version": app_conf["app_version"],
        "status": "healthy",
    }

    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status = "connected" if milvus_healthy else "disconnected"
        milvus_message = "milvus连接正常" if milvus_healthy else "milvus连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message,
        }

    except Exception as e:
        logger.error(f"milvus健康检测失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"milvus健康检测失败: {str(e)}",
        }

    overall_health = "healthy"
    status_code = 200

    if health_data["milvus"]["status"] != "connected":
        overall_health = "unhealthy"
        status_code = 503
        health_data["error"] = "数据库不可用"

    health_data["status"] = overall_health


