"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import json
from fastapi import APIRouter, HTTPException
from app.model.response import ApiResponse, SessionInfoResponse
from sse_starlette.sse import EventSourceResponse
from app.agent.mcp_client import format_exception_chain
from loguru import logger
from app.service.rag_agent_service import rag_agent_service

from app.model.request import ChatRequest, ClearRequest

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """快速对话接口
    {
        "code": 200,
        "message": "success",
        "data": {
            "success": true,
            "answer": "回答内容",
            "errorMessage": null
        }
    }

    Args:
        request: 对话请求

    Returns:
        统一格式的对话响应
    """
    session_id = request.id
    question = request.question

    try:
        logger.info(f"会话 {session_id}收到快速对话请求: {question}")
        answer = await rag_agent_service.query(
            question,
            session_id=session_id
        )

        logger.info(f"快速对话请求 {session_id} 完成")
        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None
            }
        }

    except Exception as e:
        logger.error(f"快速对话请求{session_id}响应失败: {str(e)}")
        return {
            "code": 500,
            "messages": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e)
            }
        }

@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口（基于 RAG Agent，SSE）

    返回 SSE 格式，data 字段为 JSON：

    工具调用事件:
    event: message
    data: {"type":"tool_call","data":{"tool":"工具名","status":"start|end","input":{...}}}

    内容流式事件:
    event: message
    data: {"type":"content","data":"内容块"}

    完成事件:
    event: message
    data: {"type":"done","data":{"answer":"完整答案","tool_calls":[...]}}

    Args:
        request: 对话请求

    Returns:
        SSE 事件流
    """
    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}")

    async def event_generator():
        try:
            async for event in rag_agent_service.query_stream(question=request.question, session_id=request.id):
                chat_type = event.get("type", None)
                data = event.get("data", None)

                if chat_type == "content":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "data": data,
                        }, ensure_ascii=False)
                    }
                elif chat_type == "complete":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "complete",
                            "data": data,
                        }, ensure_ascii=False)
                    }
                else:
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "data": str(data),
                        })
                    }
            logger.info(f"[会话 {request.id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {format_exception_chain(e)}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "data": str(e)
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())

@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    try:
        is_success = await rag_agent_service.cleanup(request.session_id)

        return ApiResponse(
            status="success" if is_success else "failed",
            message="会话清空成功" if is_success else "会话清空失败",
            data=None
        )

    except Exception as e:
        return ApiResponse(
            status="failed" if is_success else "failed",
            message=str(e),
            data=None
        )

@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """查询会话历史

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    try:
        session_info = await rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(session_info),
            history=session_info,
        )

    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

