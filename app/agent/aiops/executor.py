"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from app.utils.logger import logger
from app.agent.aiops.state import PlanExecuteState
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

from app.model.factory import chat_model

async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state["plan"]
    if not plan or len(plan) == 0:
        logger.info("本次执行计划为空")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    # 执行任务
    try:
        # 取出所有的工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)
        # todo MCP工具
        mcp_tools = []

        all_tools = local_tools + mcp_tools
        model_with_tools = chat_model.bind_tools(all_tools)
        tool_node = ToolNode(tools=all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

        你可以使用各种工具来完成任务。对于每个步骤：
        1. 理解步骤的目标
        2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
        3. 调用工具获取信息
        4. 返回执行结果

        注意：
        - 如果工具调用失败，请说明失败原因
        - 不要编造数据，只返回实际获取的信息
        - 执行结果要清晰、准确
        - 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        llm_response = await model_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")
            messages.append(llm_response)

            # tool_node节点调用工具
            tool_message = await tool_node.ainvoke({"messages": messages})
            messages.extend(tool_message["messages"])

            final_response = await model_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, "content") else final_response
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, "content") else llm_response

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回结果
        return {
            "plan": plan[1:],
            "past_steps": (str(task), result),
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }