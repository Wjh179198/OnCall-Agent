"""
Replanner 节点：重新规划或生成最终响应
基于 LangGraph 官方教程实现
"""
from textwrap import dedent
from typing import List, Dict, Any

from langchain.chat_models.base import _ConfigurableModel, BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from app.agent.aiops.state import PlanExecuteState
from app.agent.aiops.utils import format_tools_description
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.utils.logger import logger

from app.model.factory import chat_model


class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(description="对用户的最终响应")

class Act(BaseModel):
    """重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应"""
    )
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）"
    )


# Replanner 提示词
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：** 
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")

    MAX_STEPS = 8
    if len(past_steps) > MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制 {MAX_STEPS}，强制生成最终响应")
        return await _generate_response(state, chat_model)

    try:
        # 合并工具
        # 本地工具列表
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # todo MCP工具
        mcp_tools = []
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        all_tools = local_tools + mcp_tools
        tools_description = format_tools_description(all_tools)

    except Exception as e:
        logger.error(f"获取本地工具类列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 格式化已执行的步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])

    if plan and len(plan) > 0:
        logger.info("还有计划, 进一步评估下一步的执行计划")
        replanner_model = replanner_prompt | chat_model.with_structured_output(Act)

        try:
            messages = [
                ("user", f"用户的原始问题为: {input_text}"),
                ("user", f"已执行步骤: {steps_summary}"),
                ("user", f"剩余步骤: {".".join(plan)}"),
                ("user", f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）")
            ]

            replanner_response = await replanner_model.ainvoke({
                "messages": messages,
                "tools_description": tools_description,
            })

            # 解析结果
            if isinstance(replanner_response, Act):
                action = replanner_response.action
                new_steps = replanner_response.new_steps
            else:
                action = replanner_response.get("action", "")
                new_steps = replanner_response.get("new_steps", "")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state, chat_model)

            elif action == "replan":
                logger.info(f"决定生成新的执行计划: {len(new_steps)}")
                if len(new_steps) > 0:
                    return {
                        "new_steps": new_steps,
                    }
                else:
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    return {}
            else:
                logger.info("原有方案合理, 继续按原来的计划执行")

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划")
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, chat_model)


async def _generate_response(state: PlanExecuteState, llm: BaseChatModel | _ConfigurableModel) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    user_input = state.get("input", "")

    # 格式化执行历史
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    response_model = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {user_input}"),
            ("user", f"执行历史记录: \n{execution_history}"),
            ("user", "请根据以上的内容生成最终响应")
        ]

        final_response_obj = await response_model.ainvoke({"messages": messages})

        if isinstance(final_response_obj, Response):
            final_response = final_response_obj.response
        else:
            final_response = final_response_obj.get("response", "")

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        # 生成简单的后备响应
        fallback_response = f"""# 任务执行结果

## 原始任务
{user_input}

## 执行的步骤
{_format_simple_steps(past_steps)}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        return {"response": fallback_response}

def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
