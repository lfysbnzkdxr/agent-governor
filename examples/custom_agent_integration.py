"""Example: 在自定义 Agent 框架中嵌入 Governor 治理能力.

演示场景:
    一个手写的 ReAct 风格 Agent（Plan → Act → Observe → Reflect 循环），
    无需继承任何基类、无需修改编排逻辑，仅在 LLM 调用点嵌入 Governor，
    即可获得预算管控、模型白名单、失败降级、多租户归因、审计日志。

核心观点:
    Governor 是调用层中间件，与编排层完全解耦。
    不管你用 LangGraph / CrewAI / 自己写的 for 循环，集成方式都一样：
    把 litellm.acompletion() 换成 governor.execute()。

Prerequisites:
    pip install agent-governor
    export OPENAI_API_KEY="your-key"
"""

import asyncio

from agent_governor import (
    Governor,
    GovernorConfig,
    GovernorError,
    TenantPolicy,
)


# ============================================================
# 第一部分: 用户自定义的 Agent 框架（与 Governor 无关）
# ============================================================


class MyReActAgent:
    """一个极简的 ReAct Agent，展示自定义编排逻辑.

    编排流程: plan → execute → reflect（最多 max_steps 轮）
    这个类代表"用户自己的 Agent 框架"，可以是任何结构。
    """

    def __init__(self, governor: Governor, *, max_steps: int = 3) -> None:
        # Governor 作为依赖注入，Agent 不关心治理细节
        self._governor = governor
        self._max_steps = max_steps

    async def run(self, goal: str, *, tenant_id: str = "") -> str:
        """执行 Agent 主循环."""
        context: list[dict[str, str]] = []
        final_answer = ""

        for step in range(1, self._max_steps + 1):
            print(f"\n{'='*50}")
            print(f"  Step {step}/{self._max_steps}")
            print(f"{'='*50}")

            # --- Plan: 让 LLM 决定下一步行动 ---
            plan = await self._llm_call(
                system="你是一个任务规划器。根据目标和已有上下文，决定下一步行动。"
                "如果信息足够，直接输出 [DONE] 加最终答案。",
                user=f"目标: {goal}\n已有上下文: {context}",
                step_name=f"plan_step{step}",
                tenant_id=tenant_id,
            )
            print(f"  [Plan] {plan[:100]}...")

            # --- 判断是否完成 ---
            if "[DONE]" in plan:
                final_answer = plan.replace("[DONE]", "").strip()
                break

            # --- Act: 执行行动（这里简化为再次调用 LLM 模拟工具执行）---
            observation = await self._llm_call(
                system="你是一个工具执行器。模拟执行以下行动并返回结果。",
                user=f"行动: {plan}",
                step_name=f"act_step{step}",
                tenant_id=tenant_id,
                model="gpt-4o-mini",  # 执行步骤用便宜模型
            )
            print(f"  [Observe] {observation[:100]}...")

            # --- 记录上下文 ---
            context.append({"step": str(step), "plan": plan, "observation": observation})

        return final_answer or "达到最大步数，未得出最终答案。"

    async def _llm_call(
        self,
        *,
        system: str,
        user: str,
        step_name: str,
        tenant_id: str = "",
        model: str = "",
    ) -> str:
        """统一的 LLM 调用入口 — 唯一的 Governor 集成点.

        关键: 不管 Agent 内部有多少个节点/步骤，
        只要通过这个函数调 LLM，就自动获得治理能力。
        """
        return await self._governor.execute(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,  # 空字符串则使用 default_model
            name=step_name,
            tenant_id=tenant_id,
        )


# ============================================================
# 第二部分: 配置 Governor 治理策略
# ============================================================


def create_governor() -> Governor:
    """创建带有多租户治理策略的 Governor 实例."""
    config = GovernorConfig(
        # --- 全局策略 ---
        budget_usd=2.0,  # 总预算 $2
        models=["gpt-4o", "gpt-4o-mini"],  # 模型白名单
        fallback_model="gpt-4o-mini",  # 主模型失败时降级
        default_model="gpt-4o",
        max_concurrency=5,
        max_retries=2,
        # --- 多租户配额 ---
        tenant_policies={
            "team-research": TenantPolicy(
                budget_usd=1.5,
                max_concurrency=3,
                allowed_models=["gpt-4o", "gpt-4o-mini"],
            ),
            "team-support": TenantPolicy(
                budget_usd=0.5,
                max_concurrency=2,
                allowed_models=["gpt-4o-mini"],  # 客服团队只能用便宜模型
            ),
        },
    )
    return Governor(config=config)


# ============================================================
# 第三部分: 运行演示
# ============================================================


async def main() -> None:
    governor = create_governor()

    # --- 场景 1: 研究团队使用 Agent ---
    print("\n" + "=" * 60)
    print("  场景 1: 研究团队 Agent（预算 $1.5，可用 gpt-4o）")
    print("=" * 60)

    agent = MyReActAgent(governor, max_steps=3)
    answer = await agent.run(
        goal="分析 2024 年 LLM Agent 框架的发展趋势",
        tenant_id="team-research",
    )
    print(f"\n  [Final Answer] {answer[:200]}")

    # --- 场景 2: 客服团队使用 Agent ---
    print("\n" + "=" * 60)
    print("  场景 2: 客服团队 Agent（预算 $0.5，仅限 gpt-4o-mini）")
    print("=" * 60)

    agent_support = MyReActAgent(governor, max_steps=2)
    answer2 = await agent_support.run(
        goal="总结用户退款政策的要点",
        tenant_id="team-support",
    )
    print(f"\n  [Final Answer] {answer2[:200]}")

    # --- 场景 3: 预算耗尽时的优雅处理 ---
    print("\n" + "=" * 60)
    print("  场景 3: 预算耗尽 → 自动拒绝并给出提示")
    print("=" * 60)

    try:
        # 故意用超大预算消耗触发限制
        for i in range(20):
            await governor.execute(
                messages=[{"role": "user", "content": f"重复任务 {i}"}],
                name="budget_stress_test",
                tenant_id="team-support",  # 只有 $0.5 预算
            )
    except GovernorError as e:
        # GovernorError 是所有治理异常的基类
        # 包括 BudgetExceededError / TenantBudgetExceededError / ModelNotAllowedError 等
        print(f"\n  [Governor 拦截] {type(e).__name__}: {e}")

    # --- 查看治理报告 ---
    print("\n" + "=" * 60)
    print("  治理报告")
    print("=" * 60)

    stats = governor.get_stats()
    print(f"  总调用次数: {stats.get('total_calls', 'N/A')}")
    print(f"  总花费: ${stats.get('total_cost_usd', 0):.6f}")
    print(f"  预算剩余: ${stats.get('budget_remaining', 0):.6f}")

    # 多租户账单
    report = governor.get_billing_report()
    print(f"\n  多租户账单: {report}")

    # 审计日志摘要
    summary = governor.audit_logger.get_summary()
    print(f"\n  审计记录数: {summary['total_calls']}")
    print(f"  审计总花费: ${summary['total_cost_usd']:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
