"""LLMJudgeGate - uses an LLM to evaluate task result quality."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from agent_governor.models import Task, TaskResult, ValidationResult

logger = structlog.get_logger(__name__)

# Type for the LLM call function: takes messages, returns content string
LLMCallFn = Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]]

DEFAULT_JUDGE_PROMPT = """\
You are a quality evaluator. Rate the following response on a scale of 0.0 to 1.0.

Criteria: {criteria}

Original task: {task_description}

Response to evaluate:
{response}

Reply with ONLY a JSON object: {{"score": <float 0-1>, "reason": "<brief explanation>"}}
"""


class LLMJudgeGate:
    """Uses an LLM as a judge to score task result quality.

    The judge LLM evaluates the result against given criteria and returns
    a numeric score. If the score is below the threshold, validation fails.

    Example:
        async def my_llm_call(messages):
            return await governor.execute(messages, model="gpt-4o-mini")

        gate = LLMJudgeGate(
            llm_call=my_llm_call,
            criteria="accuracy and completeness",
            threshold=0.7,
        )
        result = await gate.check(task, task_result)
    """

    def __init__(
        self,
        llm_call: LLMCallFn,
        *,
        criteria: str = "quality, accuracy, and completeness",
        threshold: float = 0.7,
        judge_prompt: str = "",
        model: str = "",
    ) -> None:
        """Initialize LLMJudgeGate.

        Args:
            llm_call: Async function that takes messages and returns content string.
            criteria: Evaluation criteria description.
            threshold: Minimum score to pass (0.0 - 1.0).
            judge_prompt: Custom judge prompt template. Uses {criteria},
                {task_description}, {response} placeholders.
            model: Model name for logging purposes.
        """
        self._llm_call = llm_call
        self._criteria = criteria
        self._threshold = threshold
        self._judge_prompt = judge_prompt or DEFAULT_JUDGE_PROMPT
        self._model = model

    @property
    def name(self) -> str:
        return "llm_judge_gate"

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        """Use LLM to judge the quality of the result."""

        # Build task description from messages
        task_desc = ""
        if task.messages:
            last_msg = task.messages[-1]
            task_desc = last_msg.get("content", "")[:500]

        # Build judge prompt
        prompt = self._judge_prompt.format(
            criteria=self._criteria,
            task_description=task_desc,
            response=result.content[:2000],  # Truncate long responses
        )

        try:
            judge_response = await self._llm_call([{"role": "user", "content": prompt}])

            # Parse judge response
            score, reason = self._parse_judge_response(judge_response)

        except Exception as e:
            logger.warning("llm_judge_failed", error=str(e), task_id=task.id)
            # Judge failure = inconclusive, pass with warning
            return ValidationResult(
                passed=True,
                gate_name=self.name,
                score=0.5,
                reason=f"Judge call failed (inconclusive): {e}",
                details={"error": str(e)},
            )

        passed = score >= self._threshold

        return ValidationResult(
            passed=passed,
            gate_name=self.name,
            score=score,
            reason=reason,
            details={
                "threshold": self._threshold,
                "criteria": self._criteria,
                "model": self._model,
            },
        )

    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse the judge LLM response into (score, reason)."""
        import json

        # Try JSON parse
        try:
            # Handle markdown code blocks
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            score = float(data.get("score", 0.5))
            reason = str(data.get("reason", ""))
            return max(0.0, min(1.0, score)), reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract a number
        import re

        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            score = float(match.group(1))
            if score > 1.0:
                score = score / 10.0  # Handle 0-10 scale
            return max(0.0, min(1.0, score)), response[:200]

        return 0.5, f"Could not parse judge response: {response[:200]}"
