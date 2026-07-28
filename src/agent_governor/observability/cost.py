"""Cost tracker - token accumulation, model pricing, and budget alerts."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# Default pricing table (USD per 1M tokens)
# Format: model_prefix -> (input_price, output_price)
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "qwen-plus": (0.80, 2.00),
    "qwen-turbo": (0.30, 0.60),
    "qwen-max": (2.00, 6.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-coder": (0.14, 0.28),
}


@dataclass
class ModelUsage:
    """Usage statistics for a single model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


@dataclass
class CostTracker:
    """Tracks token usage and costs across models.

    Thread-safe via lock. Provides budget checking and warning callbacks.
    """

    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    _usage_by_model: dict[str, ModelUsage] = field(default_factory=dict, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _total_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _warning_emitted: bool = field(default=False, init=False)

    @property
    def total_cost(self) -> float:
        """Total accumulated cost in USD."""
        with self._lock:
            return self._total_cost

    @property
    def total_calls(self) -> int:
        """Total number of recorded calls."""
        with self._lock:
            return self._total_calls

    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record token usage and return the cost for this call.

        Args:
            model: Model identifier.
            prompt_tokens: Number of prompt/input tokens.
            completion_tokens: Number of completion/output tokens.

        Returns:
            Cost in USD for this specific call.
        """
        input_price, output_price = self._get_pricing(model)

        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

        with self._lock:
            if model not in self._usage_by_model:
                self._usage_by_model[model] = ModelUsage()

            usage = self._usage_by_model[model]
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            usage.total_tokens += prompt_tokens + completion_tokens
            usage.cost_usd += cost
            usage.call_count += 1

            self._total_cost += cost
            self._total_calls += 1

        return cost

    def check_budget(self, budget_usd: float) -> bool:
        """Check if total cost is within budget.

        Args:
            budget_usd: Budget limit in USD.

        Returns:
            True if within budget, False if exceeded.
        """
        with self._lock:
            return self._total_cost < budget_usd

    def check_budget_warning(self, budget_usd: float, warning_ratio: float = 0.8) -> bool:
        """Check if cost has crossed the warning threshold.

        Args:
            budget_usd: Budget limit in USD.
            warning_ratio: Ratio at which to warn (default 80%).

        Returns:
            True if warning threshold crossed (and not previously emitted).
        """
        if budget_usd <= 0:
            return False

        with self._lock:
            threshold = budget_usd * warning_ratio
            if self._total_cost >= threshold and not self._warning_emitted:
                self._warning_emitted = True
                logger.warning(
                    "budget_warning",
                    current_cost=f"${self._total_cost:.4f}",
                    budget=f"${budget_usd:.4f}",
                    usage_percent=f"{(self._total_cost / budget_usd) * 100:.1f}%",
                )
                return True
            return False

    def get_model_usage(self, model: str) -> ModelUsage | None:
        """Get usage stats for a specific model."""
        with self._lock:
            return copy.deepcopy(self._usage_by_model.get(model))

    def get_summary(self) -> dict[str, object]:
        """Get a summary of all usage."""
        with self._lock:
            return {
                "total_cost_usd": self._total_cost,
                "total_calls": self._total_calls,
                "models": {
                    model: {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                        "cost_usd": u.cost_usd,
                        "call_count": u.call_count,
                    }
                    for model, u in self._usage_by_model.items()
                },
            }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._usage_by_model.clear()
            self._total_cost = 0.0
            self._total_calls = 0
            self._warning_emitted = False

    def _get_pricing(self, model: str) -> tuple[float, float]:
        """Get pricing for a model, using prefix matching."""
        # Exact match first
        if model in self.pricing:
            return self.pricing[model]

        # Prefix match - sort by key length descending so "gpt-4o-mini"
        # matches before "gpt-4o"
        for prefix, price in sorted(
            self.pricing.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if model.startswith(prefix):
                return price

        # Default fallback pricing
        logger.debug("unknown_model_pricing", model=model, using_default=True)
        return (1.0, 2.0)  # Conservative default
