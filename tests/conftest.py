"""Shared test fixtures for Agent Governor tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_governor.config import GovernorConfig
from agent_governor.governor import Governor
from agent_governor.observability.cost import CostTracker


@pytest.fixture
def config() -> GovernorConfig:
    """Basic test configuration."""
    return GovernorConfig(
        budget_usd=10.0,
        models=["gpt-4o", "qwen-plus"],
        fallback_model="qwen-plus",
        default_model="gpt-4o",
        max_concurrency=5,
        audit_enabled=False,
    )


@pytest.fixture
def unlimited_config() -> GovernorConfig:
    """Configuration with no budget limit."""
    return GovernorConfig(
        budget_usd=0.0,
        models=[],
        default_model="gpt-4o",
        audit_enabled=False,
    )


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Fresh cost tracker instance."""
    return CostTracker()


@pytest.fixture
def gov_audit_config() -> GovernorConfig:
    """Configuration with audit enabled for testing."""
    return GovernorConfig(
        budget_usd=10.0,
        models=["gpt-4o"],
        default_model="gpt-4o",
        audit_enabled=True,
    )


@pytest.fixture
def governor_audit(gov_audit_config: GovernorConfig) -> Governor:
    """Governor instance with audit enabled."""
    return Governor(config=gov_audit_config)


@pytest.fixture
def governor(config: GovernorConfig) -> Governor:
    """Governor instance with test config."""
    return Governor(config=config)


@pytest.fixture
def mock_litellm_response() -> MagicMock:
    """Mock litellm response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello! How can I help you?"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    return response


@pytest.fixture
def mock_acompletion(mock_litellm_response: MagicMock) -> AsyncMock:
    """Mock litellm.acompletion."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = mock_litellm_response
        yield mock
