"""SchemaGate - validates task result content against a Pydantic model or JSON Schema."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from agent_governor.models import Task, TaskResult, ValidationResult


class SchemaGate:
    """Validates that task result content conforms to a Pydantic model.

    Attempts to parse the result content as JSON and validate it against
    the provided Pydantic model class.

    Example:
        class Report(BaseModel):
            title: str
            summary: str

        gate = SchemaGate(Report)
        result = await gate.check(task, task_result)
    """

    def __init__(self, schema: type[BaseModel], *, strict: bool = False) -> None:
        """Initialize SchemaGate.

        Args:
            schema: Pydantic model class to validate against.
            strict: If True, use strict mode validation.
        """
        self._schema = schema
        self._strict = strict

    @property
    def name(self) -> str:
        return f"schema_gate_{self._schema.__name__}"

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        """Validate result content against the schema.

        The content is expected to be valid JSON that can be parsed into
        the schema model.
        """
        # Try to parse content as JSON
        try:
            data: Any = json.loads(result.content)
        except (json.JSONDecodeError, TypeError) as e:
            return ValidationResult(
                passed=False,
                gate_name=self.name,
                score=0.0,
                reason=f"Content is not valid JSON: {e}",
                details={"content_preview": result.content[:200]},
            )

        # Validate against Pydantic model
        try:
            if self._strict:
                self._schema.model_validate(data, strict=True)
            else:
                self._schema.model_validate(data)
            return ValidationResult(
                passed=True,
                gate_name=self.name,
                score=1.0,
                reason="Schema validation passed",
            )
        except PydanticValidationError as e:
            errors = e.errors()
            return ValidationResult(
                passed=False,
                gate_name=self.name,
                score=0.0,
                reason=f"Schema validation failed: {len(errors)} error(s)",
                details={
                    "errors": [
                        {"field": ".".join(str(loc) for loc in err["loc"]), "msg": err["msg"]}
                        for err in errors[:5]  # Limit to first 5 errors
                    ]
                },
            )
