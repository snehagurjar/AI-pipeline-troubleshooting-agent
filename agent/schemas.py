from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class TroubleshootingReport(BaseModel):
    """
    Structured output returned by the AI troubleshooting agent.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    pipeline_status: Literal["FAILED", "SUCCESS"]

    failed_stage: str

    error_type: str

    error_message: str

    issue_description: str

    root_cause: str

    severity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the root-cause analysis, "
            "between 0 and 1."
        )
    )

    recommended_actions: list[str]

    suggested_fix: str

    prevention: str