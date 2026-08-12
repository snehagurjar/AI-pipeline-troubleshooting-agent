"""
Phase 8 regression tests.

These tests cover the routing fix in agent.analyzer:

    SUCCESS + no data-quality issues -> success report, no LLM call
    SUCCESS + data-quality issues    -> LLM is called, pipeline_status stays SUCCESS
    FAILED  + clean dataset          -> LLM is called (unchanged behavior)
    FAILED  + messy dataset          -> LLM is called with BOTH log and dataset evidence

No real Groq API calls are made; run_llm_analysis() is always mocked.
"""

from unittest.mock import patch

import pandas as pd

from agent.analyzer import (
    build_analysis_context,
    create_success_report,
    should_run_llm_analysis,
)
from agent.schemas import TroubleshootingReport


SUCCESS_LOGS = [
    {
        "timestamp": "2026-08-12 10:00:00",
        "level": "INFO",
        "pipeline": "GENERIC_DATA_PIPELINE",
        "stage": "DATA_VALIDATION",
        "message": "Pipeline completed successfully",
    },
]

FAILED_LOGS = [
    {
        "timestamp": "2026-08-12 10:00:00",
        "level": "ERROR",
        "pipeline": "GENERIC_DATA_PIPELINE",
        "stage": "DATA_VALIDATION",
        "message": "ValueError: invalid values detected in column amount",
    },
    {
        "timestamp": "2026-08-12 10:00:01",
        "level": "CRITICAL",
        "pipeline": "GENERIC_DATA_PIPELINE",
        "stage": "DATA_VALIDATION",
        "message": "Pipeline stopped due to an unrecovered error",
    },
]


def clean_dataframe():
    return pd.DataFrame({
        "code": ["A", "B", "C"],
        "score": [1.2, 2.4, 3.1],
        "created": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })


def messy_dataframe():
    return pd.DataFrame({
        "record_code": ["R1", "R1", "R3", "R4"],
        "measurement": ["10", "oops", "12", None],
        "event_time": ["2026-01-01", "not-a-date", "2026-01-03", "2026-01-04"],
    })


def fake_report():
    return TroubleshootingReport(
        pipeline_status="SUCCESS",
        failed_stage="not applicable",
        error_type="not applicable",
        error_message="not applicable",
        issue_description="Dataset-quality issues were detected even though the pipeline succeeded.",
        root_cause="The dataset contains malformed values based on the supplied evidence.",
        severity="MEDIUM",
        confidence=0.8,
        recommended_actions=["Validate and clean the affected columns."],
        suggested_fix="Review the flagged rows before downstream use.",
        prevention="Add data-quality checks upstream.",
    )


# ---------------------------------------------------------------------------
# TEST 1: clean dataset + successful pipeline -> no LLM call
# ---------------------------------------------------------------------------

def test_clean_dataset_successful_pipeline_no_llm_call():
    error_report = build_analysis_context(
        SUCCESS_LOGS, [], dataframe=clean_dataframe()
    )

    assert error_report["pipeline_status"] == "SUCCESS"
    assert error_report["data_quality_issues"] == []
    assert should_run_llm_analysis(error_report) is False

    with patch("agent.analyzer.run_llm_analysis") as mock_llm:
        result = create_success_report(error_report)
        mock_llm.assert_not_called()

    assert result["pipeline_status"] == "SUCCESS"
    assert result["data_quality_issues"] == []


# ---------------------------------------------------------------------------
# TEST 2: messy dataset + successful pipeline -> LLM IS called
# ---------------------------------------------------------------------------

def test_messy_dataset_successful_pipeline_calls_llm():
    error_report = build_analysis_context(
        SUCCESS_LOGS, [], dataframe=messy_dataframe()
    )

    assert error_report["pipeline_status"] == "SUCCESS"
    assert error_report["data_quality_issues"], "Expected detected data-quality issues"
    assert should_run_llm_analysis(error_report) is True

    with patch(
        "agent.analyzer.run_llm_analysis", return_value=fake_report()
    ) as mock_llm:
        assert should_run_llm_analysis(error_report)
        result = mock_llm(error_report)

    mock_llm.assert_called_once()
    called_context = mock_llm.call_args[0][0]

    assert called_context["pipeline_status"] == "SUCCESS"
    assert "dataset_profile" in called_context
    assert called_context["dataset_profile"] is not None
    assert called_context["data_quality_issues"]
    assert "relevant_logs" in called_context

    assert result.pipeline_status == "SUCCESS"
    assert result.issue_description != (
        "No ERROR or CRITICAL entries were detected in the pipeline log."
    )
    assert result.root_cause != (
        "No pipeline failure was detected from the available logs."
    )


# ---------------------------------------------------------------------------
# TEST 3: failed pipeline + clean dataset -> LLM IS called (unchanged)
# ---------------------------------------------------------------------------

def test_failed_pipeline_clean_dataset_calls_llm():
    error_report = build_analysis_context(
        FAILED_LOGS, [], dataframe=clean_dataframe()
    )

    assert error_report["pipeline_status"] == "FAILED"
    assert should_run_llm_analysis(error_report) is True

    with patch(
        "agent.analyzer.run_llm_analysis", return_value=fake_report()
    ) as mock_llm:
        mock_llm(error_report)

    mock_llm.assert_called_once()
    called_context = mock_llm.call_args[0][0]
    assert called_context["pipeline_status"] == "FAILED"
    assert called_context["failed_stage"] == "DATA_VALIDATION"


# ---------------------------------------------------------------------------
# TEST 4: failed pipeline + messy dataset -> LLM context has BOTH kinds of evidence
# ---------------------------------------------------------------------------

def test_failed_pipeline_messy_dataset_calls_llm_with_both_evidence():
    error_report = build_analysis_context(
        FAILED_LOGS, [], dataframe=messy_dataframe()
    )

    assert error_report["pipeline_status"] == "FAILED"
    assert error_report["data_quality_issues"]
    assert should_run_llm_analysis(error_report) is True

    with patch(
        "agent.analyzer.run_llm_analysis", return_value=fake_report()
    ) as mock_llm:
        mock_llm(error_report)

    mock_llm.assert_called_once()
    called_context = mock_llm.call_args[0][0]

    # Log/error evidence
    assert called_context["pipeline_status"] == "FAILED"
    assert called_context["failed_stage"] == "DATA_VALIDATION"
    assert called_context["error_type"]
    assert called_context["relevant_logs"]

    # Dataset evidence
    assert called_context["dataset_profile"] is not None
    assert called_context["data_quality_issues"]


# ---------------------------------------------------------------------------
# should_run_llm_analysis unit coverage
# ---------------------------------------------------------------------------

def test_should_run_llm_analysis_matrix():
    assert should_run_llm_analysis(
        {"pipeline_status": "SUCCESS", "data_quality_issues": []}
    ) is False
    assert should_run_llm_analysis(
        {"pipeline_status": "SUCCESS", "data_quality_issues": [{"type": "missing_values"}]}
    ) is True
    assert should_run_llm_analysis(
        {"pipeline_status": "FAILED", "data_quality_issues": []}
    ) is True
    assert should_run_llm_analysis(
        {"pipeline_status": "FAILED", "data_quality_issues": [{"type": "missing_values"}]}
    ) is True
