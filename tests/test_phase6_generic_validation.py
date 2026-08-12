
import pandas as pd
import pytest

from agent.analyzer import build_analysis_context
from agent.data_profiler import profile_dataframe
from agent.data_loader import load_data
from agent.troubleshooting_agent import TroubleshootingAgent
from pipeline.data_pipeline import validate_data


# ---------------------------------------------------------------------------
# Phase 6 dataset matrix
# ---------------------------------------------------------------------------
# The same generic code paths are used for every domain.  The schemas differ
# deliberately so that no business-specific column names are required by the
# implementation.
# ---------------------------------------------------------------------------

def clean_datasets():
    return {
        "sales": pd.DataFrame({
            "order_ref": ["S001", "S002", "S003", "S004", "S005"],
            "channel": ["web", "store", "web", "partner", "store"],
            "amount": [1200.0, 850.0, 430.0, 910.0, 670.0],
            "ordered_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "completed": [True, True, False, True, True],
        }),
        "hr": pd.DataFrame({
            "employee_ref": ["E001", "E002", "E003", "E004", "E005"],
            "department_code": ["ENG", "HR", "FIN", "OPS", "ENG"],
            "joining_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"],
            "compensation": [50000, 62000, 58000, 54000, 61000],
            "performance_score": [4.2, 3.8, 4.5, 3.9, 4.1],
        }),
        "finance": pd.DataFrame({
            "transaction_ref": ["T001", "T002", "T003", "T004", "T005"],
            "account_class": ["A", "B", "A", "C", "B"],
            "amount_value": [1200.50, 450.00, 800.75, 2300.00, 975.25],
            "transaction_date": ["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"],
            "state": ["posted", "posted", "pending", "posted", "posted"],
        }),
        "marketing": pd.DataFrame({
            "campaign_ref": ["M001", "M002", "M003", "M004", "M005"],
            "channel_type": ["search", "social", "email", "search", "social"],
            "impressions": [10000, 12000, 8000, 15000, 11000],
            "clicks": [400, 520, 300, 610, 450],
            "conversion_rate": [0.04, 0.043, 0.0375, 0.0407, 0.0409],
            "launched_on": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"],
        }),
        "inventory": pd.DataFrame({
            "item_ref": ["I001", "I002", "I003", "I004", "I005"],
            "warehouse_code": ["W1", "W1", "W2", "W2", "W3"],
            "stock_level": [120, 80, 45, 200, 75],
            "reorder_date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"],
            "supplier_code": ["SUP1", "SUP2", "SUP1", "SUP3", "SUP2"],
        }),
        "customer": pd.DataFrame({
            "profile_ref": ["C001", "C002", "C003", "C004", "C005"],
            "segment_code": ["A", "B", "A", "C", "B"],
            "engagement_score": [82.0, 74.0, 91.0, 65.0, 88.0],
            "registered_on": ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"],
            "subscribed": [True, False, True, True, False],
        }),
        "operations": pd.DataFrame({
            "operation_ref": ["O001", "O002", "O003", "O004", "O005"],
            "process_code": ["P1", "P2", "P1", "P3", "P2"],
            "duration_minutes": [35.0, 42.0, 28.0, 51.0, 39.0],
            "scheduled_for": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"],
            "completed": [True, True, True, False, True],
        }),
    }


def _make_messy(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Introduce generic quality problems without cleaning the source."""
    messy = df.copy(deep=True)

    # Missing value.
    messy.iloc[1, 1] = None

    # Duplicate full row.
    messy = pd.concat([messy, messy.iloc[[0]]], ignore_index=True)

    # Malformed value in a column that the generic profiler recognizes as
    # datetime-like.  No domain-specific column name is used here.
    date_candidates = profile_dataframe(df)["column_groups"]["datetime"]
    if date_candidates:
        date_column = date_candidates[0]
        messy[date_column] = messy[date_column].astype(object)
        messy.loc[2, date_column] = "not-a-date"

    # Mixed values in a numeric-looking column.
    numeric_columns = [
        column for column in messy.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]
    numeric_column = numeric_columns[0]
    # Explicitly use object dtype for the intentionally mixed test column.
    messy[numeric_column] = messy[numeric_column].astype(object)
    messy.loc[3, numeric_column] = "not-a-number"

    # Unusual numeric value in another numeric column where possible.
    if len(numeric_columns) > 1:
        other_numeric = numeric_columns[-1]
        if other_numeric != numeric_column:
            messy.loc[4, other_numeric] = 999999999

    return messy


@pytest.fixture(scope="module")
def dataset_matrix():
    clean = clean_datasets()
    messy = {name: _make_messy(df, name) for name, df in clean.items()}
    return clean, messy




def test_all_phase6_fixture_files_are_loadable():
    """Phase 5 loader must accept every Phase 6 CSV fixture."""
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "phase6_datasets"
    files = sorted(fixture_dir.glob("*.csv"))

    assert len(files) == 14

    for path in files:
        df = load_data(path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df.columns) >= 5

@pytest.mark.parametrize(
    "domain",
    ["sales", "hr", "finance", "marketing", "inventory", "customer", "operations"],
)
def test_same_profiler_handles_all_clean_domains(dataset_matrix, domain):
    clean, _ = dataset_matrix
    df = clean[domain]
    before = df.copy(deep=True)

    profile = profile_dataframe(df)

    assert profile["dataset"]["rows"] == len(df)
    assert profile["dataset"]["columns"] == len(df.columns)
    assert set(profile["columns"]) == {str(c) for c in df.columns}

    # Profiling is read-only.
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "domain",
    ["sales", "hr", "finance", "marketing", "inventory", "customer", "operations"],
)
def test_same_validator_detects_messy_domains(dataset_matrix, domain):
    _, messy = dataset_matrix
    df = messy[domain]
    before = df.copy(deep=True)

    issues = validate_data(df)

    issue_types = {issue["type"] for issue in issues}

    assert issues, f"No generic issues detected for {domain}"
    assert "missing_values" in issue_types
    assert "duplicate_rows" in issue_types
    assert (
        "mixed_datatype" in issue_types
        or "malformed_datetime_like_values" in issue_types
        or "numeric_outliers" in issue_types
    )

    # Validation is detection-only.
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "domain",
    ["sales", "hr", "finance", "marketing", "inventory", "customer", "operations"],
)
def test_same_workflow_builds_generic_evidence_for_messy_domains(
    dataset_matrix, domain
):
    _, messy = dataset_matrix
    df = messy[domain]

    # Generic failure log: no domain-specific business rule is used.
    logs = [
        {
            "timestamp": "2026-08-12 10:00:00",
            "level": "ERROR",
            "pipeline": "GENERIC_DATA_PIPELINE",
            "stage": "DATA_VALIDATION",
            "message": "ValueError: validation failed due to invalid data values",
        }
    ]

    context = build_analysis_context(logs, [], dataframe=df)

    assert context["pipeline_status"] == "FAILED"
    assert context["category"] == "DATA"
    assert context["failed_stage"] == "DATA_VALIDATION"
    assert context["dataset_profile"]["dataset"]["rows"] == len(df)
    assert context["dataset_profile"]["dataset"]["columns"] == len(df.columns)
    assert context["data_quality_issues"]

    # The detector must not invent a field when the log does not name one.
    assert "column" not in context


class _EvidenceBasedTestProvider:
    """
    Test double for the existing TroubleshootingAgent.

    This avoids a live Groq call while verifying that the same agent receives
    generic evidence and returns the required structured troubleshooting
    contract.
    """

    def __init__(self):
        self.received = []

    def analyze(self, system_prompt, input_data):
        from agent.schemas import TroubleshootingReport

        self.received.append(input_data)

        issues = input_data.get("data_quality_issues", [])
        first_issue = issues[0] if issues else {}

        return TroubleshootingReport(
            pipeline_status="FAILED",
            failed_stage=input_data.get("failed_stage") or "DATA_VALIDATION",
            error_type=input_data.get("error_type") or "ValueError",
            error_message=input_data.get("error_message") or "Generic validation failure",
            issue_description="Generic data-quality evidence was detected.",
            root_cause=(
                f"Observed data-quality evidence includes {first_issue.get('type', 'an unspecified issue')}."
            ),
            severity="HIGH",
            confidence=0.85,
            recommended_actions=[
                "Inspect the reported data-quality evidence.",
                "Validate the affected records before downstream processing.",
            ],
            suggested_fix="Correct the records or validation logic responsible for the observed issue after review.",
            prevention="Add automated schema and data-quality checks for future inputs.",
        )


@pytest.mark.parametrize(
    "domain",
    ["sales", "hr", "finance", "marketing", "inventory", "customer", "operations"],
)
def test_same_troubleshooting_agent_contract_for_all_messy_domains(
    dataset_matrix, domain
):
    _, messy = dataset_matrix
    df = messy[domain]

    logs = [
        {
            "timestamp": "2026-08-12 10:00:00",
            "level": "ERROR",
            "pipeline": "GENERIC_DATA_PIPELINE",
            "stage": "DATA_VALIDATION",
            "message": "ValueError: validation failed due to invalid data values",
        }
    ]

    context = build_analysis_context(logs, [], dataframe=df)

    provider = _EvidenceBasedTestProvider()
    agent = TroubleshootingAgent(provider=provider)
    report = agent.analyze(context)

    assert provider.received[0]["dataset_profile"] is not None
    assert provider.received[0]["data_quality_issues"]

    result = report.model_dump()
    required = {
        "issue_description",
        "root_cause",
        "severity",
        "recommended_actions",
        "suggested_fix",
        "prevention",
    }
    assert required.issubset(result)
    assert result["root_cause"]
    assert result["recommended_actions"]
    assert result["suggested_fix"]
    assert result["prevention"]


def test_clean_dataset_has_no_injected_quality_problems(dataset_matrix):
    clean, _ = dataset_matrix

    for domain, df in clean.items():
        issues = validate_data(df)

        # Clean fixtures are intentionally free of missing values,
        # duplicates, malformed dates, and extreme values.
        assert not any(
            issue["type"] in {
                "missing_values",
                "duplicate_rows",
                "mixed_datatype",
                "malformed_datetime_like_values",
            }
            for issue in issues
        ), f"Unexpected quality issue for clean {domain} dataset: {issues}"


def test_phase6_has_no_domain_specific_branching():
    """Guardrail: the production functions use no domain-specific branches."""
    import inspect

    production_sources = (
        inspect.getsource(profile_dataframe),
        inspect.getsource(validate_data),
        inspect.getsource(build_analysis_context),
    )

    combined = "\n".join(production_sources).lower()

    for marker in (
        "if sales:",
        "if hr:",
        "if finance:",
        "if marketing:",
        "if inventory:",
        "if customer:",
        "if operations:",
    ):
        assert marker not in combined
