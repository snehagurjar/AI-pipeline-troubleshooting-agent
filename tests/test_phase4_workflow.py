import pandas as pd

from agent.analyzer import build_analysis_context


def logs(message="ValueError: invalid values detected in column amount"):
    return [
        {
            "timestamp": "2026-08-12 10:00:00",
            "level": "ERROR",
            "pipeline": "GENERIC_DATA_PIPELINE",
            "stage": "DATA_VALIDATION",
            "message": message,
        },
        {
            "timestamp": "2026-08-12 10:00:01",
            "level": "CRITICAL",
            "pipeline": "GENERIC_DATA_PIPELINE",
            "stage": "DATA_VALIDATION",
            "message": "Pipeline stopped due to an unrecovered error",
        },
    ]


def assert_dataset_context(df):
    result = build_analysis_context(logs(), [], dataframe=df)
    assert result["dataset_profile"]["dataset"]["rows"] == len(df)
    assert result["dataset_profile"]["dataset"]["columns"] == len(df.columns)
    assert isinstance(result["data_quality_issues"], list)
    assert result["pipeline_status"] == "FAILED"
    assert result["failed_stage"] == "DATA_VALIDATION"
    return result


def test_sales_dataset_context():
    df = pd.DataFrame({
        "customer_ref": ["C1", "C2", "C3"],
        "amount": [100, 250, 180],
        "ordered_at": ["2026-01-01", "2026-01-02", "bad-date"],
    })
    result = assert_dataset_context(df)
    assert "amount" in result["dataset_profile"]["columns"]


def test_hr_dataset_context():
    df = pd.DataFrame({
        "worker_key": ["W1", "W2", "W3"],
        "tenure": [2, None, 7],
        "team": ["A", "B", "A"],
    })
    result = assert_dataset_context(df)
    assert any(i["type"] == "missing_values" for i in result["data_quality_issues"])


def test_finance_dataset_context():
    df = pd.DataFrame({
        "account_key": ["A1", "A2", "A3"],
        "balance": [1000.0, 1200.5, 999.0],
        "as_of": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    result = assert_dataset_context(df)
    assert result["dataset_profile"]["columns"]["balance"]["inferred_type"] == "numeric"


def test_messy_arbitrary_dataset_context():
    df = pd.DataFrame({
        "record_code": ["R1", "R1", "R3", "R4"],
        "measurement": ["10", "oops", "12", None],
        "event_time": ["2026-01-01", "not-a-date", "2026-01-03", "2026-01-04"],
        "flag": [True, False, True, False],
    })
    result = assert_dataset_context(df)
    assert result["data_quality_issues"]


def test_clean_arbitrary_dataset_context():
    df = pd.DataFrame({
        "code": ["A", "B", "C"],
        "score": [1.2, 2.4, 3.1],
        "created": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    result = assert_dataset_context(df)
    assert result["dataset_profile"] is not None
    # Clean data should not be forced to have a failure-related quality issue.
    assert isinstance(result["data_quality_issues"], list)
