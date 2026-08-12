import pandas as pd

from app.streamlit_app import build_upload_analysis, get_display_severity


def test_clean_dataset_has_no_quality_issues():
    df = pd.DataFrame({
        "code": ["A", "B", "C"],
        "amount": [10.0, 12.5, 8.0],
        "event_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    context = build_upload_analysis(df)
    assert context["data_quality_issues"] == []
    assert context["dataset_profile"]["dataset"]["rows"] == 3


def test_messy_dataset_uses_generic_backend():
    df = pd.DataFrame({
        "record_key": ["R1", "R1", "R3", "R4"],
        "measurement": ["10", "bad", None, "999999"],
        "event_date": ["2026-01-01", "not-a-date", "2026-01-03", "2026-01-04"],
    })
    context = build_upload_analysis(df)
    assert context["data_quality_issues"]
    assert context["category"] == "DATA"
    assert context["failed_stage"] is None


def test_severity_is_domain_independent():
    assert get_display_severity({"type": "missing_values"}) == "LOW"
    assert get_display_severity({"type": "duplicate_rows"}) == "MEDIUM"
    assert get_display_severity({"type": "empty_dataset"}) == "HIGH"


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame({"x": [1, None, 3], "y": ["a", "b", "c"]})
    before = df.copy(deep=True)
    build_upload_analysis(df)
    pd.testing.assert_frame_equal(df, before)
