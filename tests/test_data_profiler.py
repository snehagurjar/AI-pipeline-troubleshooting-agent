import pandas as pd

from agent.data_profiler import profile_dataframe


def assert_common_profile_contract(profile):
    assert set(profile) >= {"dataset", "columns", "column_groups", "quality_summary"}
    assert set(profile["column_groups"]) >= {
        "numeric",
        "categorical",
        "datetime",
        "boolean",
        "possible_id",
        "high_cardinality",
    }


def test_sales_like_dataframe():
    df = pd.DataFrame(
        {
            "customer_ref": ["C001", "C002", "C003", "C004", "C005"],
            "amount": [100.0, 250.5, 90.0, 310.0, 125.0],
            "purchase_when": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            ),
            "region": ["North", "South", "North", "West", "South"],
            "is_returned": [False, False, True, False, False],
        }
    )

    original = df.copy(deep=True)
    result = profile_dataframe(df)

    assert_common_profile_contract(result)
    assert result["dataset"] == {"rows": 5, "columns": 5}
    assert result["columns"]["amount"]["inferred_type"] == "numeric"
    assert result["columns"]["purchase_when"]["inferred_type"] == "datetime"
    assert result["columns"]["is_returned"]["inferred_type"] == "boolean"
    assert result["columns"]["amount"]["statistics"]["median"] == 125.0
    pd.testing.assert_frame_equal(df, original)


def test_hr_like_dataframe():
    df = pd.DataFrame(
        {
            "employee_ref": [101, 102, 103, 104, 105, 106],
            "team": ["Engineering", "HR", "Finance", "Sales", "HR", "Engineering"],
            "tenure_years": [2, 4, 1, 6, 3, 5],
            "active": [True, True, False, True, True, False],
        }
    )

    result = profile_dataframe(df)

    assert_common_profile_contract(result)
    assert "employee_ref" in result["column_groups"]["possible_id"]
    assert "active" in result["column_groups"]["boolean"]
    assert "team" in result["column_groups"]["categorical"]
    assert result["quality_summary"]["duplicate_rows"] == 0


def test_finance_like_dataframe():
    df = pd.DataFrame(
        {
            "account_key": ["A-1", "A-2", "A-3", "A-4", "A-5"],
            "balance": [1000.25, -20.0, 450.0, 700.5, 125.75],
            "currency": ["INR", "USD", "INR", "EUR", "USD"],
            "statement_date": ["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"],
            "verified": [True, False, True, True, False],
        }
    )

    result = profile_dataframe(df)

    assert_common_profile_contract(result)
    assert result["columns"]["balance"]["statistics"]["min"] == -20.0
    assert result["columns"]["statement_date"]["inferred_type"] == "datetime"
    assert "verified" in result["column_groups"]["boolean"]
    assert "account_key" in result["column_groups"]["possible_id"]


def test_completely_different_schema():
    df = pd.DataFrame(
        {
            "device_code": ["D01", "D02", "D03", "D04", "D05", "D06"],
            "temperature": [21.5, 22.0, None, 24.5, 23.0, 25.0],
            "status_flag": ["ok", "ok", "offline", "ok", "warning", "ok"],
            "observed_at": [
                "2026-03-01 08:00",
                "2026-03-01 09:00",
                "not-a-date",
                "2026-03-01 11:00",
                "2026-03-01 12:00",
                "2026-03-01 13:00",
            ],
            "constant_value": [7, 7, 7, 7, 7, 7],
        }
    )

    result = profile_dataframe(df)

    assert_common_profile_contract(result)
    assert result["columns"]["temperature"]["missing_count"] == 1
    assert "constant_value" in result["quality_summary"]["constant_columns"]
    assert "observed_at" in result["quality_summary"]["possible_malformed_datetime_columns"]
    assert "device_code" in result["column_groups"]["possible_id"]


def test_edge_cases_do_not_fail():
    empty = pd.DataFrame()
    assert profile_dataframe(empty)["dataset"] == {"rows": 0, "columns": 0}

    all_null = pd.DataFrame({"nothing": [None, None, None]})
    result = profile_dataframe(all_null)
    assert "nothing" in result["quality_summary"]["empty_columns"]
    assert result["columns"]["nothing"]["missing_percentage"] == 100.0

    mixed = pd.DataFrame({"value": ["10", "20", "bad", "40", "50"]})
    result = profile_dataframe(mixed)
    assert "value" in result["quality_summary"]["suspicious_mixed_type_columns"]


def test_profile_does_not_modify_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    before = df.copy(deep=True)

    profile_dataframe(df)

    pd.testing.assert_frame_equal(df, before)
