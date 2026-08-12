from agent.error_detector import (
    classify_category,
    detect_error,
)


def make_log(
    level,
    message,
    stage,
):
    return {
        "level": level,
        "message": message,
        "stage": stage,
    }


def test_sales_error_is_data_without_sales_specific_rule():
    logs = [
        make_log(
            "ERROR",
            "ValueError: invalid values detected in column amount",
            "DATA_VALIDATION",
        )
    ]

    result = detect_error(logs)

    assert result["pipeline_status"] == "FAILED"
    assert result["category"] == "DATA"
    assert result["failed_stage"] == "DATA_VALIDATION"
    assert result["column"] == "amount"


def test_hr_error_is_data():
    logs = [
        make_log(
            "WARNING",
            "Missing values detected in column employee_code",
            "DATA_VALIDATION",
        ),
        make_log(
            "ERROR",
            "TypeError: invalid datatype detected in column tenure",
            "DATA_VALIDATION",
        ),
    ]

    result = detect_error(logs)

    assert result["category"] == "DATA"
    assert result["warning_count"] == 1


def test_finance_error_is_data():
    logs = [
        make_log(
            "ERROR",
            "ValueError: numeric conversion failed for column balance",
            "DATA_VALIDATION",
        )
    ]

    result = detect_error(logs)

    assert result["category"] == "DATA"


def test_generic_datatype_error():
    logs = [
        make_log(
            "ERROR",
            "TypeError: invalid datatype during type conversion",
            "DATA_VALIDATION",
        )
    ]

    result = detect_error(logs)

    assert result["category"] == "DATA"


def test_generic_missing_value_warning():
    logs = [
        make_log(
            "WARNING",
            "Missing values detected",
            "DATA_VALIDATION",
        )
    ]

    result = detect_error(logs)

    assert result["pipeline_status"] == "SUCCESS"
    assert result["category"] == "DATA"
    assert result["warning_issues"][0]["category"] == "DATA"


def test_database_error():
    logs = [
        make_log(
            "ERROR",
            "ConnectionError: database connection refused",
            "DATABASE_LOAD",
        ),
        make_log(
            "CRITICAL",
            "Pipeline stopped due to an unrecovered error",
            "DATABASE_LOAD",
        ),
    ]

    result = detect_error(logs)

    assert result["category"] == "DATABASE"
    assert result["failed_stage"] == "DATABASE_LOAD"


def test_model_error():
    logs = [
        make_log(
            "ERROR",
            "ValueError: model expects 10 features but input has 8 features",
            "MODEL_PREPARATION",
        )
    ]

    result = detect_error(logs)

    assert result["category"] == "MODEL"


def test_no_business_domain_is_invented():
    logs = [
        make_log(
            "ERROR",
            "Invalid values detected",
            "DATA_VALIDATION",
        )
    ]

    result = detect_error(logs)

    assert result["category"] == "DATA"
    assert "column" not in result


def test_unknown_error():
    logs = [
        make_log(
            "ERROR",
            "Something unexpected happened",
            "CUSTOM_STAGE",
        )
    ]

    result = detect_error(logs)

    assert result["category"] in {
        "UNKNOWN",
        "PIPELINE",
    }


def test_empty_logs():
    result = detect_error([])

    assert result["pipeline_status"] == "SUCCESS"
    assert result["category"] == "UNKNOWN"
    assert result["error_logs"] == []
