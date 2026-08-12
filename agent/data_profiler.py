"""Generic, read-only profiling for arbitrary tabular datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


# Heuristics are intentionally schema-independent.
_HIGH_CARDINALITY_RATIO = 0.90
_HIGH_MISSING_RATIO = 0.50
_ID_UNIQUENESS_RATIO = 0.95
_ID_MIN_NON_NULL_VALUES = 5
_INFERENCE_SAMPLE_SIZE = 1000


def _percentage(value: int | float, total: int) -> float:
    if total == 0:
        return 0.0
    return round((float(value) / total) * 100.0, 2)


def _safe_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars to ordinary Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def _sample_non_null(series: pd.Series, limit: int = _INFERENCE_SAMPLE_SIZE) -> pd.Series:
    """Take a bounded sample for expensive type inference operations."""
    non_null = series.dropna()
    if len(non_null) <= limit:
        return non_null
    return non_null.iloc[:limit]


def _infer_type(series: pd.Series) -> tuple[str, dict[str, Any]]:
    """Infer a general semantic type without changing the source series."""
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean", {"numeric_parse_ratio": None, "datetime_parse_ratio": None}

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime", {"numeric_parse_ratio": None, "datetime_parse_ratio": 100.0}

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric", {"numeric_parse_ratio": 100.0, "datetime_parse_ratio": None}

    sample = _sample_non_null(series)
    if sample.empty:
        return "unknown", {"numeric_parse_ratio": 0.0, "datetime_parse_ratio": 0.0}

    # Work only on a bounded sample so large object columns remain safe.
    numeric_ratio = float(pd.to_numeric(sample, errors="coerce").notna().mean())

    try:
        datetime_ratio = float(pd.to_datetime(sample, errors="coerce").notna().mean())
    except (TypeError, ValueError):
        datetime_ratio = 0.0

    if datetime_ratio >= 0.95 and numeric_ratio < 0.95:
        inferred = "datetime"
    elif numeric_ratio >= 0.95:
        inferred = "numeric"
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        inferred = "categorical"
    else:
        inferred = "other"

    return inferred, {
        "numeric_parse_ratio": round(numeric_ratio * 100.0, 2),
        "datetime_parse_ratio": round(datetime_ratio * 100.0, 2),
    }


def _is_possible_id(
    series: pd.Series,
    inferred_type: str,
    unique_count: int,
    unique_percentage: float,
    missing_percentage: float,
    rows: int,
) -> bool:
    """Identify likely identifiers using distribution characteristics only."""
    non_null_count = int(series.notna().sum())

    if rows < _ID_MIN_NON_NULL_VALUES or non_null_count < _ID_MIN_NON_NULL_VALUES:
        return False
    if inferred_type == "datetime":
        return False
    if missing_percentage > 5.0:
        return False
    if unique_count < _ID_MIN_NON_NULL_VALUES or unique_percentage < (_ID_UNIQUENESS_RATIO * 100):
        return False

    # A near-unique column is a stronger ID signal when it is text-like,
    # integer-like, or monotonically increasing. Continuous measurements
    # are intentionally not treated as IDs merely because they are unique.
    if inferred_type == "categorical":
        return True

    if pd.api.types.is_integer_dtype(series.dtype):
        return True

    if pd.api.types.is_numeric_dtype(series.dtype):
        non_null = series.dropna()
        return bool(non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing)

    return False


def _is_suspicious_mixed_type(series: pd.Series, inferred_type: str, parse_info: dict[str, Any]) -> bool:
    """Flag object/string columns whose values show inconsistent type signals."""
    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    numeric_ratio = parse_info.get("numeric_parse_ratio") or 0.0
    datetime_ratio = parse_info.get("datetime_parse_ratio") or 0.0

    # Some values parse as numeric while a meaningful portion does not.
    if 5.0 <= numeric_ratio < 95.0:
        return True

    # Some values parse as dates while a meaningful portion does not.
    if 5.0 <= datetime_ratio < 95.0:
        return True

    # Object columns containing several actual Python value types are also
    # suspicious, except for the common string/None representation already
    # removed by dropna().
    type_count = non_null.map(type).nunique()
    return bool(type_count > 1 and inferred_type == "categorical")


def _has_malformed_datetime_values(series: pd.Series, inferred_type: str, parse_info: dict[str, Any]) -> bool:
    """Detect columns that look date-like but contain unparseable values."""
    if inferred_type == "datetime":
        return False

    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return False

    ratio = parse_info.get("datetime_parse_ratio") or 0.0
    return 5.0 <= ratio < 95.0


def _numeric_statistics(series: pd.Series) -> dict[str, Any] | None:
    """Return numeric descriptive statistics, or None for non-numeric data."""
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        return None

    non_null = series.dropna()
    if non_null.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "quartiles": {"q1": None, "q2": None, "q3": None},
        }

    try:
        quartiles = non_null.quantile([0.25, 0.50, 0.75])
        return {
            "min": _safe_scalar(non_null.min()),
            "max": _safe_scalar(non_null.max()),
            "mean": _safe_scalar(non_null.mean()),
            "median": _safe_scalar(non_null.median()),
            "std": _safe_scalar(non_null.std()),
            "quartiles": {
                "q1": _safe_scalar(quartiles.loc[0.25]),
                "q2": _safe_scalar(quartiles.loc[0.50]),
                "q3": _safe_scalar(quartiles.loc[0.75]),
            },
        }
    except (TypeError, ValueError):
        return None


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Profile any pandas DataFrame without modifying it.

    The function is intentionally schema-agnostic. It does not clean,
    transform, drop, rename, or otherwise mutate ``df``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataframe expects a pandas DataFrame")

    rows, columns = df.shape

    result: dict[str, Any] = {
        "dataset": {
            "rows": int(rows),
            "columns": int(columns),
        },
        "columns": {},
        "column_groups": {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
            "possible_id": [],
            "possible_index": [],
            "high_cardinality": [],
        },
        "quality_summary": {
            "duplicate_rows": int(df.duplicated().sum()) if rows else 0,
            "duplicate_row_percentage": _percentage(int(df.duplicated().sum()), rows),
            "empty_columns": [],
            "constant_columns": [],
            "high_missing_columns": [],
            "suspicious_mixed_type_columns": [],
            "possible_malformed_datetime_columns": [],
        },
    }

    if columns == 0:
        return result

    for column in df.columns:
        series = df[column]
        name = str(column)

        missing_count = int(series.isna().sum())
        non_null_count = rows - missing_count
        unique_count = int(series.nunique(dropna=True))
        unique_percentage = _percentage(unique_count, non_null_count)
        missing_percentage = _percentage(missing_count, rows)

        inferred_type, parse_info = _infer_type(series)
        stats = _numeric_statistics(series)

        is_empty = non_null_count == 0
        is_constant = non_null_count > 0 and unique_count <= 1
        is_high_cardinality = (
            non_null_count > 0
            and unique_percentage >= (_HIGH_CARDINALITY_RATIO * 100)
            and unique_count > 20
        )
        is_possible_id = _is_possible_id(
            series,
            inferred_type,
            unique_count,
            unique_percentage,
            missing_percentage,
            rows,
        )
        is_mixed = _is_suspicious_mixed_type(series, inferred_type, parse_info)
        malformed_datetime = _has_malformed_datetime_values(
            series, inferred_type, parse_info
        )

        result["columns"][name] = {
            "dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "high_cardinality": is_high_cardinality,
            "possible_id": is_possible_id,
            "possible_index": is_possible_id and bool(series.index.equals(pd.RangeIndex(rows))),
            "suspicious_mixed_type": is_mixed,
            "possible_malformed_datetime": malformed_datetime,
            "statistics": stats,
        }

        if inferred_type in result["column_groups"]:
            result["column_groups"][inferred_type].append(name)

        if is_possible_id:
            result["column_groups"]["possible_id"].append(name)

        if is_possible_id and pd.api.types.is_numeric_dtype(series.dtype):
            non_null = series.dropna()
            if not non_null.empty and (
                non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing
            ):
                result["column_groups"]["possible_index"].append(name)

        if is_high_cardinality:
            result["column_groups"]["high_cardinality"].append(name)

        if is_empty:
            result["quality_summary"]["empty_columns"].append(name)
        if is_constant:
            result["quality_summary"]["constant_columns"].append(name)
        if missing_percentage >= (_HIGH_MISSING_RATIO * 100):
            result["quality_summary"]["high_missing_columns"].append(name)
        if is_mixed:
            result["quality_summary"]["suspicious_mixed_type_columns"].append(name)
        if malformed_datetime:
            result["quality_summary"]["possible_malformed_datetime_columns"].append(name)

    return result


# Convenient aliases for callers that prefer a generic "profile" name.
profile_data = profile_dataframe
profile = profile_dataframe
