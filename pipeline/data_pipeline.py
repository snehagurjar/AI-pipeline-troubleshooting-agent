import io
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PIPELINE_NAME = "GENERIC_DATA_PIPELINE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "sales.csv"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline.log"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        f"{PIPELINE_NAME} | %(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="a",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE STAGES
# ============================================================

DATA_LOADING = "DATA_LOADING"
DATA_PROFILING = "DATA_PROFILING"
DATA_VALIDATION = "DATA_VALIDATION"
DATA_CLEANING = "DATA_CLEANING"
FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
MODEL_PREPARATION = "MODEL_PREPARATION"


# ============================================================
# GENERIC DATA LOADING
# ============================================================

def load_data(path: Optional[Path] = None):
    """
    Load a CSV dataset into a Pandas DataFrame.

    The default path keeps the existing Sales pipeline runnable,
    while the function itself accepts any CSV path.
    """
    data_path = Path(path) if path is not None else DATA_PATH

    logger.info(
        "[%s] Data loading started | path=%s",
        DATA_LOADING,
        data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    logger.info(
        "[%s] Data loaded successfully | rows=%d | columns=%d",
        DATA_LOADING,
        df.shape[0],
        df.shape[1]
    )

    return df


# ============================================================
# GENERIC DATA PROFILING
# ============================================================

def profile_data(df):
    """
    Generate a generic, read-only profile of the DataFrame.

    Uses the Phase 1 profiler when available. A local fallback is
    retained so this pipeline remains importable if the profiler
    module is temporarily unavailable.
    """
    logger.info(
        "[%s] Data profiling started",
        DATA_PROFILING
    )

    try:
        from agent.data_profiler import profile_dataframe
        profile = profile_dataframe(df)
    except ImportError:
        buffer = io.StringIO()
        df.info(buf=buffer)

        profile = {
            "dataset": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "columns": {
                str(column): {
                    "dtype": str(df[column].dtype),
                    "missing_count": int(df[column].isna().sum()),
                    "unique_count": int(df[column].nunique(dropna=True))
                }
                for column in df.columns
            },
            "column_groups": {},
            "quality_summary": {}
        }

    logger.info(
        "[%s] Profile generated | rows=%d | columns=%d",
        DATA_PROFILING,
        df.shape[0],
        df.shape[1]
    )

    return profile


# ============================================================
# GENERIC DATA VALIDATION
# ============================================================

def validate_data(df, expected_columns=None):
    """
    Detect and log domain-independent data-quality issues.

    IMPORTANT:
    This function does not clean or mutate the DataFrame.
    It returns a list of detected issues so downstream components
    can use the evidence for troubleshooting.
    """
    logger.info(
        "[%s] Data validation started",
        DATA_VALIDATION
    )

    issues = []

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:
        message = "Dataset is empty"
        logger.warning("[%s] %s", DATA_VALIDATION, message)
        issues.append({
            "type": "empty_dataset",
            "message": message
        })
        return issues

    if len(df.columns) == 0:
        message = "Dataset contains no columns"
        logger.warning("[%s] %s", DATA_VALIDATION, message)
        issues.append({
            "type": "empty_schema",
            "message": message
        })
        return issues

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    buffer = io.StringIO()
    df.info(buf=buffer)

    logger.info(
        "[%s] Dataset information:\n%s",
        DATA_VALIDATION,
        buffer.getvalue()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    for column in df.columns:
        count = int(missing_counts[column])
        percentage = float(missing_percentages[column])

        if count > 0:
            issue = {
                "type": "missing_values",
                "column": str(column),
                "count": count,
                "percentage": round(percentage, 2)
            }
            issues.append(issue)

            logger.warning(
                "[%s] Missing values detected | column=%s | count=%d | percentage=%.2f%%",
                DATA_VALIDATION,
                column,
                count,
                percentage
            )

        if percentage >= 50:
            issue = {
                "type": "high_missing_percentage",
                "column": str(column),
                "percentage": round(percentage, 2)
            }
            issues.append(issue)

            logger.warning(
                "[%s] High missing percentage | column=%s | percentage=%.2f%%",
                DATA_VALIDATION,
                column,
                percentage
            )

    # --------------------------------------------------------
    # Completely empty columns
    # --------------------------------------------------------

    empty_columns = [
        str(column)
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:
        issues.append({
            "type": "empty_columns",
            "columns": empty_columns
        })
        logger.warning(
            "[%s] Completely empty columns detected: %s",
            DATA_VALIDATION,
            empty_columns
        )

    # --------------------------------------------------------
    # Constant columns
    # --------------------------------------------------------

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        issues.append({
            "type": "constant_columns",
            "columns": constant_columns
        })
        logger.warning(
            "[%s] Constant columns detected: %s",
            DATA_VALIDATION,
            constant_columns
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        issues.append({
            "type": "duplicate_rows",
            "count": duplicate_rows
        })
        logger.warning(
            "[%s] Duplicate rows detected | count=%d",
            DATA_VALIDATION,
            duplicate_rows
        )

    # --------------------------------------------------------
    # Possible identifier duplication
    # --------------------------------------------------------

    try:
        from agent.data_profiler import profile_dataframe
        profile = profile_dataframe(df)
        possible_ids = profile.get("column_groups", {}).get(
            "possible_id", []
        )
    except Exception:
        possible_ids = []

    for column in possible_ids:
        duplicate_count = int(
            df[column].duplicated(keep=False).sum()
        )

        if duplicate_count > 0:
            issues.append({
                "type": "duplicate_possible_identifier",
                "column": str(column),
                "duplicate_value_rows": duplicate_count
            })
            logger.warning(
                "[%s] Duplicate values in possible identifier | column=%s | rows=%d",
                DATA_VALIDATION,
                column,
                duplicate_count
            )

    # --------------------------------------------------------
    # Suspicious numeric values using generic IQR rule
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            issue = {
                "type": "numeric_outliers",
                "column": str(column),
                "count": outlier_count,
                "lower_bound": float(lower),
                "upper_bound": float(upper)
            }
            issues.append(issue)

            logger.warning(
                "[%s] Suspicious numeric values detected | column=%s | outlier_count=%d | bounds=(%s, %s)",
                DATA_VALIDATION,
                column,
                outlier_count,
                lower,
                upper
            )

    # --------------------------------------------------------
    # Mixed / suspicious object columns
    # --------------------------------------------------------

    for column in df.columns:
        series = df[column].dropna()

        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue

        type_count = series.map(type).nunique()

        if type_count > 1:
            issue = {
                "type": "mixed_datatype",
                "column": str(column),
                "distinct_python_types": int(type_count)
            }
            issues.append(issue)

            logger.warning(
                "[%s] Mixed datatype values detected | column=%s | distinct_types=%d",
                DATA_VALIDATION,
                column,
                type_count
            )

    # --------------------------------------------------------
    # Malformed datetime-like values
    # --------------------------------------------------------

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            continue

        if (
            pd.api.types.is_datetime64_any_dtype(df[column])
            or pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            sample = series.head(1000)

            try:
                parsed = pd.to_datetime(
                    sample,
                    errors="coerce",
                    format="mixed"
                )
                parse_ratio = float(parsed.notna().mean())
            except (TypeError, ValueError):
                parse_ratio = 0.0

            # Only classify a column as datetime-like when a
            # meaningful portion of its values parse as dates.
            if 0.5 <= parse_ratio < 1.0:
                issue = {
                    "type": "malformed_datetime_like_values",
                    "column": str(column),
                    "parse_ratio": round(parse_ratio, 3),
                    # Precomputed so downstream consumers (including the
                    # LLM) don't have to derive it from parse_ratio
                    # themselves, which is an easy place to make an
                    # arithmetic mistake.
                    "unparseable_percentage": round(
                        (1.0 - parse_ratio) * 100, 2
                    )
                }
                issues.append(issue)

                logger.warning(
                    "[%s] Possible malformed datetime-like values | column=%s | parse_ratio=%.3f",
                    DATA_VALIDATION,
                    column,
                    parse_ratio
                )

    # --------------------------------------------------------
    # Schema consistency
    # --------------------------------------------------------

    if expected_columns is not None:
        expected = {str(column) for column in expected_columns}
        actual = {str(column) for column in df.columns}

        missing_schema = sorted(expected - actual)
        unexpected_schema = sorted(actual - expected)

        if missing_schema:
            issue = {
                "type": "schema_missing_columns",
                "columns": missing_schema
            }
            issues.append(issue)

            logger.warning(
                "[%s] Expected columns missing: %s",
                DATA_VALIDATION,
                missing_schema
            )

        if unexpected_schema:
            issue = {
                "type": "schema_unexpected_columns",
                "columns": unexpected_schema
            }
            issues.append(issue)

            logger.warning(
                "[%s] Unexpected columns detected: %s",
                DATA_VALIDATION,
                unexpected_schema
            )

    logger.info(
        "[%s] Data validation completed | issues=%d",
        DATA_VALIDATION,
        len(issues)
    )

    return issues


# ============================================================
# GENERIC DATA-QUALITY LOGGING
# ============================================================

def log_data_quality_issues(issues):
    """Log a compact summary of detected quality issues."""
    if not issues:
        logger.info(
            "[%s] No generic data-quality issues detected",
            DATA_VALIDATION
        )
        return

    logger.info(
        "[%s] Data-quality evidence summary | issue_count=%d",
        DATA_VALIDATION,
        len(issues)
    )

    for issue in issues:
        logger.info(
            "[%s] ISSUE | %s",
            DATA_VALIDATION,
            issue
        )


# ============================================================
# STAGE 3: DATA CLEANING - COMPATIBILITY ONLY
# ============================================================

def clean_data(df):
    """
    Compatibility wrapper.

    No cleaning is performed. The original DataFrame is returned
    unchanged so the troubleshooting system can inspect the actual
    problematic data.
    """
    logger.info(
        "[%s] Detection-only stage started | no data mutation performed",
        DATA_CLEANING
    )

    logger.info(
        "[%s] Detection-only stage completed | no imputation, row drops, "
        "deduplication, normalization, or value replacement performed",
        DATA_CLEANING
    )

    return df


# ============================================================
# STAGE 4: FEATURE ENGINEERING - GENERIC COMPATIBILITY
# ============================================================

def feature_engineering(df):
    """
    Compatibility wrapper.

    No domain-specific feature engineering is performed.
    """
    logger.info(
        "[%s] Generic feature engineering stage started",
        FEATURE_ENGINEERING
    )

    logger.info(
        "[%s] No domain-specific feature engineering configured; "
        "DataFrame retained unchanged",
        FEATURE_ENGINEERING
    )

    return df


# ============================================================
# STAGE 5: MODEL PREPARATION - GENERIC COMPATIBILITY
# ============================================================

def prepare_model_data(df):
    """
    Generic model-preparation compatibility stage.

    No target column or fixed feature schema is assumed.
    Returns the unchanged DataFrame and None target.
    """
    logger.info(
        "[%s] Generic model preparation started",
        MODEL_PREPARATION
    )

    logger.info(
        "[%s] No domain-specific target/feature schema configured | "
        "dataset retained as-is | shape=%s",
        MODEL_PREPARATION,
        df.shape
    )

    return df, None


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(data_path: Optional[Path] = None):
    current_stage = "PIPELINE_START"

    logger.info("=" * 80)
    logger.info("Pipeline started")
    logger.info("=" * 80)

    try:
        # ----------------------------------------------------
        # DATA LOADING
        # ----------------------------------------------------

        current_stage = DATA_LOADING
        df = load_data(data_path)

        # ----------------------------------------------------
        # DATA PROFILING
        # ----------------------------------------------------

        current_stage = DATA_PROFILING
        profile = profile_data(df)

        # ----------------------------------------------------
        # DATA VALIDATION
        # ----------------------------------------------------

        current_stage = DATA_VALIDATION
        issues = validate_data(df)
        log_data_quality_issues(issues)

        # ----------------------------------------------------
        # DETECTION-ONLY COMPATIBILITY STAGE
        # ----------------------------------------------------

        current_stage = DATA_CLEANING
        df = clean_data(df)

        # ----------------------------------------------------
        # GENERIC FEATURE ENGINEERING COMPATIBILITY STAGE
        # ----------------------------------------------------

        current_stage = FEATURE_ENGINEERING
        df = feature_engineering(df)

        # ----------------------------------------------------
        # GENERIC MODEL PREPARATION
        # ----------------------------------------------------

        current_stage = MODEL_PREPARATION
        model_data, target = prepare_model_data(df)

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        logger.info("=" * 80)
        logger.info(
            "Pipeline completed successfully | "
            "rows=%d | columns=%d | detected_issues=%d",
            model_data.shape[0],
            model_data.shape[1],
            len(issues)
        )
        logger.info("=" * 80)

        return {
            "data": model_data,
            "profile": profile,
            "issues": issues,
            "target": target
        }

    except Exception as error:
        # ----------------------------------------------------
        # ERROR INFORMATION
        # ----------------------------------------------------

        error_type = type(error).__name__
        error_message = str(error)

        logger.error(
            "[%s] Pipeline failed | "
            "ExceptionType=%s | "
            "Error=%s",
            current_stage,
            error_type,
            error_message,
            exc_info=True
        )

        logger.critical(
            "[%s] Pipeline stopped due to an unrecovered error",
            current_stage
        )

        # Do not hide the original error
        raise


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Accept an optional CLI argument:
    #   python -m pipeline.data_pipeline data/inventory_messy.csv
    # Falls back to the default sales.csv only if no path is given.
    cli_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_pipeline(cli_path)