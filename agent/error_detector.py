import re
from typing import Any


# ============================================================
# ERROR TYPES
# ============================================================

COMMON_ERROR_TYPES = [
    "ValueError",
    "TypeError",
    "KeyError",
    "NameError",
    "FileNotFoundError",
    "PermissionError",
    "ImportError",
    "ModuleNotFoundError",
    "MemoryError",
    "ConnectionError",
    "TimeoutError",
    "RuntimeError",
]


# ============================================================
# GENERIC ISSUE CATEGORIES
# ============================================================

ISSUE_CATEGORIES = {
    "DATA",
    "CODE",
    "MODEL",
    "DATABASE",
    "INFRASTRUCTURE",
    "PIPELINE",
    "CONFIGURATION",
    "UNKNOWN",
}


# ============================================================
# GENERIC KEYWORD GROUPS
# ============================================================

DATA_KEYWORDS = [
    "missing value",
    "missing values",
    "null value",
    "null values",
    "invalid value",
    "invalid values",
    "duplicate record",
    "duplicate records",
    "duplicate row",
    "duplicate rows",
    "duplicate identifier",
    "invalid datatype",
    "invalid data type",
    "datatype conversion",
    "data type conversion",
    "type conversion",
    "numeric conversion",
    "could not convert",
    "cannot convert",
    "conversion failed",
    "malformed date",
    "malformed dates",
    "invalid date",
    "invalid dates",
    "date parsing",
    "date parse",
    "parsing error",
    "parse error",
    "schema mismatch",
    "schema inconsistency",
    "unexpected column",
    "missing column",
    "missing columns",
    "unexpected category",
    "unexpected categories",
    "validation failure",
    "validation failed",
    "constraint violation",
    "data quality",
    "data validation",
    "mixed datatype",
    "mixed data type",
    "non-numeric",
]

DATABASE_KEYWORDS = [
    "database",
    "db connection",
    "database connection",
    "sql",
    "postgres",
    "postgresql",
    "mysql",
    "sqlite",
    "query failed",
    "query error",
    "connection refused",
    "connection reset",
    "database unavailable",
    "transaction failed",
]

MODEL_KEYWORDS = [
    "model",
    "model input",
    "model output",
    "prediction",
    "predict",
    "inference",
    "feature mismatch",
    "features mismatch",
    "feature dimension",
    "input shape",
    "shape mismatch",
    "expected features",
    "model expects",
    "training failed",
    "training error",
    "inference failed",
]

INFRASTRUCTURE_KEYWORDS = [
    "out of memory",
    "memory exhausted",
    "memory exhaustion",
    "memoryerror",
    "disk space",
    "no space left",
    "server unavailable",
    "host unavailable",
    "network unavailable",
    "network error",
    "connection timed out",
    "timed out",
    "timeout",
    "worker crashed",
    "process killed",
    "resource exhausted",
    "container crashed",
    "service unavailable",
]

CONFIGURATION_KEYWORDS = [
    "configuration",
    "config",
    "environment variable",
    "env variable",
    "invalid setting",
    "invalid configuration",
    "missing configuration",
    "configuration error",
    "config error",
    "invalid parameter",
    "missing parameter",
]

PIPELINE_KEYWORDS = [
    "pipeline failed",
    "pipeline stopped",
    "pipeline stage",
    "stage failed",
    "stage error",
    "workflow failed",
    "workflow error",
    "job failed",
    "task failed",
    "orchestration",
]

CODE_KEYWORDS = [
    "syntaxerror",
    "attributeerror",
    "nameerror",
    "typeerror",
    "importerror",
    "modulenotfounderror",
    "runtimeerror",
    "undefined variable",
    "undefined function",
    "attribute not found",
    "method not found",
    "function failed",
]

# Strong exception-to-category hints. These are not used alone;
# message/stage/context evidence is considered first.
EXCEPTION_HINTS = {
    "MemoryError": "INFRASTRUCTURE",
    "FileNotFoundError": "PIPELINE",
    "PermissionError": "CONFIGURATION",
    "ModuleNotFoundError": "CODE",
    "ImportError": "CODE",
    "NameError": "CODE",
    "AttributeError": "CODE",
    "SyntaxError": "CODE",
    "TypeError": "CODE",
    "ConnectionError": "INFRASTRUCTURE",
    "TimeoutError": "INFRASTRUCTURE",
    "RuntimeError": "PIPELINE",
}


def _normalise_text(*parts: str | None) -> str:
    """Combine evidence into normalized text for classification."""
    return " ".join(
        str(part)
        for part in parts
        if part
    ).lower()


def _keyword_score(text: str, keywords: list[str]) -> int:
    """Return the number of generic evidence patterns found."""
    return sum(1 for keyword in keywords if keyword in text)


def _extract_column_reference(message: str) -> str | None:
    """
    Extract an explicitly mentioned column/field reference when the
    log provides one. Never invents a field name.
    """
    if not message:
        return None

    patterns = [
        r"\bcolumn\s+['\"]?([^,'\";\]\)]+)",
        r"\bfield\s+['\"]?([^,'\";\]\)]+)",
        r"\bcolumn=['\"]([^'\"]+)",
        r"\bcolumn:\s*['\"]?([^,'\";\]\)]+)",
        r"\bfield:\s*['\"]?([^,'\";\]\)]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value

    return None


# ============================================================
# ISSUE CATEGORY
# ============================================================

def classify_category(
    error_type: str | None,
    message: str,
    stage: str | None,
    context: list[dict[str, Any]] | None = None,
) -> str:
    """
    Classify an issue using evidence from the exception, message,
    failed stage, and surrounding logs.

    The classifier is domain-independent. It never maps a specific
    business field such as Salary, Sales, Department, etc. to a
    category unless that exact term happens to occur in the log.
    """

    context_text = ""
    if context:
        context_text = " ".join(
            str(log.get("message", ""))
            for log in context
        )

    text = _normalise_text(
        error_type,
        message,
        stage,
        context_text,
    )

    scores = {category: 0 for category in ISSUE_CATEGORIES}

    # Evidence from generic issue concepts.
    scores["DATA"] += 3 * _keyword_score(text, DATA_KEYWORDS)
    scores["DATABASE"] += 4 * _keyword_score(text, DATABASE_KEYWORDS)
    scores["MODEL"] += 4 * _keyword_score(text, MODEL_KEYWORDS)
    scores["INFRASTRUCTURE"] += 4 * _keyword_score(
        text, INFRASTRUCTURE_KEYWORDS
    )
    scores["CONFIGURATION"] += 4 * _keyword_score(
        text, CONFIGURATION_KEYWORDS
    )
    scores["PIPELINE"] += 3 * _keyword_score(
        text, PIPELINE_KEYWORDS
    )
    scores["CODE"] += 3 * _keyword_score(text, CODE_KEYWORDS)

    # Stage evidence.
    stage_text = (stage or "").lower()

    if any(term in stage_text for term in (
        "data", "validation", "profil", "input", "ingest", "parse"
    )):
        scores["DATA"] += 3

    if any(term in stage_text for term in (
        "model", "train", "predict", "inference"
    )):
        scores["MODEL"] += 3

    if any(term in stage_text for term in (
        "database", "db", "query", "sql"
    )):
        scores["DATABASE"] += 3

    if any(term in stage_text for term in (
        "config", "setup", "initialization", "environment"
    )):
        scores["CONFIGURATION"] += 3

    if any(term in stage_text for term in (
        "pipeline", "workflow", "orchestration", "job"
    )):
        scores["PIPELINE"] += 3

    # Exception hints are supporting evidence, not absolute rules.
    hinted_category = EXCEPTION_HINTS.get(error_type or "")
    if hinted_category:
        scores[hinted_category] += 2

    # ValueError is commonly caused by data problems, but the message
    # and stage can override this when stronger evidence exists.
    if error_type == "ValueError":
        scores["DATA"] += 2

    if not any(scores.values()):
        return "UNKNOWN"

    # Deterministic tie-breaking based on evidence strength.
    priority = [
        "DATA",
        "MODEL",
        "DATABASE",
        "CODE",
        "CONFIGURATION",
        "INFRASTRUCTURE",
        "PIPELINE",
        "UNKNOWN",
    ]

    return max(
        priority,
        key=lambda category: scores.get(category, 0),
    )


# ============================================================
# ERROR TYPE EXTRACTION
# ============================================================

def extract_error_type(message: str) -> str | None:
    """Extract a known Python exception type from a log message."""
    if not message:
        return None

    # Longest names first avoids partial matches.
    for error_type in sorted(
        COMMON_ERROR_TYPES,
        key=len,
        reverse=True,
    ):
        if re.search(
            rf"\b{re.escape(error_type)}\b",
            message,
        ):
            return error_type

    return None


# ============================================================
# FAILED STAGE DETECTION
# ============================================================

def detect_failed_stage(
    parsed_logs: list[dict[str, Any]]
) -> str | None:
    """Find the stage associated with the latest ERROR/CRITICAL log."""
    for log in reversed(parsed_logs):
        if log.get("level") in {"ERROR", "CRITICAL"}:
            return log.get("stage")

    return None


# ============================================================
# ERROR LOG DETECTION
# ============================================================

def get_error_logs(
    parsed_logs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return ERROR and CRITICAL log entries."""
    return [
        log
        for log in parsed_logs
        if log.get("level") in {"ERROR", "CRITICAL"}
    ]


# ============================================================
# WARNING DETECTION
# ============================================================

def get_warning_logs(
    parsed_logs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return WARNING logs because they may contain useful evidence."""
    return [
        log
        for log in parsed_logs
        if log.get("level") == "WARNING"
    ]


# ============================================================
# BUILD ISSUE SUMMARY
# ============================================================

def build_warning_issues(
    warning_logs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert warning log entries into structured issues."""
    issues = []

    for warning in warning_logs:
        message = warning.get("message", "")
        stage = warning.get("stage")

        issue = {
            "stage": stage,
            "type": "WARNING",
            "message": message,
            "category": classify_category(
                None,
                message,
                stage,
                context=[warning],
            ),
        }

        # Only expose a column/field when the log explicitly names it.
        column = _extract_column_reference(message)
        if column:
            issue["column"] = column

        issues.append(issue)

    return issues


# ============================================================
# RELEVANT CONTEXT
# ============================================================

def collect_relevant_context(
    parsed_logs: list[dict[str, Any]],
    failed_stage: str | None
) -> list[dict[str, Any]]:
    """
    Collect WARNING/INFO/ERROR/CRITICAL logs relevant to the
    failed stage without assuming a particular business domain.
    """

    if not failed_stage:
        return parsed_logs[-20:]

    relevant = [
        log
        for log in parsed_logs
        if log.get("stage") == failed_stage
    ]

    if len(relevant) < 10:
        try:
            stage_index = next(
                i
                for i, log in enumerate(parsed_logs)
                if log.get("stage") == failed_stage
            )

            start = max(0, stage_index - 5)
            end = min(len(parsed_logs), stage_index + 10)

            for log in parsed_logs[start:end]:
                if log not in relevant:
                    relevant.append(log)

        except StopIteration:
            pass

    return relevant


# ============================================================
# MAIN ERROR DETECTOR
# ============================================================

def detect_error(
    parsed_logs: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Analyze structured logs and return a generic troubleshooting
    context.

    WARNING logs are preserved as additional detected issues.
    ERROR/CRITICAL logs determine the primary failure.
    """

    if not parsed_logs:
        return {
            "pipeline_status": "SUCCESS",
            "failed_stage": None,
            "error_type": None,
            "error_message": None,
            "category": "UNKNOWN",
            "relevant_logs": [],
            "warning_issues": [],
            "error_logs": [],
            "warning_count": 0,
            "total_error_entries": 0,
            "total_warning_entries": 0,
        }

    error_logs = get_error_logs(parsed_logs)
    warning_logs = get_warning_logs(parsed_logs)
    warning_issues = build_warning_issues(warning_logs)

    if not error_logs:
        category = (
            classify_category(
                None,
                warning_issues[0]["message"],
                warning_issues[0].get("stage"),
                context=parsed_logs[-20:],
            )
            if warning_issues
            else "UNKNOWN"
        )

        return {
            "pipeline_status": "SUCCESS",
            "failed_stage": None,
            "error_type": None,
            "error_message": None,
            "category": category,
            "relevant_logs": parsed_logs[-20:],
            "warning_issues": warning_issues,
            "error_logs": [],
            "warning_count": len(warning_logs),
            "total_error_entries": 0,
            "total_warning_entries": len(warning_logs),
        }

    # Prefer ERROR because it normally carries the primary exception.
    primary_error = next(
        (
            log
            for log in error_logs
            if log.get("level") == "ERROR"
        ),
        error_logs[0],
    )

    error_message = primary_error.get("message", "")
    failed_stage = detect_failed_stage(parsed_logs)
    error_type = extract_error_type(error_message)

    relevant_logs = collect_relevant_context(
        parsed_logs,
        failed_stage,
    )

    # Include warnings because they can explain or accompany the failure.
    for warning in warning_logs:
        if warning not in relevant_logs:
            relevant_logs.append(warning)

    relevant_logs = relevant_logs[-30:]

    category = classify_category(
        error_type,
        error_message,
        failed_stage,
        context=relevant_logs,
    )

    report = {
        "pipeline_status": "FAILED",
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error_message": error_message,
        "category": category,
        "error_logs": error_logs,
        "warning_issues": warning_issues,
        "warning_count": len(warning_logs),
        "relevant_logs": relevant_logs,
        "total_error_entries": len(error_logs),
        "total_warning_entries": len(warning_logs),
    }

    # Only include an explicitly mentioned column/field.
    column = _extract_column_reference(error_message)
    if column:
        report["column"] = column

    return report
