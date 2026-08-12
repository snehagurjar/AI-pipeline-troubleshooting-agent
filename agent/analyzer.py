import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from agent.log_parser import parse_log_file
from agent.error_detector import detect_error
from agent.data_profiler import profile_dataframe
from agent.troubleshooting_agent import (
    GroqProvider,
    TroubleshootingAgent,
)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class PipelineAnalysisError(Exception):
    """Base exception for pipeline analysis errors."""
    pass


class LogFileError(PipelineAnalysisError):
    """Raised when the log file cannot be read."""
    pass


class LLMAnalysisError(PipelineAnalysisError):
    """Raised when the LLM analysis fails."""
    pass


# ============================================================
# LOG READER
# ============================================================

def read_log_file(log_file_path: str) -> str:
    path = Path(log_file_path)

    if not path.exists():
        raise LogFileError(f"Log file not found: {log_file_path}")

    if not path.is_file():
        raise LogFileError(f"Log path is not a file: {log_file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LogFileError(f"Unable to read log file: {error}") from error

    if not content.strip():
        raise LogFileError(f"Log file is empty: {log_file_path}")

    return content


# ============================================================
# PARSE LOGS
# ============================================================

def parse_pipeline_logs(log_file_path: str):
    try:
        parsed_logs, malformed_lines = parse_log_file(Path(log_file_path))
    except Exception as error:
        raise PipelineAnalysisError(f"Log parsing failed: {error}") from error

    return parsed_logs, malformed_lines


# ============================================================
# DATASET CONTEXT
# ============================================================

def build_dataset_context(
    dataset_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Build generic dataset evidence for the troubleshooting context.

    This function is strictly read-only. It profiles and validates the
    supplied DataFrame/dataset and never cleans, drops, converts, or
    otherwise modifies the input data.
    """
    if dataframe is not None and dataset_path is not None:
        raise ValueError("Provide either dataframe or dataset_path, not both")

    if dataframe is None and dataset_path is None:
        return {}

    try:
        if dataframe is None:
            path = Path(dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            if path.suffix.lower() == ".csv":
                dataframe = pd.read_csv(path)
            else:
                raise ValueError(
                    "Dataset profiling currently supports CSV input. "
                    "Other file types can be added in the generic file-input phase."
                )

        # Import the generic validation function from the pipeline layer.
        from pipeline.data_pipeline import validate_data

        profile = profile_dataframe(dataframe)
        quality_issues = validate_data(dataframe)

        return {
            "dataset_profile": profile,
            "data_quality_issues": quality_issues,
        }

    except Exception as error:
        # Dataset context is supplementary evidence. A profiling failure
        # must not hide the primary pipeline/log analysis failure.
        return {
            "dataset_profile": None,
            "data_quality_issues": [],
            "dataset_context_error": f"Dataset context unavailable: {error}",
        }


# ============================================================
# BUILD ANALYSIS CONTEXT
# ============================================================

def build_analysis_context(
    parsed_logs: list[dict[str, Any]],
    malformed_lines: list[dict[str, Any]],
    dataset_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Combine deterministic log evidence with generic dataset evidence."""
    try:
        error_report = detect_error(parsed_logs)
    except Exception as error:
        raise PipelineAnalysisError(f"Error detection failed: {error}") from error

    error_report["total_parsed_logs"] = len(parsed_logs)
    error_report["malformed_log_lines"] = len(malformed_lines)
    error_report["malformed_log_examples"] = malformed_lines[:5]

    dataset_context = build_dataset_context(
        dataset_path=dataset_path,
        dataframe=dataframe,
    )

    # Keep the error-detector fields intact and add dataset evidence as
    # separate top-level fields for the existing LLM architecture.
    error_report.update(dataset_context)

    return error_report


# ============================================================
# HANDLE NO ERROR
# ============================================================

def create_success_report(error_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_status": "SUCCESS",
        "failed_stage": None,
        "error_type": None,
        "error_message": None,
        "issue_description": (
            "No ERROR or CRITICAL entries were detected in the pipeline log."
        ),
        "root_cause": (
            "No pipeline failure was detected from the available logs."
        ),
        "severity": "LOW",
        "confidence": 1.0,
        "recommended_actions": [],
        "suggested_fix": (
            "No fix is required based on the available logs."
        ),
        "prevention": (
            "Continue monitoring pipeline logs and maintain appropriate "
            "data and pipeline validation."
        ),
        "total_parsed_logs": error_report["total_parsed_logs"],
        "malformed_log_lines": error_report["malformed_log_lines"],
        # Preserve generic dataset evidence in successful runs too.
        "dataset_profile": error_report.get("dataset_profile"),
        "data_quality_issues": error_report.get("data_quality_issues", []),
    }


# ============================================================
# LLM ROUTING DECISION
# ============================================================

def should_run_llm_analysis(error_report: dict[str, Any]) -> bool:
    """
    Decide whether the structured context should be sent to the LLM
    troubleshooting workflow.

    The pipeline can FAIL on its own, or it can SUCCEED while the
    dataset itself still has quality issues worth reporting. Either
    case requires the LLM. A clean SUCCESS with no dataset-quality
    issues does not.
    """
    pipeline_failed = error_report.get("pipeline_status") == "FAILED"
    has_data_quality_issues = bool(error_report.get("data_quality_issues"))

    return pipeline_failed or has_data_quality_issues


# ============================================================
# RUN LLM ANALYSIS
# ============================================================

def run_llm_analysis(error_report: dict[str, Any]):
    try:
        provider = GroqProvider()
        troubleshooting_agent = TroubleshootingAgent(provider=provider)
        return troubleshooting_agent.analyze(error_report)
    except Exception as error:
        raise LLMAnalysisError(
            f"LLM troubleshooting analysis failed: {error}"
        ) from error


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_pipeline_log(
    log_file_path: str,
    dataset_path: str | Path | None = None,
):
    """
    End-to-end workflow with optional generic dataset evidence.

    Log File -> Parse -> Error Detector -> Dataset Profiler/Validator
    -> Combined Structured Context -> Existing Groq Agent -> Pydantic
    """
    raw_logs = read_log_file(log_file_path)

    if not raw_logs.strip():
        raise LogFileError(
            "The supplied log file contains no usable content."
        )

    parsed_logs, malformed_lines = parse_pipeline_logs(log_file_path)

    if not parsed_logs:
        raise PipelineAnalysisError(
            "No valid log entries could be parsed from the log file."
        )

    error_report = build_analysis_context(
        parsed_logs,
        malformed_lines,
        dataset_path=dataset_path,
    )

    if should_run_llm_analysis(error_report):
        return run_llm_analysis(error_report)

    return create_success_report(error_report)


# ============================================================
# DISPLAY REPORT
# ============================================================

def print_report(report):
    print("\n")
    print("=" * 80)
    print("FINAL PIPELINE TROUBLESHOOTING REPORT")
    print("=" * 80)

    if hasattr(report, "model_dump_json"):
        print(report.model_dump_json(indent=2))
    else:
        print(json.dumps(report, indent=2, default=str))

    print("=" * 80)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    print("=" * 80)
    print("AI PIPELINE TROUBLESHOOTING AGENT")
    print("STEP 8 - COMPLETE WORKFLOW")
    print("=" * 80)

    log_file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join("logs", "pipeline.log")
    )

    # Optional second argument: dataset path.
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nAnalyzing log file: {log_file_path}")
    if dataset_path:
        print(f"Dataset evidence: {dataset_path}")

    try:
        report = analyze_pipeline_log(
            log_file_path,
            dataset_path=dataset_path,
        )
        print_report(report)

    except LogFileError as error:
        print("\nLOG FILE ERROR")
        print("-" * 80)
        print(str(error))

    except LLMAnalysisError as error:
        print("\nLLM ANALYSIS ERROR")
        print("-" * 80)
        print(str(error))

    except PipelineAnalysisError as error:
        print("\nPIPELINE ANALYSIS ERROR")
        print("-" * 80)
        print(str(error))

    except Exception as error:
        print("\nUNEXPECTED ERROR")
        print("-" * 80)
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
