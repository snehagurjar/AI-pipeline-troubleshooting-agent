"""Streamlit UI for the generic AI Pipeline Troubleshooting Agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ============================================================
# PROJECT ROOT
# ============================================================

# streamlit_app.py is inside:
# troubleshooting-agent/app/
#
# Project root is:
# troubleshooting-agent/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import streamlit as st


def load_custom_css() -> None:
    """Load the custom Streamlit UI styling."""
    css_path = PROJECT_ROOT / "app" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, icon: str, tone: str = "default") -> None:
    """Render a compact dashboard metric card."""
    st.markdown(
        f"""
        <div class=\"metric-card {tone}\">
            <div class=\"metric-icon\">{icon}</div>
            <div class=\"metric-content\">
                <div class=\"metric-label\">{label}</div>
                <div class=\"metric-value\">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_issue_card(number: int, issue: dict[str, Any]) -> None:
    """Render one styled data-quality issue card."""
    severity = get_display_severity(issue)
    severity_class = severity.lower()
    issue_type = str(issue.get("type", "Data quality issue")).replace("_", " ").title()
    location = issue.get("column") or issue.get("stage") or "Not specified"
    evidence_items = [
        f"{key}={value}"
        for key, value in issue.items()
        if key not in {"type", "column", "stage"}
    ]
    evidence = ", ".join(evidence_items) or "Observed in dataset validation."

    st.markdown(
        f"""
        <div class=\"issue-card {severity_class}\">
            <div class=\"issue-top\">
                <div>
                    <span class=\"issue-number\">#{number}</span>
                    <span class=\"issue-title\">{issue_type}</span>
                </div>
                <span class=\"severity-badge {severity_class}\">{severity}</span>
            </div>
            <div class=\"issue-grid\">
                <div><span class=\"issue-key\">Location</span><span>{location}</span></div>
                <div><span class=\"issue-key\">Evidence</span><span>{evidence}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


from agent.data_loader import (
    DataLoaderError,
    EmptyDataFileError,
    MalformedDataFileError,
    UnsupportedFileTypeError,
    load_data,
)

from agent.data_profiler import profile_dataframe
from agent.error_detector import detect_error
from pipeline.data_pipeline import validate_data


# ============================================================
# DATA QUALITY ISSUE -> PIPELINE LOG EVIDENCE
# ============================================================

def _issue_to_log(
    issue: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Represent one generic data-quality issue as pipeline evidence."""

    issue_type = str(
        issue.get("type", "data_quality_issue")
    )

    column = issue.get("column")

    details = []

    if column:
        details.append(f"column={column}")

    for key in (
        "count",
        "percentage",
        "duplicate_value_rows",
        "parse_ratio",
        "lower_bound",
        "upper_bound",
    ):
        if key in issue:
            details.append(f"{key}={issue[key]}")

    message = f"Data quality issue detected: {issue_type}"

    if details:
        message += " | " + " | ".join(details)

    return {
        "timestamp": f"upload-{index}",
        "level": "WARNING",
        "pipeline": "GENERIC_DATA_PIPELINE",
        "stage": "DATA_VALIDATION",
        "message": message,
    }


# ============================================================
# BUILD DATASET ANALYSIS CONTEXT
# ============================================================

def build_upload_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build generic analysis context using the existing backend.

    This function does NOT modify the uploaded DataFrame.
    """

    profile = profile_dataframe(dataframe)

    issues = validate_data(dataframe)

    parsed_logs = [
        _issue_to_log(issue, index)
        for index, issue in enumerate(issues, start=1)
    ]

    # Run existing deterministic error detector.
    detected = detect_error(parsed_logs)

    return {
        **detected,
        "dataset_profile": profile,
        "data_quality_issues": issues,
        "upload_analysis": True,
    }


# ============================================================
# ISSUE SEVERITY
# ============================================================

def get_display_severity(
    issue: dict[str, Any],
) -> str:
    """Map generic issue types to display severity."""

    issue_type = str(
        issue.get("type", "")
    ).lower()

    if issue_type in {
        "empty_dataset",
        "empty_schema",
        "empty_columns",
    }:
        return "HIGH"

    if issue_type in {
        "duplicate_rows",
        "duplicate_possible_identifier",
        "malformed_datetime_like_values",
        "suspicious_mixed_type",
        "numeric_outliers",
        "schema_mismatch",
    }:
        return "MEDIUM"

    if issue_type in {
        "high_missing_percentage",
        "missing_values",
        "constant_columns",
    }:
        return "LOW"

    return "MEDIUM"


# ============================================================
# LLM TROUBLESHOOTING ASSESSMENT
# ============================================================

def run_troubleshooting_assessment(
    context: dict[str, Any],
):
    """
    Send generic dataset evidence to the existing
    Groq/Pydantic troubleshooting agent.

    No fixes are executed.
    """

    issues = context.get(
        "data_quality_issues",
        [],
    )

    if not issues:
        return None

    llm_context = dict(context)

    # IMPORTANT: This upload-and-analyze flow is detection-only.
    # validate_data() never raises and never halts execution - it only
    # logs WARNING-level data-quality observations. There is no actual
    # pipeline failure here, so pipeline_status must reflect that
    # ("SUCCESS"), matching the same convention used by the CLI
    # workflow (pipeline/data_pipeline.py + agent/analyzer.py) for a
    # dataset that completes but still has quality issues worth
    # reporting. Falsely marking this "FAILED" caused the LLM to
    # fabricate claims like "the pipeline stopped", since the system
    # prompt treats pipeline_status/failed_stage/error_message as
    # observed fact, not inference.
    llm_context["pipeline_status"] = "SUCCESS"
    llm_context["failed_stage"] = None

    llm_context["error_type"] = None

    llm_context["error_message"] = (
        "No pipeline execution error occurred. The dataset was "
        "analyzed by a read-only data-quality validator, which "
        "produced the WARNING-level issues listed in "
        "data_quality_issues below. These are data-quality "
        "observations only, not a runtime failure."
    )

    llm_context["category"] = "DATA"

    llm_context["warning_issues"] = context.get(
        "warning_issues",
        [],
    )

    # Existing LLM architecture.
    from agent.analyzer import run_llm_analysis

    return run_llm_analysis(llm_context)


# ============================================================
# RENDER TROUBLESHOOTING REPORT
# ============================================================

def _render_report(report: Any) -> None:
    """Render the existing troubleshooting report."""

    if hasattr(report, "model_dump"):
        data = report.model_dump()
    else:
        data = report

    st.subheader("Troubleshooting Analysis")

    st.markdown(
        f"**Issue Description**  \n"
        f"{data.get('issue_description', 'Not available')}"
    )

    st.markdown(
        f"**Probable Root Cause**  \n"
        f"{data.get('root_cause', 'Not available')}"
    )

    # --------------------------------------------------------
    # Recommended Actions
    # --------------------------------------------------------

    st.markdown("**Recommended Actions**")

    actions = data.get(
        "recommended_actions"
    ) or []

    if actions:
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.write(
            "No additional actions were recommended "
            "based on the available evidence."
        )

    # --------------------------------------------------------
    # Suggested Fix
    # --------------------------------------------------------

    st.markdown(
        f"**Suggested Fix**  \n"
        f"{data.get('suggested_fix', 'Not available')}"
    )

    # --------------------------------------------------------
    # Prevention
    # --------------------------------------------------------

    st.markdown(
        f"**Prevention**  \n"
        f"{data.get('prevention', 'Not available')}"
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity = data.get("severity")

    if severity:
        st.markdown(
            f"**Severity:** {severity}"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = data.get("confidence")

    if isinstance(confidence, (int, float)):
        confidence = max(
            0.0,
            min(1.0, float(confidence)),
        )

        st.progress(
            confidence,
            text=f"Confidence: {confidence:.0%}",
        )
    else:
        st.write(
            "Confidence: Not available"
        )


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main() -> None:

    st.set_page_config(
        page_title="AI Pipeline Troubleshooting Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_custom_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">⌁</div>
            <div>
                <div class="eyebrow">AI • DATA QUALITY • TROUBLESHOOTING</div>
                <h1>AI Pipeline Troubleshooting Agent</h1>
                <p>Upload any tabular dataset and get evidence-based data quality insights and troubleshooting guidance.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-row">
            <span>✓ Generic datasets</span>
            <span>✓ CSV & Excel</span>
            <span>✓ Read-only analysis</span>
            <span>✓ AI-assisted recommendations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        """
        <div class="section-heading">
            <div><span class="section-kicker">01</span><span class="section-title">Upload dataset</span></div>
            <span class="section-hint">CSV • XLSX • XLS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your dataset here or browse files",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX, XLS",
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">↑</div>
                <h3>Ready when you are</h3>
                <p>Upload a CSV or Excel file to start the analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        dataframe = load_data(
            uploaded_file
        )

    except UnsupportedFileTypeError as error:

        st.error(
            f"Unsupported file type: {error}"
        )

        return

    except EmptyDataFileError as error:

        st.error(
            f"The uploaded file is empty: {error}"
        )

        return

    except MalformedDataFileError as error:

        st.error(
            f"The uploaded file could not be parsed: {error}"
        )

        return

    except DataLoaderError as error:

        st.error(
            f"Unable to load the dataset: {error}"
        )

        return

    except Exception as error:

        st.error(
            "Unexpected file-loading error: "
            f"{type(error).__name__}: {error}"
        )

        return

    # ========================================================
    # STORE ORIGINAL DATAFRAME
    # ========================================================

    # Store the original DataFrame only.
    # No cleaning or modification is performed.

    st.session_state["uploaded_dataframe"] = dataframe
    st.session_state["uploaded_filename"] = uploaded_file.name

    # ========================================================
    # FILE INFORMATION
    # ========================================================

    file_extension = uploaded_file.name.rsplit(".", 1)[-1].upper()

    st.markdown(
        f"""
        <div class="file-banner">
            <div class="file-symbol">▦</div>
            <div class="file-details">
                <strong>{uploaded_file.name}</strong>
                <span>{file_extension} file • {len(dataframe):,} rows • {len(dataframe.columns):,} columns</span>
            </div>
            <div class="file-status">LOADED</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-heading compact">
            <div><span class="section-kicker">02</span><span class="section-title">Data preview</span></div>
            <span class="section-hint">First 10 rows</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        dataframe.head(10),
        use_container_width=True,
    )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.markdown("<div class=\"analyze-wrap\">", unsafe_allow_html=True)
    if st.button(
        "Analyze Dataset  →",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing dataset..."
        ):

            try:

                # --------------------------------------------
                # Generic deterministic analysis
                # --------------------------------------------

                context = build_upload_analysis(
                    dataframe
                )

                st.session_state[
                    "analysis_context"
                ] = context

                st.session_state.pop(
                    "troubleshooting_report",
                    None,
                )

                # --------------------------------------------
                # LLM analysis only when issues exist
                # --------------------------------------------

                if context.get(
                    "data_quality_issues"
                ):

                    report = (
                        run_troubleshooting_assessment(
                            context
                        )
                    )

                    st.session_state[
                        "troubleshooting_report"
                    ] = report

            except Exception as error:

                st.error(
                    "Analysis failed: "
                    f"{type(error).__name__}: {error}"
                )

                return

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # GET ANALYSIS RESULT
    # ========================================================

    context = st.session_state.get(
        "analysis_context"
    )

    if not context:
        return

    profile = (
        context.get(
            "dataset_profile"
        ) or {}
    )

    quality = (
        profile.get(
            "quality_summary",
            {},
        )
    )

    columns = (
        profile.get(
            "columns",
            {},
        )
    )

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.markdown(
        """
        <div class="section-heading">
            <div><span class="section-kicker">03</span><span class="section-title">Dataset overview</span></div>
            <span class="section-hint">Automated profile</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_info = (
        profile.get(
            "dataset",
            {},
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("Rows", f"{dataset_info.get('rows', len(dataframe)):,}", "▤")
    with c2:
        render_metric_card("Columns", f"{dataset_info.get('columns', len(dataframe.columns)):,}", "⌗")
    with c3:
        render_metric_card("Missing values", f"{int(dataframe.isna().sum().sum()):,}", "◇", "warning" if dataframe.isna().sum().sum() else "success")
    with c4:
        render_metric_card("Duplicate rows", f"{int(dataframe.duplicated().sum()):,}", "⧉", "warning" if dataframe.duplicated().sum() else "success")

    # ========================================================
    # DATA TYPES
    # ========================================================

    dtype_table = pd.DataFrame(
        [
            {
                "Column": name,
                "Data Type": info.get(
                    "dtype"
                ),
                "Inferred Type": info.get(
                    "inferred_type"
                ),
                "Missing": info.get(
                    "missing_count",
                    0,
                ),
            }
            for name, info in columns.items()
        ]
    )

    st.markdown(
        """
        <div class="subsection-title">Data types & missing values</div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        dtype_table,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DETECTED ISSUES
    # ========================================================

    st.markdown(
        """
        <div class="section-heading">
            <div><span class="section-kicker">04</span><span class="section-title">Detected issues</span></div>
            <span class="section-hint">Evidence from validation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    issues = context.get(
        "data_quality_issues",
        [],
    )

    if not issues:

        st.markdown(
            """
            <div class="success-state">
                <div class="success-icon">✓</div>
                <div><strong>No significant issues detected</strong><span>The available validation checks found no data-quality problems.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for number, issue in enumerate(
            issues,
            start=1,
        ):

            severity = (
                get_display_severity(
                    issue
                )
            )

            issue_type = (
                str(
                    issue.get(
                        "type",
                        "Data quality issue",
                    )
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            location = (
                issue.get("column")
                or issue.get("stage")
                or "Not specified"
            )

            evidence = ", ".join(
                f"{key}={value}"
                for key, value in issue.items()
                if key not in {
                    "type",
                    "column",
                    "stage",
                }
            )

            if not evidence:

                evidence = (
                    "Observed in dataset validation."
                )

            render_issue_card(number, issue)

    # ========================================================
    # TROUBLESHOOTING REPORT
    # ========================================================

    report = st.session_state.get(
        "troubleshooting_report"
    )

    if report is not None:

        st.markdown(
            """
            <div class="section-heading">
                <div><span class="section-kicker">05</span><span class="section-title">AI troubleshooting analysis</span></div>
                <span class="section-hint">Evidence-based guidance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_report(
            report
        )

        st.info(
            "Recommendations are evidence-based only. "
            "No data cleaning or fixes were executed."
        )

    elif issues:

        st.warning(
            "Issues were detected, but no troubleshooting "
            "report is available."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()