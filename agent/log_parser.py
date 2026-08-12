import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"


# ============================================================
# LOG LINE PATTERN
# ============================================================

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2},\d{3})\s*\|\s*"
    r"(?P<level>INFO|WARNING|ERROR|CRITICAL)\s*\|\s*"
    r"(?P<pipeline>[^|]+?)\s*\|\s*"
    r"(?P<content>.*)$"
)


# ============================================================
# PARSE SINGLE LOG LINE
# ============================================================

def parse_log_line(line):
    """
    Parse one structured pipeline log line.

    Returns:
        dict if the line matches the expected format
        None if the line is malformed or unsupported
    """

    line = line.strip()

    if not line:
        return None

    match = LOG_PATTERN.match(line)

    if not match:
        return None

    timestamp = match.group("timestamp")
    level = match.group("level")
    pipeline = match.group("pipeline").strip()
    content = match.group("content").strip()

    # --------------------------------------------------------
    # Extract pipeline stage from:
    #
    # [DATA_CLEANING] Message
    # --------------------------------------------------------

    stage_match = re.match(
        r"^\[(?P<stage>[A-Z_]+)\]\s*(?P<message>.*)$",
        content
    )

    if stage_match:

        stage = stage_match.group("stage")
        message = stage_match.group("message").strip()

    else:

        # Pipeline-level messages such as:
        #
        # Pipeline started
        #
        stage = "PIPELINE"

        message = content

    return {
        "timestamp": timestamp,
        "level": level,
        "pipeline": pipeline,
        "stage": stage,
        "message": message
    }


# ============================================================
# PARSE COMPLETE LOG FILE
# ============================================================

def parse_log_file(log_file=LOG_FILE):
    """
    Read and parse the complete pipeline log file.

    Returns:
        list of structured log entries
    """

    if not log_file.exists():

        raise FileNotFoundError(
            f"Log file not found: {log_file}"
        )

    parsed_logs = []

    malformed_lines = []

    with log_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            parsed_entry = parse_log_line(line)

            if parsed_entry is None:

                # ------------------------------------------------
                # IMPORTANT:
                # Do not crash because of malformed log lines.
                # ------------------------------------------------

                if line.strip():

                    malformed_lines.append({
                        "line_number": line_number,
                        "content": line.strip()
                    })

                continue

            parsed_logs.append(parsed_entry)

    return parsed_logs, malformed_lines


# ============================================================
# IDENTIFY ERRORS
# ============================================================

def get_error_entries(parsed_logs):
    """
    Return ERROR and CRITICAL log entries.
    """

    return [
        entry
        for entry in parsed_logs
        if entry["level"] in {
            "ERROR",
            "CRITICAL"
        }
    ]


# ============================================================
# COLLECT CONTEXT
# ============================================================

def collect_context(
    parsed_logs,
    error_index,
    context_before=5,
    context_after=2
):
    """
    Collect surrounding log entries around an error.

    Example:

        INFO
        INFO
        WARNING
        ERROR   <-- target
        CRITICAL

    The function returns nearby entries to provide
    execution context for future troubleshooting.
    """

    start_index = max(
        0,
        error_index - context_before
    )

    end_index = min(
        len(parsed_logs),
        error_index + context_after + 1
    )

    return parsed_logs[start_index:end_index]


# ============================================================
# BUILD TROUBLESHOOTING CONTEXT
# ============================================================

def build_error_context(parsed_logs):
    """
    Identify ERROR/CRITICAL entries and collect
    surrounding log entries.
    """

    error_contexts = []

    for index, entry in enumerate(parsed_logs):

        if entry["level"] not in {
            "ERROR",
            "CRITICAL"
        }:
            continue

        surrounding_logs = collect_context(
            parsed_logs,
            index
        )

        error_contexts.append({
            "error": entry,
            "context": surrounding_logs
        })

    return error_contexts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI PIPELINE TROUBLESHOOTING AGENT")
    print("STEP 5 - LOG PARSER")
    print("=" * 70)

    print(f"\nReading log file:\n{LOG_FILE}\n")

    parsed_logs, malformed_lines = parse_log_file()

    print(
        f"Total parsed log entries: {len(parsed_logs)}"
    )

    print(
        f"Malformed lines skipped: {len(malformed_lines)}"
    )

    # --------------------------------------------------------
    # ERROR DETECTION
    # --------------------------------------------------------

    error_entries = get_error_entries(
        parsed_logs
    )

    print(
        f"ERROR/CRITICAL entries: {len(error_entries)}"
    )

    # --------------------------------------------------------
    # DISPLAY ERRORS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ERROR ENTRIES")
    print("-" * 70)

    for entry in error_entries:

        print(
            f"\nTimestamp : {entry['timestamp']}"
        )

        print(
            f"Level     : {entry['level']}"
        )

        print(
            f"Pipeline  : {entry['pipeline']}"
        )

        print(
            f"Stage     : {entry['stage']}"
        )

        print(
            f"Message   : {entry['message']}"
        )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    error_contexts = build_error_context(
        parsed_logs
    )

    print("\n" + "-" * 70)
    print("ERROR CONTEXT")
    print("-" * 70)

    for context_number, item in enumerate(
        error_contexts,
        start=1
    ):

        print(
            f"\nContext #{context_number}"
        )

        for entry in item["context"]:

            print(
                f"{entry['timestamp']} | "
                f"{entry['level']} | "
                f"{entry['stage']} | "
                f"{entry['message']}"
            )

    # --------------------------------------------------------
    # MALFORMED LOGS
    # --------------------------------------------------------

    if malformed_lines:

        print("\n" + "-" * 70)
        print("MALFORMED LOG LINES")
        print("-" * 70)

        for item in malformed_lines:

            print(
                f"Line {item['line_number']}: "
                f"{item['content']}"
            )

    print("\n" + "=" * 70)
    print("LOG PARSING COMPLETED")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()