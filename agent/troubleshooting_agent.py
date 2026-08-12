import json
import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import TroubleshootingReport


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


# ============================================================
# LLM PROVIDER ABSTRACTION
# ============================================================

class LLMProvider(ABC):
    """
    Abstract interface for an LLM provider.

    The troubleshooting agent depends on this interface,
    not directly on Groq.
    """

    @abstractmethod
    def analyze(
        self,
        system_prompt: str,
        input_data: dict[str, Any]
    ) -> TroubleshootingReport:
        """
        Analyze structured troubleshooting information.
        """
        raise NotImplementedError


# ============================================================
# GROQ PROVIDER
# ============================================================

class GroqProvider(LLMProvider):
    """
    Groq implementation of the LLM provider.
    """

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to the .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

    def analyze(
        self,
        system_prompt: str,
        input_data: dict[str, Any]
    ) -> TroubleshootingReport:

        # ----------------------------------------------------
        # Convert structured error report into JSON
        # ----------------------------------------------------

        user_message = (
            "Analyze the following structured pipeline "
            "troubleshooting information.\n\n"

            "IMPORTANT:\n"
            "- Use only the supplied evidence.\n"
            "- Clearly distinguish observed facts from "
            "probable inference.\n"
            "- Do not invent missing information.\n"
            "- Do not claim that any action was executed.\n\n"

            "INPUT:\n"
            f"{json.dumps(input_data, indent=2)}"
        )

        # ----------------------------------------------------
        # Generate JSON schema from Pydantic
        # ----------------------------------------------------

        schema = TroubleshootingReport.model_json_schema()

        # ----------------------------------------------------
        # Call Groq
        # ----------------------------------------------------

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "troubleshooting_report",
                    "strict": True,
                    "schema": schema
                }
            }
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            result = json.loads(content)

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Groq returned invalid JSON."
            ) from error

        # ----------------------------------------------------
        # Validate with Pydantic
        # ----------------------------------------------------

        try:

            validated_result = (
                TroubleshootingReport.model_validate(
                    result
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Groq response failed Pydantic validation."
            ) from error

        return validated_result


# ============================================================
# TROUBLESHOOTING AGENT
# ============================================================

class TroubleshootingAgent:
    """
    Main troubleshooting agent.

    The agent is provider-independent.
    """

    def __init__(
        self,
        provider: LLMProvider
    ):

        self.provider = provider

    def analyze(
        self,
        error_report: dict[str, Any]
    ) -> TroubleshootingReport:

        if not error_report:

            raise ValueError(
                "Error report cannot be empty."
            )

        return self.provider.analyze(
            system_prompt=SYSTEM_PROMPT,
            input_data=error_report
        )


# ============================================================
# LOAD ERROR REPORT
# ============================================================

def get_error_report():

    from agent.error_detector import detect_error
    from agent.log_parser import parse_log_file

    parsed_logs, malformed_lines = (
        parse_log_file()
    )

    error_report = detect_error(
        parsed_logs
    )

    # Add parser metadata
    error_report[
        "malformed_log_lines"
    ] = len(malformed_lines)

    return error_report


# ============================================================
# MAIN
# ============================================================
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AI PIPELINE TROUBLESHOOTING AGENT")
    print("STEP 7 - GROQ TROUBLESHOOTING AGENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Get structured error information
    # --------------------------------------------------------

    print("\nReading pipeline logs...")

    error_report = get_error_report()

    print(
        f"Pipeline status: "
        f"{error_report['pipeline_status']}"
    )

    print(
        f"Failed stage: "
        f"{error_report['failed_stage']}"
    )

    print(
        f"Error type: "
        f"{error_report['error_type']}"
    )

    print(
        f"Category: "
        f"{error_report['category']}"
    )

    # --------------------------------------------------------
    # Create Groq provider
    # --------------------------------------------------------

    provider = GroqProvider()

    # --------------------------------------------------------
    # Create troubleshooting agent
    # --------------------------------------------------------

    agent = TroubleshootingAgent(
        provider=provider
    )

    # --------------------------------------------------------
    # Send structured data to Groq
    # --------------------------------------------------------

    print(
        "\nSending structured failure "
        "information to Groq..."
    )

    result = agent.analyze(
        error_report
    )
    
    print("DEBUG: Groq response received")
    print("DEBUG RESULT:", result)
    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TROUBLESHOOTING REPORT")
    print("=" * 80)

    print(
        result.model_dump_json(
            indent=2
        )
    )

    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()