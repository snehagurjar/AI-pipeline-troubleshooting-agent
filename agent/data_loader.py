"""
Generic file/data input layer.

Responsibilities:
    file -> pandas.DataFrame

Supported formats:
    CSV, XLSX, XLS

This module intentionally does NOT clean, validate, profile, or
troubleshoot the loaded data.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DataLoaderError(Exception):
    """Base exception for data loading failures."""


class UnsupportedFileTypeError(DataLoaderError):
    """Raised when the supplied file extension is unsupported."""


class DataFileNotFoundError(DataLoaderError):
    """Raised when the supplied file does not exist."""


class EmptyDataFileError(DataLoaderError):
    """Raised when a file contains no readable tabular data."""


class MalformedDataFileError(DataLoaderError):
    """Raised when a supported file cannot be parsed."""


PathLikeInput = Union[str, Path, BinaryIO]


def _get_extension(file_source: PathLikeInput) -> str:
    """Return a lowercase extension when the source has one."""
    if isinstance(file_source, (str, Path)):
        return Path(file_source).suffix.lower()

    name = getattr(file_source, "name", "")
    return Path(str(name)).suffix.lower()


def _validate_path(file_source: str | Path) -> Path:
    path = Path(file_source)

    if not path.exists():
        raise DataFileNotFoundError(f"Data file not found: {path}")

    if not path.is_file():
        raise DataLoaderError(f"Data path is not a file: {path}")

    if path.stat().st_size == 0:
        raise EmptyDataFileError(f"Data file is empty: {path}")

    return path


def _read_csv(source: PathLikeInput) -> pd.DataFrame:
    """Read CSV with a small, controlled encoding fallback."""
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue

    raise MalformedDataFileError(
        f"Unable to decode CSV file using supported encodings: {encodings}"
    ) from last_error


def _read_excel(source: PathLikeInput, extension: str) -> pd.DataFrame:
    """Read XLSX/XLS while preserving pandas' normal parsing behavior."""
    try:
        return pd.read_excel(source, engine="openpyxl" if extension == ".xlsx" else "xlrd")
    except ImportError as error:
        dependency = "openpyxl" if extension == ".xlsx" else "xlrd"
        raise DataLoaderError(
            f"Excel support requires the '{dependency}' package. "
            f"Install it with: pip install {dependency}"
        ) from error


def load_data(file_source: PathLikeInput) -> pd.DataFrame:
    """
    Load a CSV/XLSX/XLS file into a pandas DataFrame.

    The function is read-only with respect to the source file and does not
    perform data cleaning, validation, profiling, or troubleshooting.

    `file_source` may be:
      - a filesystem path
      - pathlib.Path
      - a file-like object such as Streamlit's UploadedFile
    """
    extension = _get_extension(file_source)

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file format: '{extension or 'unknown'}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if isinstance(file_source, (str, Path)):
        source = _validate_path(file_source)
    else:
        source = file_source

        if hasattr(source, "seek"):
            source.seek(0)

        # Empty uploaded/file-like object check where supported.
        if hasattr(source, "getvalue"):
            if not source.getvalue():
                raise EmptyDataFileError("Uploaded data file is empty.")
        elif hasattr(source, "read"):
            current = source.tell() if hasattr(source, "tell") else None
            content = source.read(1)
            if content in (b"", ""):
                raise EmptyDataFileError("Data file is empty.")
            if current is not None:
                source.seek(current)

    try:
        if extension == ".csv":
            dataframe = _read_csv(source)
        else:
            dataframe = _read_excel(source, extension)
    except DataLoaderError:
        raise
    except (pd.errors.EmptyDataError, EOFError) as error:
        raise EmptyDataFileError(
            f"No readable tabular data was found in '{getattr(file_source, 'name', file_source)}'."
        ) from error
    except Exception as error:
        raise MalformedDataFileError(
            f"Unable to parse '{getattr(file_source, 'name', file_source)}' "
            f"as {extension} data: {error}"
        ) from error

    if dataframe is None or dataframe.empty and len(dataframe.columns) == 0:
        raise EmptyDataFileError(
            f"No readable tabular data was found in '{getattr(file_source, 'name', file_source)}'."
        )

    return dataframe
