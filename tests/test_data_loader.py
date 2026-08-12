import pandas as pd
import pytest

from agent.data_loader import (
    DataFileNotFoundError,
    EmptyDataFileError,
    UnsupportedFileTypeError,
    load_data,
)


def test_load_csv(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text(
        "employee_code,score,active\nE01,91,True\nE02,84,False\n",
        encoding="utf-8",
    )

    df = load_data(source)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["employee_code", "score", "active"]
    assert df.shape == (2, 3)
    assert df.iloc[0]["employee_code"] == "E01"


def test_load_xlsx(tmp_path):
    source = tmp_path / "sample.xlsx"

    expected = pd.DataFrame(
        {
            "code": ["A1", "A2"],
            "amount": [100.5, 200.0],
            "status": ["open", "closed"],
        }
    )
    expected.to_excel(source, index=False)

    df = load_data(source)

    pd.testing.assert_frame_equal(df, expected)


def test_unsupported_file_type(tmp_path):
    source = tmp_path / "sample.json"
    source.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        load_data(source)


def test_missing_file(tmp_path):
    with pytest.raises(DataFileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_empty_file(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_bytes(b"")

    with pytest.raises(EmptyDataFileError):
        load_data(source)


def test_loader_does_not_modify_loaded_values(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text(
        "name,value\nalpha,10\nbeta,20\n",
        encoding="utf-8",
    )

    df = load_data(source)

    assert df.to_dict(orient="records") == [
        {"name": "alpha", "value": 10},
        {"name": "beta", "value": 20},
    ]
