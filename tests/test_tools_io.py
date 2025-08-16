from pathlib import Path

import pandas as pd
import pytest

from agent.tools import (
    describe_columns,
    load_csv,
    preview,
    save_table,
    schema_infer,
)


def test_load_csv_and_preview() -> None:
    df = load_csv(Path("tests/data/sales.csv"))
    assert df.shape == (4, 5)
    # preview respects cap & request
    assert preview(df, 2).shape[0] == 2
    assert preview(df, 999).shape[0] == 4  # capped at dataset size (<= 50)


def test_load_csv_caps() -> None:
    # Force a tiny max_cells to trigger the safety check
    with pytest.raises(ValueError):
        _ = load_csv(Path("tests/data/sales.csv"), max_cells=5)


def test_schema_infer() -> None:
    df = pd.read_csv("tests/data/sales.csv")
    sch = schema_infer(df)
    assert set(sch.keys()) == {"columns", "dtypes", "nulls"}
    assert "product" in sch["columns"]
    assert isinstance(sch["dtypes"]["year"], str)
    assert sch["nulls"]["product"] == 0


def test_save_table_roundtrip(tmp_path: Path) -> None:
    df = pd.read_csv("tests/data/sales.csv")
    out = save_table(df, tmp_path / "sales_out.csv")
    assert out.exists()
    back = pd.read_csv(out)
    assert len(back) == len(df)
    assert list(back.columns) == list(df.columns)


def test_describe_columns() -> None:
    df = pd.read_csv("tests/data/sales.csv")
    desc = describe_columns(df)
    # index is columns
    assert "product" in desc.index
    # required keys exist
    for key in ["dtype", "non_null", "nulls"]:
        assert key in desc.columns
