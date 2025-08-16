from pathlib import Path

import pandas as pd
import pytest

from agent.tools import filter_df, groupby_agg, load_csv, plot


def _load() -> pd.DataFrame:
    return load_csv(Path("tests/data/sales.csv"))


def test_filter_df_simple() -> None:
    df = _load()
    out = filter_df(df, "year == 2024 and qty > 3")
    # rows: (A,NA,2024,10,2.5) and (B,EU,2024,5,3.0)
    assert len(out) == 2
    assert set(out["product"]) == {"A", "B"}

    with pytest.raises(ValueError):
        _ = filter_df(df, "year === 2024")  # invalid expr


def test_groupby_agg_sum_mean() -> None:
    df = _load()
    agg = groupby_agg(df, by=["product"], metrics={"qty": "sum", "price": "mean"})
    # products A,B,C exist
    assert set(agg["product"]) == {"A", "B", "C"}
    # A has qty 17 (10 + 7)
    a_row = agg[agg["product"] == "A"].iloc[0]
    assert int(a_row["qty"]) == 17


def test_plot_bar_and_scatter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # save into tmp directory by monkeypatching artifacts dir at runtime
    from agent import tools as t
    monkeypatch.setattr(t, "ARTIFACTS_DIR", tmp_path)

    df = _load()
    out1 = plot(df, kind="bar", x="product", y="qty", title="qty by product")
    out2 = plot(df, kind="scatter", x="qty", y="price", title="qty vs price")
    assert out1.exists() and out1.suffix == ".png"
    assert out2.exists() and out2.suffix == ".png"
