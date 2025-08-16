from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

# -------------------------
# Safety caps / constants
# -------------------------
MAX_CELLS_DEFAULT = 2_000_000  # rows * cols
MAX_PREVIEW_ROWS = 50
MAX_GROUP_ROWS = 1_000
MAX_PLOT_POINTS = 2_000
ARTIFACTS_DIR = Path("artifacts")


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# CSV I/O + schema helpers
# -------------------------
def load_csv(
    path_or_url: str | Path,
    *,
    max_cells: int = MAX_CELLS_DEFAULT,
    dtype_backend: str = "numpy_nullable",
) -> pd.DataFrame:
    """Read CSV (local/URL). Enforce a max cell cap."""
    df = pd.read_csv(path_or_url, low_memory=False, dtype_backend=dtype_backend)
    if int(df.size) > max_cells:
        raise ValueError(
            f"CSV too large for safety: {df.size:,} cells > cap {max_cells:,}. "
            "Reduce file or sample first."
        )
    return df


def schema_infer(df: pd.DataFrame, *, sample_rows: int = 1_000) -> dict[str, Any]:
    """Return columns, dtypes, and null counts (using a small head sample)."""
    sample = df.head(sample_rows)
    return {
        "columns": list(sample.columns),
        "dtypes": {c: str(sample[c].dtype) for c in sample.columns},
        "nulls": {c: int(sample[c].isna().sum()) for c in sample.columns},
    }


def preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a small preview (capped)."""
    n = max(0, min(n, MAX_PREVIEW_ROWS))
    return df.head(n if n > 0 else 5)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save CSV into artifacts/ by default; return final path."""
    _ensure_artifacts_dir()
    out = Path(path)
    if out.parent == Path("."):
        out = ARTIFACTS_DIR / out.name
    df.to_csv(out, index=False)
    return out


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary: dtype/nulls; stats for numeric; uniques/top for text."""
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        item: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
        }
        if is_numeric_dtype(s):
            if s.notna().any():
                item.update(
                    {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }
                )
            else:
                item.update({"mean": None, "std": None, "min": None, "max": None})
        else:
            item.update(
                {
                    "unique": int(s.nunique(dropna=True)),
                    "top": (s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None),
                }
            )
        rows.append(item)
    return pd.DataFrame(rows).set_index("column")


# -------------------------
# Analysis tools
# -------------------------
def filter_df(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """
    Filter rows using a pandas query expression.
    Example: "year == 2024 and region in ['NA','EU'] and price > 0"
    """
    try:
        return df.query(expr, engine="python")
    except Exception as err:
        # keep original cause for debuggability
        raise ValueError(f"Invalid filter expression: {expr}") from err


def groupby_agg(df: pd.DataFrame, by: list[str], metrics: dict[str, str]) -> pd.DataFrame:
    """
    Group by columns and aggregate numeric columns with given metrics.
    Example: by=['product'], metrics={'qty':'sum','price':'mean'}
    """
    for col in by:
        if col not in df.columns:
            raise ValueError(f"Missing group-by column: {col}")
    for col in metrics:
        if col not in df.columns:
            raise ValueError(f"Missing metric column: {col}")

    grouped = df.groupby(by, dropna=False).agg(metrics).reset_index()
    if len(grouped) > MAX_GROUP_ROWS:
        grouped = grouped.head(MAX_GROUP_ROWS)
    return grouped


def plot(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> Path:
    """
    Create a simple chart and save it into artifacts/.
    kind: 'line' | 'bar' | 'hist' | 'scatter'
    Caps points to keep charts readable & fast.
    """
    _ensure_artifacts_dir()

    if x not in df.columns:
        raise ValueError(f"Missing x column: {x}")
    if y is not None and y not in df.columns and kind in {"line", "bar", "scatter"}:
        raise ValueError(f"Missing y column: {y}")
    if hue is not None and hue not in df.columns:
        raise ValueError(f"Missing hue column: {hue}")

    # downsample for readability
    sample = df
    if kind in {"line", "scatter"} and len(df) > MAX_PLOT_POINTS:
        sample = df.sample(MAX_PLOT_POINTS, random_state=42)

    fig, ax = plt.subplots()

    if kind == "line":
        if hue:
            for k, part in sample.groupby(hue):
                ax.plot(part[x], part[y], label=str(k))
            ax.legend()
        else:
            ax.plot(sample[x], sample[y])
    elif kind == "bar":
        if hue:
            small = sample.groupby([x, hue])[y].sum().unstack(fill_value=0)
            small.plot(kind="bar", ax=ax)
        else:
            small = sample.groupby(x)[y].sum()
            small.plot(kind="bar", ax=ax)
    elif kind == "hist":
        ax.hist(sample[x].dropna())
    elif kind == "scatter":
        if y is None:
            raise ValueError("scatter requires y")
        if hue:
            for k, part in sample.groupby(hue):
                ax.scatter(part[x], part[y], label=str(k))
            ax.legend()
        else:
            ax.scatter(sample[x], sample[y])
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported plot kind: {kind}")

    ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    # save
    name = filename or f"{kind}_{x}{'_' + y if y else ''}.png"
    out = ARTIFACTS_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
