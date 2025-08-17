from __future__ import annotations

from pathlib import Path
from typing import Any

# Use headless backend for CI/tests (safe to leave enabled)
import matplotlib

matplotlib.use("Agg")
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
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def _ensure_artifacts_dir() -> None:
    """Make sure artifacts/ and artifacts/plots/ exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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
    except Exception as err:  # keep original cause for debuggability
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


# -------------------------
# Visualization tool (Day 6)
# -------------------------
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
    Create a simple chart and save it into artifacts/plots/.
    kind: 'line' | 'bar' | 'hist' | 'scatter'
    Caps points to keep charts readable & fast.
    """
    _ensure_artifacts_dir()

    if x not in df.columns:
        raise ValueError(f"Missing x column: {x}")
    if kind in {"line", "bar", "scatter"} and (y is None or y not in df.columns):
        raise ValueError(f"{kind} requires a valid y column")
    if hue is not None and hue not in df.columns:
        raise ValueError(f"Missing hue column: {hue}")

    # downsample for readability
    sample = df if len(df) <= MAX_PLOT_POINTS else df.sample(MAX_PLOT_POINTS, random_state=42)

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
            sample.groupby(x)[y].sum().plot(kind="bar", ax=ax)
    elif kind == "hist":
        ax.hist(sample[x].dropna())
    elif kind == "scatter":
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

    # save to artifacts/plots
    name = filename or f"{kind}_{x}{'_' + y if y else ''}.png"
    out = PLOTS_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
