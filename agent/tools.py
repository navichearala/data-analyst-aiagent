from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Safety caps
MAX_CELLS_DEFAULT = 2_000_000  # ~2M cells (rows * cols)
MAX_PREVIEW_ROWS = 50
ARTIFACTS_DIR = Path("artifacts")


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(
    path_or_url: str | Path,
    *,
    max_cells: int = MAX_CELLS_DEFAULT,
    dtype_backend: str = "numpy_nullable",
) -> pd.DataFrame:
    """
    Read a CSV (local path or URL) into a DataFrame, then enforce a cell cap.
    Raises ValueError if df.size exceeds `max_cells`.
    """
    df = pd.read_csv(
        path_or_url,
        low_memory=False,
        dtype_backend=dtype_backend,  # pandas>=2.0
    )
    cells = int(df.size)
    if cells > max_cells:
        raise ValueError(
            f"CSV too large for safety: {cells:,} cells > cap {max_cells:,} "
            "(reduce file or sample first)."
        )
    return df


def schema_infer(df: pd.DataFrame, *, sample_rows: int = 1_000) -> dict[str, Any]:
    """
    Return a simple schema summary:
      - columns: list[str]
      - dtypes: dict[col->dtype str]
      - nulls: dict[col->null count]
    Uses head(sample_rows) to keep it fast.
    """
    sample = df.head(sample_rows)
    return {
        "columns": list(sample.columns),
        "dtypes": {c: str(sample[c].dtype) for c in sample.columns},
        "nulls": {c: int(sample[c].isna().sum()) for c in sample.columns},
    }


def preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the first n rows, capped at MAX_PREVIEW_ROWS to protect privacy & speed.
    """
    n = max(0, min(n, MAX_PREVIEW_ROWS))
    return df.head(n if n > 0 else 5).head(MAX_PREVIEW_ROWS)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a small/medium table to CSV under artifacts/ (if no folder provided).
    Returns the final path.
    """
    _ensure_artifacts_dir()
    out = Path(path)
    if out.parent == Path("."):
        out = ARTIFACTS_DIR / out.name
    df.to_csv(out, index=False)
    return out


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column summary with dtype, non_null, nulls and basic stats.
    Numeric: mean, std, min, max
    Non-numeric: unique, top (most frequent)
    """
    rows = []
    for col in df.columns:
        s = df[col]
        item: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
        }
        if is_numeric_dtype(s):
            item.update(
                {
                    "mean": float(s.mean()) if s.notna().any() else None,
                    "std": float(s.std()) if s.notna().any() else None,
                    "min": float(s.min()) if s.notna().any() else None,
                    "max": float(s.max()) if s.notna().any() else None,
                }
            )
        else:
            item.update(
                {
                    "unique": int(s.nunique(dropna=True)),
                    "top": (s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None),
                }
            )
        rows.append(item)
    return pd.DataFrame(rows).set_index("column")
