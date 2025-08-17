import uuid
from pathlib import Path

import pandas as pd

from agent.tools import PLOTS_DIR, plot


def _fresh_name(kind: str) -> str:
    # unique filename so repeated runs don't clash
    return f"{kind}_{uuid.uuid4().hex[:8]}.png"


def test_plot_creates_pngs() -> None:
    # tiny demo dataset
    df = pd.DataFrame(
        {
            "x": list(range(10)),
            "y": [i * i for i in range(10)],
            "cat": ["A"] * 5 + ["B"] * 5,
        }
    )

    # line, bar, scatter need y; hist only needs x
    kinds = ["line", "bar", "scatter", "hist"]
    outs: list[Path] = []

    for k in kinds:
        if k == "hist":
            out = plot(df, kind=k, x="x", y=None, title="hist demo", filename=_fresh_name(k))
        else:
            out = plot(df, kind=k, x="x", y="y", hue="cat", title=f"{k} demo", filename=_fresh_name(k))

        outs.append(out)
        # saved in artifacts/plots/
        assert out.parent == PLOTS_DIR
        assert out.suffix == ".png"
        assert out.exists()
        assert out.stat().st_size > 0  # not an empty file

    # (optional) cleanup after asserting
    for p in outs:
        try:
            p.unlink()
        except Exception:
            pass
