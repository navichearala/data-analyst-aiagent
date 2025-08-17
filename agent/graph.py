# agent/graph.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

# import your tools
from agent.tools import groupby_agg, load_csv  # add plot later if you want


# ---------- Types ----------
class Action(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    final_answer: str  # when present, it's the end


@dataclass
class StepRecord:
    thought: str
    action: Action | None
    observation: str


@dataclass
class AgentState:
    df: pd.DataFrame | None = None
    last_table: pd.DataFrame | None = None
    trace: list[StepRecord] = field(default_factory=list)  # filled at runtime


# ---------- Tool adapters (execute + tiny observation) ----------
def _obs_from_df(df: pd.DataFrame, max_cols: int = 6, max_rows: int = 5) -> str:
    cols = list(df.columns)[:max_cols]
    return f"DataFrame rows={len(df)} cols={len(df.columns)} preview=\n{df[cols].head(max_rows).to_string(index=False)}"


def _tool_load_csv(args: dict[str, Any], st: AgentState) -> str:
    path = args.get("path")
    if not path:
        raise ValueError("load_csv requires 'path'")
    df = load_csv(path)
    st.df = df
    return _obs_from_df(df)


def _tool_groupby_agg(args: dict[str, Any], st: AgentState) -> str:
    if st.df is None:
        raise ValueError("No DataFrame loaded yet; call load_csv first.")
    by = args.get("by")
    metrics = args.get("metrics")
    if not isinstance(by, list) or not isinstance(metrics, dict):
        raise ValueError("groupby_agg requires 'by': list[str] and 'metrics': dict[str,str]")
    out = groupby_agg(st.df, by=by, metrics=metrics)
    st.last_table = out
    return _obs_from_df(out)


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any], AgentState], str]] = {
    "load_csv": _tool_load_csv,
    "groupby_agg": _tool_groupby_agg,
    # "plot": _tool_plot,  # add later if desired
}


# ---------- Planner (Reason) ----------
def _rule_based_planner(query: str, st: AgentState, csv_path: Path) -> Action:
    q = query.lower()

    # 1) Always load first if needed
    if st.df is None:
        return Action(tool="load_csv", args={"path": str(csv_path)})

    # 2) If we already produced a table in the last step, finalize now
    if st.last_table is not None:
        tbl = st.last_table.head(5).to_string(index=False)
        return Action(final_answer=f"Here is the table you asked for (showing up to 5 rows):\n{tbl}")

    # 3) Otherwise, decide the next tool
    if ("average" in q or "mean" in q) and ("price" in q and "product" in q):
        return Action(tool="groupby_agg", args={"by": ["product"], "metrics": {"price": "mean"}})

    # 4) Fallback final
    return Action(final_answer="Sorry, this minimal Day-7 planner currently supports only 'average price by product'.")


# ---------- Main loop ----------
def run(query: str, csv_path: str | Path, max_steps: int = 8) -> dict[str, Any]:
    """
    Minimal Reason -> Act loop with a hard step cap.
    Returns dict with 'final' answer text and a 'trace' of steps.
    """
    st = AgentState(df=None, last_table=None, trace=[])
    csv_path = Path(csv_path)

    for step_idx in range(max_steps):
        # 1) Reason
        action = _rule_based_planner(query, st, csv_path)

        if "final_answer" in action:
            # record and return
            st.trace.append(StepRecord(thought="finish", action=action, observation="done"))
            return {
                "final": action["final_answer"],
                "trace": [sr.__dict__ for sr in st.trace],
                "steps": step_idx + 1,
            }

        # 2) Act (safe: whitelist only)
        tool_name = action.get("tool", "")
        tool_fn = TOOL_REGISTRY.get(tool_name)
        thought = f"Use tool {tool_name} with args={action.get('args', {})}"
        if tool_fn is None:
            st.trace.append(StepRecord(thought=thought, action=action, observation=f"Unknown tool: {tool_name}"))
            # graceful degrade into final
            return {
                "final": f"Planner asked for unknown tool '{tool_name}'.",
                "trace": [sr.__dict__ for sr in st.trace],
                "steps": step_idx + 1,
            }

        try:
            obs = tool_fn(action.get("args", {}), st)
        except Exception as err:  # noqa: BLE001  (acceptable for agent loop)
            obs = f"Tool error: {err!s}"

        # 3) Observe
        st.trace.append(StepRecord(thought=thought, action=action, observation=obs))

    # Hard cap reached
    return {
        "final": "Stopped: step cap reached before final_answer.",
        "trace": [sr.__dict__ for sr in st.trace],
        "steps": max_steps,
    }
