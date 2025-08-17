from pathlib import Path

from agent.graph import run


def test_agent_smoke_average_price_by_product() -> None:
    csv = Path("tests/data/sales.csv")  # your existing sample
    res = run("What is the average price by product? show a small table.", csv)
    assert "final" in res
    assert isinstance(res["final"], str)
    assert res["steps"] <= 8
    # We should have at least one step and some observation text
    assert res["trace"]
    # The final answer should contain something table-like
    assert "table" in res["final"].lower() or "|" in res["final"] or "\n" in res["final"]
