from pathlib import Path
from agent.graph import run

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="tests/data/sales.csv")
    p.add_argument("question", nargs="+")
    args = p.parse_args()
    res = run(" ".join(args.question), args.csv, max_steps=8)
    print("FINAL:\n", res["final"])
    print("\nSTEPS:", res["steps"])
    print("\nTRACE:")
    for i, step in enumerate(res["trace"], 1):
        print(f"{i:02d}. {step['thought']}\n   -> {step['observation']}\n")

