## Design (Day 3)

### Problem
Give data analysts a helper that can read a CSV, answer plain-English questions, run basic analysis, and produce tables/plots—without writing code by hand.

### Architecture (high level)
1. **User** asks a question (“Show top 5 products by revenue in 2024”).
2. **Agent loop** (LLM):
   - Understand the task → plan steps.
   - Choose a **tool** (CSV load, filter, groupby, plot, model).
   - Run the tool → read results.
   - Repeat until done, then summarize and show results.
3. **Runtime**: Python + Pandas/Matplotlib (tools), OpenAI API (LLM).

### Data Flow
`CSV file → load_csv → dataframe → filter/group/aggregate → (optional) model/train → visualize → answer + artifacts (png/csv)`

### Tools (initial)
- `load_csv(path_or_url)` – read small/medium CSV.
- `preview(n)` – show head/tail sample.
- `describe()` – numeric summary.
- `filter(expr)` – boolean filter.
- `groupby_agg(cols, metrics)` – aggregations.
- `plot(kind, x, y, hue?)` – save chart.
- `train_model(kind, target, features)` – simple baseline.
- `evaluate_model(metric)` – accuracy/RMSE.

### Safety & Limits
- **Privacy**: never print full raw data or PII; cap previews to 50 rows.
- **Cost/latency**: small answers first; avoid unnecessary tool calls.
- **Refuse**: illegal, dangerous, sensitive PII extraction, or requests to access files outside project.
- **Determinism**: prefer deterministic ops; state assumptions.
