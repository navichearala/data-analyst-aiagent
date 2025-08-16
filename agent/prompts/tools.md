# Tools & Rules

The agent may only use the tools below. Each call must specify **inputs** and expect the **outputs** described.

---

## load_csv(path_or_url)
**Use when** starting an analysis.  
**Inputs:** string path or URL  
**Returns:** dataframe handle `df` in memory  
**Rules:** Fail fast if file > ~100MB or columns are ambiguous.

**Example plan:**  
- load_csv("examples/sales_2024.csv")  
- preview(5)

---

## preview(n=5)
**Use when** you need to confirm columns/types.  
**Inputs:** integer n (≤ 50)  
**Returns:** small table

---

## describe()
**Use when** you need numeric summary (mean, std, min, max, quartiles).  
**Returns:** small table of stats.

---

## filter(expr)
**Use when** subsetting data.  
**Inputs:** boolean expression using column names, e.g.  
`"year == 2024 and region in ['NA','EU'] and revenue > 0"`  
**Returns:** filtered dataframe; mention row count.

**Rule:** Validate columns exist before filtering.

---

## groupby_agg(cols, metrics)
**Use when** computing aggregations.  
**Inputs:**  
- `cols`: list of columns to group by  
- `metrics`: dict like `{"revenue":"sum", "qty":"mean"}`  
**Returns:** aggregated table (≤ 100 rows)

---

## plot(kind, x, y, hue=None, title=None)
**Use when** a chart clarifies the answer.  
**Kinds:** "line", "bar", "hist", "scatter"  
**Returns:** saves `artifacts/<auto-name>.png` and returns the path.  
**Rule:** Do not plot more than ~200 points for readability.

---

## train_model(kind, target, features)
**Use when** user asks for a simple prediction or classification.  
**Kinds:** "linear_regression", "logistic_regression"  
**Returns:** fitted model handle + summary (R²/accuracy).  
**Rule:** Split 80/20; set random_state=42; scale features if needed.

---

## evaluate_model(metric)
**Use when** validating a fitted model.  
**Metrics:** "rmse", "mae", "accuracy", "f1".  
**Returns:** metric value and 1-line interpretation.

---

## save_report(markdown, filename)
**Use when** you want a lightweight report.  
**Returns:** path `artifacts/<filename>.md`.

---

## General Safeguards
- Cap previews to **50 rows** and aggregated tables to **100 rows**.
- Never print raw PII; anonymize if present.
- If a requested operation risks heavy memory/compute, warn and ask to sample.
- Always state assumptions (e.g., “interpreting `date` as YYYY-MM-DD”).
