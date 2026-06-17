# 🤖 Data Analyst AI Agent

> An autonomous AI agent that performs end-to-end data analysis through natural language — from raw data ingestion to insight generation, visualization, and reporting.

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)

---

## 📌 Overview

This project was built as part of a **30-day AI agent challenge** to explore how autonomous agents can replace repetitive data analyst workflows. The agent accepts natural language questions about a dataset and returns:

- **Automated EDA** — summary statistics, distributions, correlations
- **Dynamic visualizations** — charts generated on-demand based on query context
- **Insight narratives** — plain-English interpretation of findings
- **Exportable reports** — saved artifacts in the `artifacts/` directory

---

## 🏗️ Architecture

```
User Query (natural language)
        ↓
   Agent Orchestrator (LLM)
        ↓
  ┌──────────────────────────────────┐
  │  Tool Router                   │
  │  ├── data_loader_tool           │
  │  ├── eda_tool                   │
  │  ├── visualization_tool         │
  │  └── summary_writer_tool        │
  └──────────────────────────────────┘
        ↓
   Artifact Output (charts, reports)
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/navichearala/data-analyst-aiagent.git
cd data-analyst-aiagent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the agent
python agent/main.py
```

---

## 📁 Project Structure

```
data-analyst-aiagent/
├── agent/               # Core agent logic & tool definitions
├── artifacts/           # Generated charts and analysis outputs
├── tests/               # Unit tests (pytest)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project config
├── pytest.ini           # Test config
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 💬 Key Learnings

- Designing multi-tool LLM agent architectures with clear tool boundaries
- Prompt engineering for reliable tool routing and structured outputs
- Challenges of non-determinism in agent decision-making
- Testing strategies for AI-driven systems

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

*Part of my Data Science portfolio — [View more projects](https://github.com/navichearala)*
