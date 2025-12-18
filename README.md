# éclat — EY Techathon 6.0 Prototype

**Project Overview**
- **Purpose:** éclat is a prototype system for responsibly automating customer engagement and underwriting decisions with strong auditability, conservative financial provisioning, and human-review guardrails.
- **Scope:** research + prototype for EY Techathon 6.0 demonstrating theory-grounded decision automation that maps model outputs to underwriting and provisioning artifacts.

**Problem Statement II (EY Techathon 6.0)**
- **Chosen challenge:** Build an auditable, safety-first automated decisioning pipeline that recommends a discrete action (e.g., `pursue`, `escalate`, `defer`) for inbound leads while computing conservative provisioning measures (PD / EL) and persisting decision artifacts for regulatory and internal audit.
- **Key constraints:** explainability (deterministic surrogate arithmetic), conservative provisioning (upper percentile PD estimates), fairness checks and subgroup monitoring, and human-review for gray-band decisions.

**Prototype Phase**
- Rapid, reproducible experimentation using Jupyter notebooks and small prototype scripts.
- Modeling approach: lightweight surrogates + bootstrap/Bayesian heuristics for uncertainty; deterministic artifact generation for audit trails; optional SHAP-based local explanations (guarded behind an install check).
- Artifacts: plots and JSON outputs saved under `artifacts/` for reproducibility and audit.

**Repository Layout**
- `notebooks/` : core notebooks used to document theory, experiments, and runnable demos.
- `html/` : generated HTML exports of the notebooks for sharing and review.
- `flowcharts/` : system-level diagrams and process flow artifacts.
- `prototype/` and `scripts/` : helper scripts and data-generation utilities used during prototyping.
- `app/` : Streamlit UI for interactive exploration and demoing model decisions (a minimal demo is provided at `app/app.py`).
- `artifacts/` : model outputs, images, and generated JSON artifacts used for audits and reporting.

**UI Setup (Dummy Values) — Quick Demo**
- A minimal Streamlit UI is provided at `app/app.py` to let reviewers interact with dummy feature inputs and observe a computed PD and portfolio EL.

Quick Start (Windows)

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
# if you only want to run the UI demo, ensure streamlit is installed:
pip install streamlit
```

3. Run the Streamlit demo

```powershell
streamlit run app/app.py
```

Notes on the demo
- The demo uses simple logistic scoring and exposes PD, conservative PD (percentile), and a sample Expected Loss calculation so reviewers can observe how decisions map to financial metrics.
- To reproduce notebook-based outputs, open the notebooks in `notebooks/` or view their HTML exports in `html/`.