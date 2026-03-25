# DeciSense

**DeciSense** is an agentic AI system designed to automate end-to-end tabular data science workflows — from raw dataset ingestion to model recommendation and experimentation — using a structured, decision-driven approach.

The system is built around a simple idea:

> _A user should be able to upload raw data and receive meaningful insights, recommendations, and (optionally) trained models — without manual intervention._

---

## 🚀 Vision

DeciSense aims to simulate the workflow of a **senior data scientist**:

1. Understand the dataset
2. Diagnose data quality and risks
3. Recommend appropriate modeling strategies
4. Execute experiments (only when approved)
5. Deliver interpretable, actionable results

Rather than blindly training models, DeciSense emphasizes **analysis-first decision making**.

---

## ⚙️ Core Workflow (MVP v0)

```

User → Upload Dataset (Telegram)
↓
[Analysis Phase]
↓
Agent performs:

* Data validation (tabular check)
* Dataset profiling (missing values, imbalance, etc.)
* Task inference (classification/regression)
* Risk detection (leakage, high cardinality, etc.)
  ↓
  Agent recommends:
* Best initial model
* Initial hyperparameters
* Reasoning behind selection
  ↓
  User decision:
  ├─ "no"  → Return analysis-only bundle
  └─ "yes" → Proceed to training
  ↓
  Model training + evaluation
  ↓
  Final results + artifacts

```

---

## 🧠 System Architecture

DeciSense is composed of three main layers:

### 1. OpenClaw (Agent Orchestration Layer)
- Handles multi-agent reasoning and workflow control
- Manages Telegram communication
- Maintains session state and approval flow

### 2. Python DS Engine (Execution Layer)
- Performs all data science operations:
  - Data loading & validation
  - Profiling & EDA
  - Model selection & training
  - Evaluation & overfitting checks
- Produces structured outputs for agent interpretation

### 3. Artifact & State Layer
- Stores experiment runs
- Packages results into `.tar.gz`
- Tracks user sessions and decisions

---

## 📁 Project Structure

```

project/
├─ openclaw/
│  ├─ config/
│  └─ skills/
│
├─ ds_engine/
│  ├─ intake/
│  ├─ profiling/
│  ├─ planning/
│  ├─ modeling/
│  ├─ evaluation/
│  ├─ reporting/
│  └─ pipeline.py
│
├─ runs/
│  └─ <run_id>/
│
├─ bot_state/
│
├─ requirements.txt
└─ README.md

```

---

## 🔍 Key Features (MVP Scope)

- Tabular dataset validation
- Automated dataset profiling
- Task type inference (classification / regression)
- Model recommendation with reasoning
- Conditional execution (user approval before training)
- Basic experiment tracking
- Artifact packaging (`.tar.gz`)
- Telegram-based interaction

---

## 🧪 Model Scope (Initial)

**Classification**
- Logistic Regression (baseline)
- Random Forest
- CatBoost / LightGBM (if available)

**Regression**
- Linear / Ridge Regression
- Random Forest
- CatBoost / LightGBM

---

## 📦 Output

Depending on user decision:

### Analysis Only
- Dataset summary
- Data quality report
- Risk flags (e.g., leakage, imbalance)
- Model recommendation
- Reasoning and assumptions

### Full Pipeline
- All analysis artifacts
- Trained models
- Evaluation metrics
- Experiment logs
- Final summary

All outputs are packaged as a downloadable `.tar.gz`.

---

## 🔐 Design Principles

- **Analysis before execution**  
  Avoid unnecessary computation unless justified.

- **Transparency over automation**  
  Always explain _why_ a model is recommended.

- **Deterministic core, agentic interface**  
  Python handles execution; agents handle reasoning.

- **User-in-the-loop decisions**  
  Critical steps require explicit approval.

---

## ⚠️ Current Limitations

- Tabular data only
- No automatic feature engineering (beyond basic preprocessing)
- Limited hyperparameter tuning
- Target column inference may require user confirmation
- No deployment/export of production-ready models yet

---

## 🛠️ Future Directions

- Multi-agent specialization (Analyst, Scientist, Reviewer)
- Advanced hyperparameter optimization
- Feature engineering pipelines
- Model interpretability (e.g., SHAP)
- Experiment tracking dashboard
- API + web interface (beyond Telegram)

---

## 📌 Status

🚧 Early-stage (MVP in development)

This project is actively being built as part of a personal AI/Data Science portfolio.

---

## 🤝 Contributing

Currently not open for contributions.  
This may change as the project stabilizes.

---

## 📜 License

TBD
