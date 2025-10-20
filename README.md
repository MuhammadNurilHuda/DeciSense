# [Ongoing...]🧠 DeciSense

An intelligent, config-driven data science framework that automates EDA, feature engineering, model training, and reporting. It uses built-in logic and optional LLM reasoning to make smart, bias-aware decisions across the entire ML pipeline.

---

## 🚧 Project Status
**Current Stage:** Early Development
**Goal:** Build the foundation for a modular, reproducible, and adaptive ML automation framework.

---

## 🧩 Core Features (Planned)
- Automated **EDA** (missingness, outliers, imbalance, leakage hints)
- Config-driven **preprocessing** & **feature engineering**
- Smart **model selection & evaluation**
- **LLM-assisted decisions** for imputations, encoders, and bias detection
- Auto-generated **reports** (HTML/JSON) with metrics, plots, and insights
- Reproducible **pipelines** and consistent test-time transformations

---

## 🛠️ Implementation Plan

### Phase 1 – MVP
1. Set up repository structure  
```

src/
core/
preprocess/
features/
models/
eval/
report/
llm_agent/
configs/
data/
artifacts/
reports/
tests/

````
2. Implement CLI skeleton:  
- `deci run --config config.yaml`  
- `deci eda`, `deci train`, `deci report`
3. Build data loader, schema inference, and stratified split.  
4. Add EDA module: missingness, outliers, imbalance, leakage hints.  
5. Basic model zoo: Logistic Regression, Random Forest, LightGBM.  
6. CV + metrics (ROC-AUC, F1, RMSE); save artifacts & metrics.json.  
7. Generate HTML report summarizing pipeline & performance.

### Phase 2 – Smart Engine
1. Implement **LLMDecisionAgent** using open LLMs (Phi-3, Mistral-7B).  
2. Generate dynamic preprocessing configs (imputer, encoder, scaler).  
3. Integrate reasoning logs (`report/llm_decision.log`).  
4. Extend feature engineering: datetime, interactions, text (TF-IDF).  
5. Add explainability (Permutation Importance, optional SHAP).  

### Phase 3 – Reporting & Stability
1. Segment error analysis (by region, channel, etc.).  
2. Add drift/stability checks (PSI, cross-fold variance).  
3. Enhance HTML report with interactive visualizations.  
4. Package as installable CLI (`pip install decisense`).  

### Phase 4 – Community & Extensions
1. Add plugin system for custom transformers & models.  
2. Integrate experiment tracking (MLflow optional).  
3. Add Dockerfile + example notebooks.  
4. Release v0.2 Beta on PyPI.

---

## ⚡ Quick Start (Planned)
```bash
pip install decisense
deci run --config configs/exp.yaml
````

---

## 🧭 Roadmap Summary

| Phase | Focus        | Key Deliverable            |
| ----- | ------------ | -------------------------- |
| 1     | MVP          | EDA, model zoo, report     |
| 2     | Smart Engine | LLM-based decision-making  |
| 3     | Reporting    | HTML + stability           |
| 4     | Community    | Plugins, tracking, release |

---

## 📜 License

MIT License (to be finalized)

---

## 🤝 Contributing

Contributions are welcome!
Please open an issue or pull request for ideas, bugs, or feature suggestions.

---

## 🌟 Vision

> *“Let your data think for itself.”*
> DeciSense turns data pipelines into decision pipelines — bridging automation, reasoning, and insight.
