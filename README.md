# DeciSense

**DeciSense** is a local-first agentic AI system designed to automate end-to-end tabular data science workflows — from raw dataset ingestion to model recommendation, training approval, experimentation, and artifact packaging.

The system is built around a simple idea:

> A user should be able to upload raw tabular data and receive meaningful analysis, recommendations, and optionally trained models with minimal manual intervention.

DeciSense emphasizes **analysis-first decision making**. It does not blindly train models immediately. Instead, it first analyzes the dataset, recommends a suitable first model with reasoning, and asks the user whether to continue to training.

---

## MVP 0 Status

MVP 0 is considered successful.

The current system can run locally end-to-end and can be operated from Telegram through OpenClaw. In this MVP, the Python engine does the actual data science work, while OpenClaw acts as the chat/orchestration layer that connects Telegram messages and uploaded files to the local DeciSense CLI.

What works today:

- Local dataset analysis through `ds_engine`
- Local model recommendation and training
- User approval before training
- Analysis-only package generation
- Full training package generation
- OpenClaw skill integration
- Telegram interaction through OpenClaw

Current responsibility split:

```text
Telegram / OpenClaw
  -> receive user messages and file uploads
  -> call the DeciSense CLI
  -> send CLI output back to the user

Python ds_engine
  -> load, validate, profile, recommend, train, evaluate, and package

LLM / model provider
  -> power the OpenClaw agent response and orchestration layer
  -> not yet the primary data-analysis or modeling decision maker
```

The next major direction is to make DeciSense more model-oriented: the LLM should interpret structured dataset summaries and propose modeling plans, while local Python code validates and executes those plans.

---

## Vision

DeciSense aims to simulate the workflow of a **senior data scientist**:

1. Understand the dataset
2. Validate whether it is suitable for downstream analysis
3. Diagnose data quality and target-related risks
4. Recommend an appropriate first modeling strategy
5. Ask the user whether to continue to model training
6. Train and evaluate models only after explicit approval
7. Package results into reproducible `.tar.gz` artifacts

The long-term goal is to connect this workflow to an agent orchestration layer such as OpenClaw, with Telegram as a user-facing interface.

---

## Core Workflow (MVP v0)

```text
User uploads tabular dataset
        ↓
DeciSense runs analysis phase
        ↓
System performs:
- File loading
- Tabular validation
- Task inference
- Schema profiling
- Data quality checks
- Target profiling
- Model recommendation
        ↓
System asks:
"Based on the analysis, the recommended first model is X.
Do you want to continue to training? Reply yes / no."
        ↓
User decision:
    ├─ no  → Create analysis-only package
    └─ yes → Run model training and create full training package
```

---

## System Architecture

DeciSense is composed of four main layers.

### 1. OpenClaw / Agent Orchestration Layer

This is the intended external orchestration layer.

In the current MVP, OpenClaw integration is handled through a **CLI wrapper + skill file**, not a custom OpenClaw plugin yet.

Responsibilities:

- Receive upload/text events from Telegram or another chat interface
- Call the DeciSense CLI
- Send returned messages back to the user
- Send generated packages when available

### 2. Python DS Engine

This is the deterministic execution layer.

Responsibilities:

- Load datasets
- Validate tabular structure
- Infer target and task type
- Profile schema, data quality, and target distribution
- Recommend model candidates
- Prepare modeling data
- Train and evaluate models
- Generate reports and packages

### 3. Workflow Layer

This layer coordinates higher-level flows.

Main workflows:

- Analysis workflow
- Training approval workflow
- Training workflow
- Session-aware service flow

### 4. Artifact & Session Layer

This layer handles persistence.

Responsibilities:

- Store run artifacts under `runs/<run_id>/`
- Store active chat sessions under `bot_state/sessions.json`
- Package analysis-only and full-training results into `.tar.gz`

---

## Project Structure

```text
DeciSense/
├─ README.md
├─ .gitignore
│
├─ bot_state/
│  └─ sessions.json
│
├─ runs/
│  └─ <run_id>/
│
├─ openclaw/
│  ├─ config/
│  └─ skills/
│     └─ decisense-tabular-ds/
│        └─ SKILL.md
│
├─ ds_engine/
│  ├─ interfaces/
│  │  └─ decisense_cli.py
│  │
│  ├─ intake/
│  │  ├─ load_data.py
│  │  ├─ validate_tabular.py
│  │  └─ infer_task.py
│  │
│  ├─ profiling/
│  │  ├─ schema_profile.py
│  │  ├─ data_quality.py
│  │  └─ target_profile.py
│  │
│  ├─ planning/
│  │  └─ model_recommender.py
│  │
│  ├─ modeling/
│  │  ├─ preprocess.py
│  │  └─ train_models.py
│  │
│  ├─ reporting/
│  │  ├─ analysis_report.py
│  │  ├─ telegram_messages.py
│  │  ├─ packager.py
│  │  └─ training_packager.py
│  │
│  ├─ workflows/
│  │  ├─ analysis_workflow.py
│  │  ├─ approval_state.py
│  │  ├─ approval_workflow.py
│  │  ├─ training_workflow.py
│  │  ├─ session_store.py
│  │  └─ decisense_service.py
│  │
│  └─ pipeline.py
│
└─ tests/
   └─ ds_engine/
      ├─ intake/
      ├─ profiling/
      ├─ planning/
      ├─ modeling/
      ├─ reporting/
      ├─ workflows/
      ├─ interfaces/
      └─ integration/
```

---

## Key Features

### MVP v0

- Tabular dataset loading
- CSV, TSV, Excel, and Parquet support
- Duplicate header detection for delimited files
- Tabular validation
- Target column inference
- Classification/regression task inference
- Schema profiling
- Data quality diagnostics
- Target profiling
- Model recommendation with reasoning
- Strict user approval before training
- Session-based chat workflow
- `reset` command for clearing active session state
- Analysis-only package generation
- Full training package generation
- Local CLI interface for OpenClaw/Telegram integration

---

## Supported Data Scope

DeciSense currently supports **tabular data only**.

Supported file types:

- `.csv`
- `.tsv`
- `.xlsx`
- `.xls`
- `.parquet`

Non-tabular data such as images, raw text corpora, audio, graph data, and time-series-specific workflows are outside the current MVP scope.

---

## Model Scope

### Classification

Currently supported local training models:

- Logistic Regression
- Random Forest Classifier
- HistGradientBoostingClassifier

### Regression

Currently supported local training models:

- Ridge Regression
- Random Forest Regressor
- HistGradientBoostingRegressor

### Optional / Future Model Families

DeciSense can recommend stronger tabular model families such as CatBoost when optional dependency behavior is enabled, but the current local training implementation focuses on scikit-learn-compatible models.

---

## Local Development Setup

DeciSense is designed to run locally first.

### Clone the repository

```bash
git clone <repo-url>
cd DeciSense
```

### Install dependencies with uv

Recommended:

```bash
uv sync --dev
```

Alternative, when adding dependencies during development:

```bash
uv add pandas scikit-learn openpyxl pyarrow joblib
uv add --dev pytest
```

### Run tests

```bash
uv run pytest
```

Expected result for MVP 0:

```text
145 passed
```

Run a specific test group:

```bash
uv run pytest tests/ds_engine/intake
uv run pytest tests/ds_engine/profiling
uv run pytest tests/ds_engine/planning
uv run pytest tests/ds_engine/modeling
uv run pytest tests/ds_engine/reporting
uv run pytest tests/ds_engine/workflows
uv run pytest tests/ds_engine/interfaces
uv run pytest tests/ds_engine/integration
```

### Code quality

DeciSense uses Ruff for linting and formatting.

Run lint checks:

```bash
uv run ruff check .
```

Apply safe lint fixes:

```bash
uv run ruff check . --fix
```

Check formatting:

```bash
uv run ruff format --check .
```

Format the codebase:

```bash
uv run ruff format .
```

Recommended local check before commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/smoke_test_local.py
```

### Continuous Integration

GitHub Actions runs:

```bash
uv sync --dev --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/smoke_test_local.py
```

The smoke test validates the local end-to-end CLI flow:

- upload dataset
- analysis recommendation
- no branch → analysis-only package
- yes branch → training + full training package

---

## Quickstart Checklist

Use this checklist after cloning the repository.

### 1. Verify the Python engine

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/smoke_test_local.py
```

### 2. Run a local CLI analysis

```bash
uv run python -m ds_engine.interfaces.decisense_cli analyze-upload \
  --chat-id "local_chat_1" \
  --file-path "sample_data/demo_churn.csv" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

If you do not have `sample_data/demo_churn.csv`, use any local CSV with a target-like column such as `target`, `label`, `class`, or `outcome`.

### 3. Approve or reject training

Approve training:

```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "local_chat_1" \
  --message-text "yes" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

Stop before training:

```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "local_chat_1" \
  --message-text "no" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

Reset the local chat session:

```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "local_chat_1" \
  --message-text "reset" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

---

## Local CLI Usage

DeciSense exposes a local CLI entry point that can be used by OpenClaw, Telegram adapters, or manual local testing.

### Start analysis for an uploaded dataset

```bash
uv run python -m ds_engine.interfaces.decisense_cli analyze-upload \
  --chat-id "local_chat_1" \
  --file-path "path/to/dataset.csv" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

This will:

- copy the uploaded dataset into `runs/<run_id>/raw/`
- run the analysis workflow
- validate the dataset
- infer the target column when possible
- profile schema, data quality, and target distribution
- recommend an initial model
- return a JSON payload containing a chat-ready message

### Handle a text reply

```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "local_chat_1" \
  --message-text "yes" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

Supported text messages:

| Message | Behavior |
|---|---|
| `yes` | Continue to model training and create a full training package |
| `no` | Stop before training and create an analysis-only package |
| `reset` | Clear the active session for the chat |
| `target:<column_name>` | Manually provide the target column when inference is unresolved |

Only `yes` and `no` are accepted for training approval.

Localized replies such as `ya`, `tidak`, `lanjut`, or `batal` are intentionally treated as invalid in MVP v0 to keep the approval flow strict and unambiguous.

---

## CLI Output

The CLI returns JSON.

Compact output includes:

```json
{
  "chat_id": "local_chat_1",
  "status": "analysis_started",
  "run_id": "run_20260508_abcdef",
  "session_state": "waiting_training_approval",
  "message_type": "training_approval",
  "text": "The dataset analysis is complete...",
  "expects_reply": true,
  "reply_hint": "yes / no",
  "package_path": null,
  "message_metadata": {},
  "errors": []
}
```

Important fields:

| Field | Meaning |
|---|---|
| `text` | Message to send back to the user |
| `message_type` | Type of response generated |
| `expects_reply` | Whether the system expects another user reply |
| `reply_hint` | Expected reply format |
| `package_path` | Path to generated `.tar.gz` package, if available |
| `session_state` | Persisted session state |
| `status` | Service-level status |

Use `--full-json` to print the full nested workflow result.

---

## Session Behavior

DeciSense is session-based.

Each chat ID can have one active DeciSense session at a time.

If a user uploads a new dataset while a session is still active, DeciSense rejects the new upload and asks the user to either continue the current flow or send `reset`.

### Reset command

The reset command is:

```text
reset
```

It must be an exact text match after trimming whitespace.

Accepted:

```text
reset
RESET
 reset 
```

Not accepted:

```text
/reset
please reset
reset now
tolong reset
```

The slash command `/reset` is intentionally not used because some chat runtimes reserve slash-prefixed commands for system-level command handling.

`reset` clears only the active session state. It does not delete artifacts under `runs/`.

---

## OpenClaw Integration

DeciSense is designed to be called by OpenClaw through a thin CLI wrapper.

Current integration approach:

```text
OpenClaw / Telegram event
        ↓
DeciSense CLI
        ↓
DeciSenseService
        ↓
Analysis / Approval / Training workflows
        ↓
JSON response + package path
```

The current OpenClaw-facing skill lives at:

```text
openclaw/skills/decisense-tabular-ds/SKILL.md
```

The skill teaches the agent to call the DeciSense CLI for two event types.

### Dataset upload

```bash
uv run python -m ds_engine.interfaces.decisense_cli analyze-upload \
  --chat-id "<chat_id>" \
  --file-path "<uploaded_file_path>" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

### Text message

```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "<chat_id>" \
  --message-text "<message_text>" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

If `package_path` is present in the returned JSON, the adapter should send the generated `.tar.gz` file back to the user when the channel supports file attachments.

---

## OpenClaw + Telegram Setup

Follow these steps when running DeciSense through Telegram.

### 1. Install OpenClaw

Install OpenClaw using the official OpenClaw installation flow for your platform.

Verify:

```bash
openclaw --version
openclaw status
```

Keep OpenClaw updated:

```bash
openclaw update --yes --timeout 1800
```

For WSL/Ubuntu users, if OpenClaw update or build fails with `libatomic.so.1`, install:

```bash
sudo apt-get update
sudo apt-get install -y libatomic1
```

### 2. Configure the OpenAI model token

For OpenClaw, prefer OpenClaw auth profiles over a project `.env` file.

```bash
openclaw models auth paste-token --provider openai --profile-id openai:default
```

Paste your OpenAI API key when prompted.

Then verify raw model inference:

```bash
openclaw infer model run \
  --model openai/gpt-5.4-mini \
  --prompt "reply with only OK" \
  --json
```

Expected result:

```json
{
  "ok": true,
  "provider": "openai",
  "model": "gpt-5.4-mini",
  "outputs": [
    {
      "text": "OK"
    }
  ]
}
```

Set the validated model as the default:

```bash
openclaw models set openai/gpt-5.4-mini
openclaw models status
```

### 3. Configure Telegram

Create a Telegram bot with BotFather and copy the bot token.

Add the Telegram token to OpenClaw:

```bash
openclaw channels add --channel telegram --token "<telegram_bot_token>"
```

Verify the channel:

```bash
openclaw channels status --probe
openclaw status --all
```

Expected status:

```text
Telegram default: enabled, configured, running, connected
```

### 4. Make the DeciSense skill available to OpenClaw

The DeciSense skill file is:

```text
openclaw/skills/decisense-tabular-ds/SKILL.md
```

Make sure OpenClaw can discover this skill from its active workspace or managed skills directory.

Verify:

```bash
openclaw skills list
```

Expected:

```text
decisense-tabular-ds ... ready
```

### 5. Verify the full agent runtime

Raw model inference is not enough. Telegram uses the OpenClaw agent runtime, so verify that path too:

```bash
openclaw agent \
  --local \
  --agent main \
  --message "reply with only OK" \
  --model openai/gpt-5.4-mini \
  --json \
  --timeout 120
```

Expected result:

```json
{
  "payloads": [
    {
      "text": "OK"
    }
  ],
  "meta": {
    "aborted": false
  }
}
```

### 6. Test Telegram

Send this message to the Telegram bot:

```text
reply with only OK
```

Expected reply:

```text
OK
```

Then upload a supported tabular dataset and ask DeciSense to analyze it.

### 7. Reset stale sessions when needed

If Telegram still behaves like an older session, send:

```text
reset
```

Then start the interaction again.

---

## Secrets and Tokens

Do not commit tokens or secrets.

The repository ignores:

```text
.env
.env.*
*.env
secrets/
*.pem
*.key
*.crt
```

Recommended token handling:

| Secret | Recommended location |
|---|---|
| OpenAI API key for OpenClaw | OpenClaw auth profile via `openclaw models auth paste-token` |
| Telegram bot token | OpenClaw channel config or OpenClaw secret management |
| Local app experiments | Local `.env`, never committed |

Why not rely only on terminal `export OPENAI_API_KEY=...`?

OpenClaw may run as a system service, so environment variables exported in one terminal may not be visible to the gateway. OpenClaw auth profiles are more reliable for this setup.

---

## OpenClaw Runtime Notes

### Bootstrap file

If OpenClaw keeps injecting first-run onboarding instructions, check:

```text
~/.openclaw/workspace/BOOTSTRAP.md
```

After OpenClaw setup is complete, remove or rename that file. A stale bootstrap file can make a simple prompt slow or unexpected because it forces OpenClaw to run onboarding instructions before answering normally.

### Slow Telegram replies

Slow replies usually come from OpenClaw agent context size, not Telegram itself.

The agent runtime may load:

- workspace instructions
- skill descriptions
- tool descriptions
- memory
- session history

To improve latency:

- create a dedicated lightweight DeciSense OpenClaw agent
- keep only the DeciSense skill active for that agent
- reduce active tools where possible
- use lower thinking settings where supported
- reset long-running Telegram sessions
- keep the OpenClaw workspace minimal

---

## Artifact Outputs

DeciSense writes run artifacts under:

```text
runs/<run_id>/
```

Typical structure:

```text
runs/<run_id>/
├─ raw/
│  └─ uploaded dataset
│
├─ analysis/
│  ├─ analysis_report.json
│  ├─ analysis_report.md
│  ├─ pipeline_result.json
│  └─ package_manifest.json
│
├─ training/
│  ├─ prepared_modeling_dataset.json
│  ├─ model_training_result.json
│  ├─ training_summary.txt
│  └─ training_manifest.json
│
├─ package/
│  └─ full_training/
│     ├─ analysis/
│     ├─ training/
│     ├─ source_data/
│     └─ package_manifest.json
│
└─ bundles/
   ├─ <run_id>_analysis_only.tar.gz
   └─ <run_id>_full_training.tar.gz
```

By default, generated packages do not include the raw source dataset unless explicitly configured.

---

## Output Packages

Depending on user decision, DeciSense can generate two package types.

### Analysis-only package

Created when the user replies:

```text
no
```

Contains:

- analysis report JSON
- analysis report Markdown
- pipeline result JSON
- package manifest

### Full training package

Created when the user replies:

```text
yes
```

Contains:

- analysis report
- pipeline result
- analysis workflow result
- prepared modeling dataset metadata
- model training result
- training workflow result
- training summary
- fitted model pipelines when available
- package manifest

---

## Design Principles

### Analysis before execution

DeciSense analyzes the dataset first and recommends a model before running training.

### User-in-the-loop approval

Training only runs after the user explicitly replies:

```text
yes
```

If the user replies:

```text
no
```

DeciSense stops before training and returns an analysis-only package.

### Deterministic core, agentic interface

The Python engine performs the actual data science operations.

The agent layer should orchestrate the workflow, communicate with the user, and pass events to the service layer.

### Local-first execution

MVP v0 is designed to run locally to avoid unnecessary infrastructure cost, especially for training workloads.

### Transparent recommendations

Model recommendations include reasoning, concerns, and initial parameters.

---

## Current Limitations

- Tabular data only
- Session state is JSON-backed and intended for local MVP usage
- Telegram/OpenClaw integration currently uses a CLI wrapper, not a custom OpenClaw plugin
- Training currently focuses on local scikit-learn-compatible models
- Model recommendation is currently rule-based
- The LLM does not yet perform the main data science reasoning
- CatBoost can be recommended when configured, but local training depends on future optional model support
- No advanced hyperparameter optimization yet
- No automatic feature engineering beyond basic preprocessing
- No SHAP or advanced interpretability layer yet
- No deployment/export of production-ready model services yet
- User-provided modeling context at upload time is not supported yet
- The approval flow accepts only strict `yes` / `no`

---

## Future Directions

The next major direction is a more model-oriented DeciSense.

Target architecture:

```text
Local Python engine computes dataset facts
        ↓
LLM reads structured dataset summary
        ↓
LLM proposes target, metric, model, and training plan
        ↓
Local validator checks the plan
        ↓
User approves
        ↓
Local Python engine executes training
```

The intended split:

- The LLM thinks, explains, prioritizes, and recommends.
- The local Python engine validates, trains, evaluates, and packages.

Expected next updates:

- LLM-assisted dataset interpretation
- LLM-assisted target column recommendation
- LLM-assisted metric selection
- Structured JSON training plans generated by the model
- Local validation before executing any model-generated plan
- Hybrid comparison between rule-based and LLM-based recommendations
- Upload-time user context such as business goal, target column, metric preference, and constraints
- Dedicated lightweight DeciSense OpenClaw agent for faster Telegram replies
- Smaller active skill/tool context for DeciSense chat
- Advanced hyperparameter optimization
- Feature engineering pipelines
- Model interpretability reports
- Experiment tracking dashboard
- Custom OpenClaw plugin
- API or web interface
- More robust artifact lifecycle management
- Expanded model family support

---

## Status

MVP 0 complete and locally validated.

This project is being built as a personal AI/Data Science portfolio project.

---

## Contributing

Currently not open for contributions.

This may change as the project stabilizes.

---

## License

TBD
