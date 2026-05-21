from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(".smoke/local_e2e").resolve()

    if root.exists():
        shutil.rmtree(root)

    sample_dir = root / "sample_data"
    runs_root = root / "runs"
    session_store = root / "bot_state" / "sessions.json"

    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = sample_dir / "demo_churn.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "age,income,tenure_months,target",
                "25,5000,6,0",
                "32,7000,12,1",
                "41,9000,24,0",
                "29,6500,8,1",
                "36,8000,18,0",
                "45,11000,30,1",
                "28,6200,7,0",
                "39,8700,20,1",
                "52,13000,36,0",
                "34,7400,14,1",
                "27,5900,5,0",
                "48,11500,33,1",
                "31,6800,9,0",
                "43,9600,22,1",
                "38,8200,19,0",
                "50,12100,34,1",
                "26,5600,4,0",
                "46,10800,28,1",
                "35,7900,16,0",
                "44,9900,25,1",
            ]
        ),
        encoding="utf-8",
    )

    print("Running DeciSense local smoke test...")
    print(f"Workspace: {root}")

    # Branch no: analysis-only package
    analyze_no = _run_cli(
        [
            "analyze-upload",
            "--chat-id",
            "smoke_chat_no",
            "--file-path",
            str(dataset_path),
            "--run-id",
            "smoke_no",
            "--runs-root",
            str(runs_root),
            "--session-store",
            str(session_store),
        ]
    )
    _assert(
        analyze_no["status"] == "analysis_started", "analysis branch no did not start"
    )
    _assert(
        analyze_no["message_type"] == "training_approval",
        "branch no did not ask approval",
    )

    reply_no = _run_cli(
        [
            "handle-text",
            "--chat-id",
            "smoke_chat_no",
            "--message-text",
            "no",
            "--runs-root",
            str(runs_root),
            "--session-store",
            str(session_store),
        ]
    )
    _assert(
        reply_no["message_type"] == "package_ready", "no branch did not create package"
    )
    _assert(reply_no["package_path"], "no branch package path missing")
    _assert(
        Path(reply_no["package_path"]).exists(), "analysis-only package file missing"
    )

    # Branch yes: training package
    analyze_yes = _run_cli(
        [
            "analyze-upload",
            "--chat-id",
            "smoke_chat_yes",
            "--file-path",
            str(dataset_path),
            "--run-id",
            "smoke_yes",
            "--runs-root",
            str(runs_root),
            "--session-store",
            str(session_store),
        ]
    )
    _assert(
        analyze_yes["status"] == "analysis_started", "analysis branch yes did not start"
    )
    _assert(
        analyze_yes["message_type"] == "training_approval",
        "branch yes did not ask approval",
    )

    reply_yes = _run_cli(
        [
            "handle-text",
            "--chat-id",
            "smoke_chat_yes",
            "--message-text",
            "yes",
            "--runs-root",
            str(runs_root),
            "--session-store",
            str(session_store),
        ]
    )
    _assert(
        reply_yes["message_type"] == "training_completed",
        "yes branch did not finish training",
    )
    _assert(reply_yes["package_path"], "yes branch package path missing")
    _assert(
        Path(reply_yes["package_path"]).exists(), "full training package file missing"
    )

    print("")
    print("Smoke test passed.")
    print(f"Analysis-only package: {reply_no['package_path']}")
    print(f"Full training package: {reply_yes['package_path']}")
    return 0


def _run_cli(args: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "ds_engine.interfaces.decisense_cli",
        *args,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "CLI command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return json.loads(completed.stdout)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    raise SystemExit(main())
