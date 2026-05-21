from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ds_engine.workflows.decisense_service import (
    DeciSenseService,
    DeciSenseServiceConfig,
    DeciSenseServiceResult,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DeciSense CLI interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        service = _build_service(args)

        if args.command == "analyze-upload":
            result = service.start_analysis_for_upload(
                chat_id=args.chat_id,
                file_path=args.file_path,
                run_id=args.run_id,
            )
            _print_service_result(result, full_json=args.full_json)
            return 0

        if args.command == "handle-text":
            result = service.handle_text_message(
                chat_id=args.chat_id,
                message_text=args.message_text,
            )
            _print_service_result(result, full_json=args.full_json)
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2

    except Exception as exc:
        _print_json(
            {
                "status": "cli_error",
                "message": str(exc),
                "errors": [str(exc)],
            }
        )
        return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="decisense",
        description="DeciSense local service CLI for OpenClaw/Telegram integration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze-upload",
        help="Start DeciSense analysis for an uploaded tabular dataset.",
    )
    _add_common_service_args(analyze_parser)
    analyze_parser.add_argument("--chat-id", required=True)
    analyze_parser.add_argument(
        "--file-path",
        required=True,
        help="Path to exactly one uploaded tabular dataset file.",
    )
    analyze_parser.add_argument("--run-id", default=None)

    text_parser = subparsers.add_parser(
        "handle-text",
        help="Handle a text message such as reset, yes, no, or target:<column>.",
    )
    _add_common_service_args(text_parser)
    text_parser.add_argument("--chat-id", required=True)
    text_parser.add_argument("--message-text", required=True)

    return parser


def _add_common_service_args(parser: argparse.ArgumentParser) -> None:
    """Add service configuration arguments shared by all subcommands."""
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Directory used for run artifacts.",
    )
    parser.add_argument(
        "--session-store",
        default="bot_state/sessions.json",
        help="JSON file used for persistent chat sessions.",
    )
    parser.add_argument(
        "--no-copy-uploads",
        action="store_true",
        help="Use uploaded file in place instead of copying it to runs/<run_id>/raw/.",
    )
    parser.add_argument(
        "--include-source-data",
        action="store_true",
        help="Include raw source data in generated packages.",
    )
    parser.add_argument(
        "--no-persist-training-artifacts",
        action="store_true",
        help="Do not persist intermediate training artifacts.",
    )
    parser.add_argument(
        "--no-training-package",
        action="store_true",
        help="Do not create a full training package after yes approval.",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the full nested service result instead of the compact payload.",
    )


def _build_service(args: argparse.Namespace) -> DeciSenseService:
    """Create DeciSense service from CLI args."""
    config = DeciSenseServiceConfig(
        runs_root=Path(args.runs_root),
        session_store_path=Path(args.session_store),
        copy_uploads_to_runs=not args.no_copy_uploads,
        include_source_data_in_packages=args.include_source_data,
        persist_training_artifacts=not args.no_persist_training_artifacts,
        create_training_package=not args.no_training_package,
    )
    return DeciSenseService(config=config)


def _print_service_result(
    result: DeciSenseServiceResult,
    *,
    full_json: bool,
) -> None:
    """Print service result as JSON."""
    if full_json:
        _print_json(result.to_dict())
        return

    _print_json(_compact_service_payload(result))


def _compact_service_payload(result: DeciSenseServiceResult) -> dict[str, Any]:
    """Return compact JSON payload suitable for OpenClaw agent use."""
    message_payload = result.telegram_message.to_dict()

    return {
        "chat_id": result.chat_id,
        "status": result.status,
        "run_id": result.run_id,
        "session_state": result.session.state if result.session else None,
        "message_type": result.telegram_message.message_type,
        "text": result.telegram_message.text,
        "expects_reply": result.telegram_message.expects_reply,
        "reply_hint": result.telegram_message.reply_hint,
        "package_path": str(result.package_path) if result.package_path else None,
        "message_metadata": message_payload["metadata"],
        "errors": list(result.errors),
    }


def _print_json(payload: dict[str, Any]) -> None:
    """Print JSON payload to stdout."""
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
