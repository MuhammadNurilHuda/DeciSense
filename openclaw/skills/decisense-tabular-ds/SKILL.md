---
name: decisense-tabular-ds
description: Use DeciSense to analyze tabular datasets, recommend an initial ML model, handle yes/no training approval, and generate analysis/training packages.
---

# DeciSense Tabular DS Skill

Use this skill when the user wants DeciSense to process a tabular dataset through the local DeciSense engine.

DeciSense supports:
- tabular dataset analysis
- target inference
- target clarification using `target:<column_name>`
- model recommendation
- strict training approval using `yes` or `no`
- `reset` to clear the active chat session
- analysis-only package when the user replies `no`
- full training package when the user replies `yes`

## Important behavior

- Only handle tabular files.
- Do not treat `ya`, `tidak`, `lanjut`, or localized replies as yes/no approval.
- Only `yes` and `no` are valid approval replies.
- The reset command is exactly `reset`, not `/reset`.
- `reset` clears only the active session state. It does not delete artifacts under `runs/`.
- If the user uploads a new dataset while a session is active, run the CLI anyway; DeciSense will return an `active_session_exists` message.

## Commands

When the user uploads a dataset file, run:

```bash
uv run python -m ds_engine.interfaces.decisense_cli analyze-upload \
  --chat-id "<chat_id>" \
  --file-path "<uploaded_file_path>" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

When the user sends a text message such as yes, no, reset, or target:<column_name>, run:
```bash
uv run python -m ds_engine.interfaces.decisense_cli handle-text \
  --chat-id "<chat_id>" \
  --message-text "<message_text>" \
  --runs-root "runs" \
  --session-store "bot_state/sessions.json"
```

## Response handling

The CLI returns JSON.

Use these fields:
- text: message to send back to the user
- expects_reply: whether DeciSense expects another reply
- reply_hint: expected reply format
- package_path: package file path if a package was created
- message_type: message category
- status: service status
- session_state: persisted session state

If package_path is present, send the package file to the user if the channel supports file attachments. Otherwise, tell the user the generated local path.

Typical flow
1. User uploads dataset.
2. Run analyze-upload.
3. Send the returned text.
4. If user replies no, run handle-text; DeciSense creates an analysis-only package.
5. If user replies yes, run handle-text; DeciSense runs training and creates a full training package.
6. If user replies invalid text, send the returned invalid-reply message.
7. If user sends reset, run handle-text and send the reset confirmation.