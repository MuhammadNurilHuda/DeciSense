1. open wsl
2. cd ~/projects/DeciSense
3. uv sync --extra dev
4. uv run pytest
5. uv run python <file>.py
    e.g., uv run python ds_engine/pipeline.py
6. add package to uv
    - uv add <package>
    - uv add --dev <package> -> for dev
    - uv sync --upgrade

DO NOT USE 
- pip install ...
ALWAYS USE:
- uv add ...

Always use WSL

Use project environment only

Check installed packages
- uv pip list

REBUILD ENV IF BROKEN
rm -rf .venv
uv sync --extra dev


Quick Start (TL;DR)
wsl
cd ~/projects/DeciSense
uv sync --extra dev
uv run pytest