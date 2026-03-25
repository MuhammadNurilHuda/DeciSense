# Makefile
.PHONY: help setup test lint format clean

help:
	@echo "Available commands:"
	@echo "  setup    Install dependencies and pre-commit"
	@echo "  test     Run tests with coverage"
	@echo "  lint     Run linting checks"
	@echo "  format   Format code with black"
	@echo "  clean    Remove cache files"

setup:
	python -m venv venv
	./venv/bin/pip install -U pip
	./venv/bin/pip install -r requirements-dev.txt
	./venv/bin/pre-commit install
	mkdir -p data outputs
	touch data/.gitkeep outputs/.gitkeep

test:
	pytest tests/ -v --cov=decisense --cov-report=html

lint:
	flake8 decisense/
	mypy decisense/

format:
	black decisense/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov/