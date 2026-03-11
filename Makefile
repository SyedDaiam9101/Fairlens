.PHONY: install lint test docker-build serve demo-image demo-webcam init-db clean

# Variables
PYTHON := python
PIP := pip
UVICORN := uvicorn
DOCKER := docker

# Installation
install:
	poetry install

install-dev:
	poetry install --with dev

# Linting
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Type checking
typecheck:
	mypy src/detectify/

# Testing
test:
	pytest -v --cov=detectify tests/

test-fast:
	pytest -v tests/ -x --tb=short

# Database
init-db:
	$(PYTHON) scripts/init_db.py

migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

# Server
serve:
	$(UVICORN) detectify.api.server:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	$(UVICORN) detectify.api.server:app --host 0.0.0.0 --port 8000

# Demo commands
demo-image:
	$(PYTHON) -m detectify inference --source sample.jpg --output output.jpg

demo-webcam:
	$(PYTHON) -m detectify inference

demo-video:
	$(PYTHON) -m detectify inference --source sample.mp4 --output output.mp4

# Docker
docker-build:
	$(DOCKER) build -t detectify:latest .

docker-run:
	$(DOCKER) run -p 8000:8000 --rm detectify:latest

docker-run-gpu:
	$(DOCKER) run --gpus all -p 8000:8000 --rm detectify:latest

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/detectify/__pycache__
	rm -rf htmlcov/ .coverage
	rm -f detectify.db
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Help
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies with Poetry"
	@echo "  lint         - Run linters (ruff, black)"
	@echo "  test         - Run tests with coverage"
	@echo "  init-db      - Initialize database tables"
	@echo "  serve        - Start FastAPI dev server"
	@echo "  demo-image   - Run detection on sample.jpg"
	@echo "  demo-webcam  - Run live webcam detection"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
