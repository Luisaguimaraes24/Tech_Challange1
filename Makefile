# Makefile — Telco Churn Prediction (FIAP Tech Challenge Fase 1)
# Referência: Eng. Software, Aulas 04–05

.PHONY: help lint lint-fix test test-cov test-schema test-pipeline test-api train run run-prod eda baselines mlp all clean install

CYAN  := \033[0;36m
RESET := \033[0m

help:
	@echo ""
	@echo "Telco Churn Prediction — Comandos disponíveis:"
	@echo "  install      Instala dependências do projeto (incluindo dev)"
	@echo "  lint         Roda ruff check e format --check"
	@echo "  lint-fix     Roda ruff e aplica correções automáticas"
	@echo "  test         Roda todos os testes com pytest"
	@echo "  test-cov     Roda testes com relatório de cobertura"
	@echo "  test-schema  Testes de schema (pandera)"
	@echo "  test-pipeline Testes unitários do pipeline"
	@echo "  test-api     Testes da API"
	@echo "  train        Treina o pipeline e salva em models/"
	@echo "  run          Sobe a API FastAPI com --reload"
	@echo "  run-prod     Sobe a API sem reload (produção)"
	@echo "  eda          Roda EDA completa"
	@echo "  all          Pipeline completo: eda -> train -> lint -> test"
	@echo "  clean        Remove artefatos gerados"
	@echo ""

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ api/ tests/ --fix
	ruff format src/ api/ tests/ --check

lint-fix:
	ruff check src/ api/ tests/ --fix
	ruff format src/ api/ tests/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing

test-schema:
	pytest tests/test_schema.py -v

test-pipeline:
	pytest tests/test_pipeline.py -v

test-api:
	pytest tests/test_api.py -v

train:
	python src/pipeline.py

run:
	uvicorn api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000 --workers 1

eda:
	python notebooks/01_eda.py

baselines:
	python notebooks/02_baselines.py

mlp:
	python notebooks/04_mlp.py

all: eda train lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Limpeza concluida."