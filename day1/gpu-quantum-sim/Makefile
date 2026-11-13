# Makefile for gpu-quantum-sim: NVIDIA CUDA-Q Quantum Simulation & Chemistry Benchmark Suite

PYTHON ?= python
PIP ?= pip

.PHONY: all install test benchmark vqe clean help

all: install test benchmark vqe ## Run full pipeline: install, test, benchmark GHZ, and execute H2 VQE

install: ## Install pinned dependencies
	$(PIP) install -r requirements.txt

test: ## Execute complete pytest validation suite
	$(PYTHON) -m pytest tests/ -v

benchmark: ## Run GHZ statevector simulation benchmark across qubit counts
	$(PYTHON) src/ghz_benchmark.py

benchmark-quick: ## Run reduced GHZ benchmark (< 1 minute)
	$(PYTHON) src/ghz_benchmark.py --quick

vqe: ## Run H2 VQE potential energy surface sweep & convergence analysis
	$(PYTHON) src/h2_vqe.py

vqe-quick: ## Run reduced H2 VQE sweep (< 2 minutes)
	$(PYTHON) src/h2_vqe.py --quick

clean: ## Remove generated logs and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

help: ## Show this help message
	@echo "Usage: make [target]"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'
