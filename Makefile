# XGBoost on macOS needs the OpenMP runtime (libomp). Rather than require Homebrew,
# point it at the copy scikit-learn already ships. Harmless on other systems.
OMP := DYLD_LIBRARY_PATH=$(CURDIR)/.venv/lib/python3.11/site-packages/sklearn/.dylibs
# uv run drops DYLD_* variables, so call the project's interpreter directly.
PY := $(OMP) .venv/bin/python

.PHONY: setup train train-synthetic serve monitor-sample fairness fairness-crossfit test

setup:            ## create .venv with all dependencies
	uv sync

train:            ## train on the real UCI dataset (data/raw/)
	$(PY) -m credit_default_gbm.train --data uci

train-synthetic:  ## dry run on synthetic data, no download needed
	$(PY) -m credit_default_gbm.train --data synthetic

serve:            ## start the scoring API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m uvicorn credit_default_gbm.api:app --port 8000

monitor-sample:   ## drift check on a synthetic batch with inflated balances
	$(PY) -c "from credit_default_gbm.data import make_synthetic as m; d=m(3000, seed=5); d[[f'BILL_AMT{i}' for i in range(1,7)]]*=1.5; d.drop(columns='default_next_month').to_csv('reports/sample_batch.csv', index=False)"
	$(PY) -m credit_default_gbm.monitor --batch reports/sample_batch.csv

fairness:         ## fairness audit of the approved model (by sex and age band)
	$(PY) -m credit_default_gbm.fairness --data uci

fairness-crossfit: ## calibration gaps on all customers via cross-fitting, with drivers
	$(PY) -m credit_default_gbm.fairness --data uci --cross-fit

test:
	$(PY) -m pytest -q
