"""Paths and constants shared across the pipeline."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
UCI_FILE = DATA_RAW / "default of credit card clients.xls"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"

ID_COL = "ID"
TARGET = "default_next_month"
SEED = 42

MONTHS = range(1, 7)  # 1 = most recent month, 6 = oldest
PAY_STATUS = [f"PAY_{m}" for m in MONTHS]
BILL_AMT = [f"BILL_AMT{m}" for m in MONTHS]
PAY_AMT = [f"PAY_AMT{m}" for m in MONTHS]

# Attributes most credit-scoring rules bar from decisions. They are kept in the
# raw data for fairness checks but never become model features.
PROTECTED = ["SEX", "MARRIAGE", "AGE"]

RAW_COLUMNS = [ID_COL, "LIMIT_BAL", "EDUCATION", *PROTECTED, *PAY_STATUS, *BILL_AMT, *PAY_AMT]

# Release gate: the gradient-boosted model must clearly beat a logistic-regression
# baseline on held-out data, or it is not approved for serving.
GATE_MIN_AUC_LIFT = 0.005
GATE_MAX_DECILE_INVERSIONS = 1

# Population stability index thresholds (industry convention).
PSI_WATCH = 0.10
PSI_ACT = 0.25
