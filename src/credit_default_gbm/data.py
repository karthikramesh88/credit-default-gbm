"""Load and validate input data: the public UCI credit-card dataset, or a synthetic stand-in."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BILL_AMT, ID_COL, PAY_AMT, PAY_STATUS, RAW_COLUMNS, TARGET, UCI_FILE


class DataValidationError(ValueError):
    """Raised when input data breaks the expected schema. Training stops rather than guess."""


def load_uci(path: Path = UCI_FILE) -> pd.DataFrame:
    """Load 'Default of Credit Card Clients' (Taiwan, 2005; 30,000 card holders).

    Source: UCI Machine Learning Repository, dataset 350. The sheet has a title row,
    so the real header is on row 2. The repayment-status column for the most recent
    month is named PAY_0 in the source; it is renamed PAY_1 to match the other months.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from "
            "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
            "and put the .xls file in data/raw/ (see README)."
        )
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    validate(df)
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Fail loudly on schema problems instead of training on bad data."""
    required = RAW_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if df.empty:
        raise DataValidationError("no rows")
    if df[ID_COL].duplicated().any():
        raise DataValidationError(f"{int(df[ID_COL].duplicated().sum())} duplicate IDs")
    numeric = [c for c in required if c != ID_COL]
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise DataValidationError(f"non-numeric columns: {bad}")
    if df[numeric].isna().any().any():
        nulls = df[numeric].isna().sum()
        raise DataValidationError(f"nulls found: {nulls[nulls > 0].to_dict()}")
    if (df["LIMIT_BAL"] <= 0).any():
        raise DataValidationError("credit limit must be positive")
    if require_target and not set(df[TARGET].unique()) <= {0, 1}:
        raise DataValidationError("target must be 0/1")


def make_synthetic(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Synthetic card holders shaped like the UCI data, for tests and a first dry run.

    Default risk rises with recent late payments and high utilisation, so a working
    model should find real signal. It is not a substitute for the real dataset.
    """
    rng = np.random.default_rng(seed)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 360_000, 500_000], n)
    risk = rng.normal(0, 1, n)
    df = pd.DataFrame({
        ID_COL: np.arange(1, n + 1),
        "LIMIT_BAL": limit,
        "EDUCATION": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02]),
        "SEX": rng.choice([1, 2], n),
        "MARRIAGE": rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02]),
        "AGE": rng.integers(21, 70, n),
    })
    for i, col in enumerate(PAY_STATUS):
        late = np.clip(np.round(risk * 1.2 + rng.normal(-0.6, 1, n) - i * 0.1), -2, 8)
        df[col] = late.astype(int)
    util = np.clip(0.45 + 0.2 * risk + rng.normal(0, 0.2, n), 0, 1.2)
    for i, col in enumerate(BILL_AMT):
        df[col] = np.round(limit * np.clip(util * (1 - 0.03 * i) + rng.normal(0, 0.05, n), -0.05, 1.3))
    for i, col in enumerate(PAY_AMT):
        share = np.clip(0.25 - 0.08 * risk + rng.normal(0, 0.08, n), 0, 1)
        df[col] = np.round(np.maximum(df[BILL_AMT[min(i + 1, 5)]], 0) * share)
    # Non-linear on purpose, as in real card books: risk jumps once a customer is two or
    # more months behind, maxed-out cards matter far more if already late, and both
    # idle and maxed-out cards carry more risk than moderate use (a U-shape). A straight-
    # line model cannot capture these, so the boosted model has something real to beat.
    late_now = df["PAY_1"] > 0
    logit = (
        -3.4
        + 2.0 * (df["PAY_1"] >= 2)
        + 0.8 * (df["PAY_2"] >= 2)
        + 1.8 * ((util > 0.85) & late_now)
        + 10.0 * (util - 0.45) ** 2
        + 0.3 * risk
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
