"""Train, validate and (if it passes the release gate) approve a new model version.

    python -m credit_default_gbm.train --data uci        # real data in data/raw/
    python -m credit_default_gbm.train --data synthetic  # dry run, no download needed
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import model_store
from .config import (
    GATE_MAX_DECILE_INVERSIONS, GATE_MIN_AUC_LIFT, ID_COL, MODELS_DIR, SEED, TARGET,
)
from .data import load_uci, make_synthetic
from .features import MONOTONE, build_features
from .monitor import reference_profile

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
)


def split(raw: pd.DataFrame, seed: int):
    """60/20/20 train / validation / test, stratified on the outcome.

    The UCI data has no application date, so this split is random. With real
    lending data, split by time instead (train on older months, test on newer)
    so the test mirrors how the model meets future customers.
    """
    train, rest = train_test_split(raw, test_size=0.4, stratify=raw[TARGET], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return train, valid, test


def evaluate(y: np.ndarray, p: np.ndarray, rank_score: np.ndarray | None = None) -> dict:
    """Ranking metrics (AUC, Gini, KS, deciles) use rank_score; calibration metrics use p.

    Isotonic calibration maps scores onto flat steps, so ranking on the calibrated
    probability would add artificial ties. The calibration is monotone, so the raw
    score carries the same order without them.
    """
    r = p if rank_score is None else rank_score
    fpr, tpr, _ = roc_curve(y, r)
    auc = roc_auc_score(y, r)
    deciles = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False)
    grouped = pd.Series(y).groupby(deciles.values)
    rates, sizes = grouped.mean().tolist(), grouped.size().tolist()
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(float(np.max(tpr - fpr)), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "default_rate_by_decile": [round(r, 4) for r in rates],
        "decile_inversions": _inversions(rates, sizes),
    }


def _inversions(rates: list[float], sizes: list[int]) -> int:
    """Count deciles whose default rate falls below the decile before by more than noise.

    A drop counts only if it exceeds two standard errors of the difference, so small
    test sets do not fail the gate on sampling luck.
    """
    count = 0
    for (a, na), (b, nb) in zip(zip(rates, sizes), zip(rates[1:], sizes[1:])):
        se = np.sqrt(a * (1 - a) / na + b * (1 - b) / nb)
        count += int(a - b > 2 * se)
    return count


def gate(model_m: dict, baseline_m: dict) -> dict:
    checks = {
        "beats_baseline_auc": model_m["auc"] - baseline_m["auc"] >= GATE_MIN_AUC_LIFT,
        "calibration_no_worse": model_m["brier"] <= baseline_m["brier"],
        "rank_ordering_holds": model_m["decile_inversions"] <= GATE_MAX_DECILE_INVERSIONS,
    }
    return {"approved": all(checks.values()), "checks": checks}


def train(raw: pd.DataFrame, source: str, models_dir: Path = MODELS_DIR, seed: int = SEED) -> dict:
    tr, va, te = split(raw, seed)
    X = {k: build_features(d) for k, d in (("train", tr), ("valid", va), ("test", te))}
    y = {k: d[TARGET].to_numpy() for k, d in (("train", tr), ("valid", va), ("test", te))}

    # Baseline: the model credit teams have used for decades. The new model must beat it.
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X["train"], y["train"])
    baseline_m = evaluate(y["test"], baseline.predict_proba(X["test"])[:, 1])

    monotone = {name: MONOTONE.get(name, 0) for name in X["train"].columns}
    clf = xgb.XGBClassifier(**PARAMS, monotone_constraints=monotone, random_state=seed)
    clf.fit(X["train"], y["train"], eval_set=[(X["valid"], y["valid"])], verbose=False)
    booster = clf.get_booster()
    best = int(clf.best_iteration)

    # Calibrate raw scores to true default probabilities on the validation set.
    raw_valid = booster.predict(xgb.DMatrix(X["valid"]), iteration_range=(0, best + 1))
    # Bounded so no customer is ever scored as certain to pay or certain to default.
    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(raw_valid, y["valid"])

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "rows": {k: int(len(v)) for k, v in y.items()},
        "default_rate_train": round(float(y["train"].mean()), 4),
        "features": list(X["train"].columns),
        "monotone_constraints": {k: v for k, v in monotone.items() if v},
        "params": {k: v for k, v in PARAMS.items()},
        "best_iteration": best,
    }
    model = model_store.Model(booster, calibrator, metadata)
    p_valid = model.probability(X["valid"])
    p_test = model.probability(X["test"])
    raw_test = model.raw_score(X["test"])
    metadata["band_cutoffs"] = [float(c) for c in np.quantile(p_valid, [0.2, 0.4, 0.6, 0.8])]
    metadata["metrics_test"] = {"xgboost": evaluate(y["test"], p_test, raw_test), "logistic_baseline": baseline_m}
    metadata["gate"] = gate(metadata["metrics_test"]["xgboost"], baseline_m)
    metadata["reference"] = reference_profile(X["train"], model.probability(X["train"]))
    metadata["status"] = "approved" if metadata["gate"]["approved"] else "rejected"

    model_store.save(models_dir / version, booster, calibrator, metadata)
    if metadata["gate"]["approved"]:
        model_store.promote(models_dir, version)
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["uci", "synthetic"], default="uci")
    ap.add_argument("--rows", type=int, default=30_000, help="synthetic rows")
    args = ap.parse_args()
    raw = load_uci() if args.data == "uci" else make_synthetic(args.rows, seed=SEED)
    assert raw[ID_COL].is_unique
    meta = train(raw, source=args.data)
    m, b = meta["metrics_test"]["xgboost"], meta["metrics_test"]["logistic_baseline"]
    print(f"version {meta['version']}  ({meta['status'].upper()})  trees used: {meta['best_iteration'] + 1}")
    print(f"{'':10}{'AUC':>8}{'Gini':>8}{'KS':>8}{'Brier':>8}")
    for name, x in (("xgboost", m), ("baseline", b)):
        print(f"{name:10}{x['auc']:>8}{x['gini']:>8}{x['ks']:>8}{x['brier']:>8}")
    print("default rate by score decile (low to high risk):", m["default_rate_by_decile"])
    print("release gate:", json.dumps(meta["gate"]["checks"]))


if __name__ == "__main__":
    main()
