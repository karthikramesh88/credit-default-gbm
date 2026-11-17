"""Fairness audit: does the approved model treat groups differently, by sex and age band?

    python -m credit_default_gbm.fairness --data uci

Sex, marital status and age are never model features, but other features can stand in
for them. This audit scores the held-out test set (the same split training used) and
compares, for each group:

- calibration: does the predicted default rate match the actual one?
- ranking: AUC within the group
- decisions, under a policy of declining band E (the riskiest fifth): approval rate,
  adverse impact ratio against the most-approved group, wrongly declined good
  customers, and missed defaults

Different approval rates alone are not proof of bias if actual default rates differ
too. The warning signs are a group whose risk is over-predicted, or whose good
customers are declined more often.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import model_store
from .config import REPORTS_DIR, SEED, TARGET
from .data import load_uci, make_synthetic
from .features import build_features
from .train import split

FOUR_FIFTHS = 0.8  # adverse impact ratio below this is the conventional red flag
MISCALIBRATION_FLAG = 0.03  # predicted vs actual default rate gap, absolute

AGE_BANDS = [(21, 29, "21-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 120, "50+")]


def groups(raw: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.Series("other", index=raw.index)
    for lo, hi, label in AGE_BANDS:
        age[raw["AGE"].between(lo, hi)] = label
    return {
        "sex": raw["SEX"].map({1: "male", 2: "female"}).fillna("other"),
        "age band": age,
    }


def audit_table(y: np.ndarray, p: np.ndarray, declined: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows = []
    for name in sorted(group.unique()):
        m = (group == name).to_numpy()
        yg, pg, dg = y[m], p[m], declined[m]
        good, bad = yg == 0, yg == 1
        rows.append({
            "group": name,
            "customers": int(m.sum()),
            "actual default rate": yg.mean(),
            "predicted default rate": pg.mean(),
            "AUC": roc_auc_score(yg, pg) if 0 < yg.sum() < len(yg) else np.nan,
            "approval rate": 1 - dg.mean(),
            "good customers declined": dg[good].mean() if good.any() else np.nan,
            "defaults approved": (~dg[bad]).mean() if bad.any() else np.nan,
        })
    t = pd.DataFrame(rows)
    t["calibration gap"] = t["predicted default rate"] - t["actual default rate"]
    t["adverse impact ratio"] = t["approval rate"] / t["approval rate"].max()
    t["flags"] = [
        ", ".join(f for f, hit in (
            ("adverse impact < 0.8", r["adverse impact ratio"] < FOUR_FIFTHS),
            ("risk over-predicted", r["calibration gap"] > MISCALIBRATION_FLAG),
            ("risk under-predicted", r["calibration gap"] < -MISCALIBRATION_FLAG),
        ) if hit) or "none"
        for _, r in t.iterrows()
    ]
    return t


def run_audit(raw: pd.DataFrame, model: model_store.Model, seed: int = SEED) -> dict[str, pd.DataFrame]:
    _, _, test = split(raw, seed)
    features = build_features(test)
    p = model.probability(features)
    declined = np.asarray(model.band(p)) == "E"
    y = test[TARGET].to_numpy()
    return {name: audit_table(y, p, declined, g) for name, g in groups(test).items()}


def out_of_fold(raw: pd.DataFrame, folds: int = 5, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """Score every customer with a model that never saw them (k-fold cross-fitting).

    Gives group estimates on the whole dataset instead of one 20% test set, so small
    gaps can be told apart from noise. Returns calibrated probabilities and each
    feature's TreeSHAP contribution (log-odds) per customer.
    """
    import tempfile

    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    from .train import train

    raw = raw.reset_index(drop=True)
    p = np.zeros(len(raw))
    contrib = None
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for k, (fit_idx, held_idx) in enumerate(splitter.split(raw, raw[TARGET])):
        with tempfile.TemporaryDirectory() as d:
            meta = train(raw.iloc[fit_idx], "cross-fit", Path(d), seed=seed + k)
            model = model_store.load(meta["version"], Path(d))
        f = build_features(raw.iloc[held_idx])[meta["features"]]
        p[held_idx] = model.probability(f)
        c = model.booster.predict(xgb.DMatrix(f), pred_contribs=True,
                                  iteration_range=(0, meta["best_iteration"] + 1))[:, :-1]
        if contrib is None:
            contrib = pd.DataFrame(0.0, index=raw.index, columns=meta["features"])
        contrib.iloc[held_idx] = c
    return p, contrib


def calibration_by_group(y: np.ndarray, p: np.ndarray, group: pd.Series) -> pd.DataFrame:
    """Predicted minus actual default rate per group, with a z-score against sampling noise."""
    t = (pd.DataFrame({"y": y, "p": p, "group": np.asarray(group)})
         .groupby("group").agg(customers=("y", "size"), actual=("y", "mean"), predicted=("p", "mean")))
    t["gap"] = t["predicted"] - t["actual"]
    t["z"] = t["gap"] / np.sqrt(t["actual"] * (1 - t["actual"]) / t["customers"])
    t["significant"] = t["z"].abs() >= 2
    return t.reset_index()


def drivers(contrib: pd.DataFrame, in_group: pd.Series, top: int = 5) -> pd.Series:
    """Features that push the group's predicted risk furthest from everyone else's (log-odds)."""
    mask = np.asarray(in_group)
    diff = contrib[mask].mean() - contrib[~mask].mean()
    return diff.reindex(diff.abs().sort_values(ascending=False).index)[:top]


def to_markdown(tables: dict[str, pd.DataFrame], model: model_store.Model) -> str:
    pct = ["actual default rate", "predicted default rate", "approval rate",
           "good customers declined", "defaults approved", "calibration gap"]
    lines = [f"# Fairness audit — model {model.version}", "",
             f"Held-out test set; policy: decline risk band E (probability of default above "
             f"{model.metadata['band_cutoffs'][-1]:.1%}).", ""]
    for name, t in tables.items():
        shown = t.copy()
        for c in pct:
            shown[c] = shown[c].map(lambda v: f"{v:+.1%}" if c == "calibration gap" else f"{v:.1%}")
        shown["AUC"] = shown["AUC"].map("{:.3f}".format)
        shown["adverse impact ratio"] = shown["adverse impact ratio"].map("{:.2f}".format)
        lines += [f"## By {name}", "", shown.to_markdown(index=False), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["uci", "synthetic"], default="uci")
    ap.add_argument("--cross-fit", action="store_true",
                    help="estimate calibration gaps on every customer via 5-fold cross-fitting, and their drivers")
    args = ap.parse_args()
    if args.cross_fit:
        raw = load_uci() if args.data == "uci" else make_synthetic(30_000, seed=SEED)
        raw = raw.reset_index(drop=True)
        p, contrib = out_of_fold(raw)
        y = raw[TARGET].to_numpy()
        for name, g in groups(raw).items():
            print(f"\ncalibration by {name} ({len(raw):,} customers, each scored by a model that never saw them)")
            print(calibration_by_group(y, p, g).round(4).to_string(index=False))
            for label in sorted(g.unique()):
                print(f"  drivers for {label}: " + ", ".join(
                    f"{k} {v:+.3f}" for k, v in drivers(contrib, g == label, top=3).items()))
        return
    model = model_store.load()
    if model.metadata["data_source"] != args.data:
        raise SystemExit(f"approved model was trained on {model.metadata['data_source']!r}, not {args.data!r}")
    raw = load_uci() if args.data == "uci" else make_synthetic(30_000, seed=SEED)
    tables = run_audit(raw, model)
    report = to_markdown(tables, model)
    out = Path(REPORTS_DIR) / f"fairness_{model.version}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
