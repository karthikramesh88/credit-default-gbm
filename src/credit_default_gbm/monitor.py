"""Drift monitoring: has the population being scored moved away from the one the model learned on?

    python -m credit_default_gbm.monitor --batch path/to/new_applicants.csv

Uses the population stability index (PSI) on every feature and on the model score.
"""

import argparse
import json

import numpy as np
import pandas as pd

from .config import PSI_ACT, PSI_WATCH

SCORE = "__score__"


def _edges(values: np.ndarray, bins: int = 10) -> list[float]:
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])).tolist()


def _shares(values: np.ndarray, edges: list[float]) -> list[float]:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return (counts / max(len(values), 1)).tolist()


def reference_profile(features: pd.DataFrame, scores: np.ndarray) -> dict:
    """Bin edges and shares from the training population, stored with the model."""
    profile = {}
    for name, values in [*features.items(), (SCORE, pd.Series(scores))]:
        edges = _edges(values.to_numpy())
        profile[name] = {"edges": edges, "shares": _shares(values.to_numpy(), edges)}
    return profile


def psi(expected: list[float], actual: list[float], floor: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected), floor, None)
    a = np.clip(np.asarray(actual), floor, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "investigate" if value >= PSI_ACT else "watch" if value >= PSI_WATCH else "stable"


def drift_report(reference: dict, features: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, ref in reference.items():
        values = scores if name == SCORE else features[name].to_numpy()
        value = psi(ref["shares"], _shares(values, ref["edges"]))
        rows.append({"feature": "model score" if name == SCORE else name, "psi": round(value, 4), "status": status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def main() -> None:
    from . import model_store
    from .data import validate
    from .features import build_features

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="CSV of raw applicant rows")
    args = ap.parse_args()
    model = model_store.load()
    raw = pd.read_csv(args.batch)
    validate(raw, require_target=False)
    features = build_features(raw)
    report = drift_report(model.metadata["reference"], features, model.probability(features))
    print(f"model {model.version}; {len(raw):,} rows scored")
    print(report.to_string(index=False))
    flagged = report[report["status"] != "stable"]
    print(json.dumps({"overall": "investigate" if (report["status"] == "investigate").any()
                      else "watch" if len(flagged) else "stable"}))


if __name__ == "__main__":
    main()
