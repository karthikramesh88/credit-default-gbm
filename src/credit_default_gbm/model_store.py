"""Save, load and score with a versioned model. Each version is a folder; LATEST names the approved one."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODELS_DIR
from .features import DESCRIPTIONS, build_features

BANDS = ["A", "B", "C", "D", "E"]  # A = lowest risk


@dataclass
class Model:
    booster: xgb.Booster
    calibrator: object
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def _matrix(self, features: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(features[self.metadata["features"]])

    def _iterations(self) -> tuple[int, int]:
        return (0, self.metadata["best_iteration"] + 1)

    def raw_score(self, features: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._matrix(features), iteration_range=self._iterations())

    def probability(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default next month."""
        return self.calibrator.predict(self.raw_score(features))

    def band(self, probability: np.ndarray) -> list[str]:
        cutoffs = self.metadata["band_cutoffs"]
        return [BANDS[int(np.searchsorted(cutoffs, p, side="right"))] for p in probability]

    def reason_codes(self, features: pd.DataFrame, top: int = 3) -> list[list[str]]:
        """The features pushing each applicant's risk up most (TreeSHAP contributions)."""
        contribs = self.booster.predict(
            self._matrix(features), pred_contribs=True, iteration_range=self._iterations()
        )[:, :-1]  # last column is the bias term
        names = self.metadata["features"]
        out = []
        for row in contribs:
            order = np.argsort(row)[::-1]
            out.append([DESCRIPTIONS[names[i]] for i in order[:top] if row[i] > 0])
        return out

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        features = build_features(raw)
        p = self.probability(features)
        return pd.DataFrame({
            "probability_of_default": p,
            "risk_band": self.band(p),
            "reason_codes": self.reason_codes(features),
        }, index=raw.index)


def save(version_dir: Path, booster: xgb.Booster, calibrator, metadata: dict) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(version_dir / "model.json")
    joblib.dump(calibrator, version_dir / "calibrator.joblib")
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def promote(models_dir: Path, version: str) -> None:
    (models_dir / "LATEST").write_text(version)


def load(version: str | None = None, models_dir: Path = MODELS_DIR) -> Model:
    if version is None:
        latest = models_dir / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(f"no approved model in {models_dir}; run training first")
        version = latest.read_text().strip()
    d = models_dir / version
    booster = xgb.Booster()
    booster.load_model(d / "model.json")
    return Model(booster, joblib.load(d / "calibrator.joblib"), json.loads((d / "metadata.json").read_text()))
