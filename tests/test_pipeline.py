import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_default_gbm import model_store
from credit_default_gbm.config import BILL_AMT, PAY_AMT, PAY_STATUS, PROTECTED
from credit_default_gbm.data import DataValidationError, make_synthetic, validate
from credit_default_gbm.features import build_features, feature_names
from credit_default_gbm.monitor import drift_report
from credit_default_gbm.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    meta = train(make_synthetic(20000, seed=1), source="synthetic", models_dir=models_dir)
    return models_dir, meta


def test_features_exclude_protected_and_are_finite():
    raw = make_synthetic(500)
    raw.loc[:10, BILL_AMT] = 0  # nothing owed: ratios must not divide by zero
    f = build_features(raw)
    assert not set(PROTECTED) & set(f.columns)
    assert list(f.columns) == feature_names()
    assert np.isfinite(f.to_numpy()).all()


def test_validation_rejects_bad_data():
    raw = make_synthetic(100)
    with pytest.raises(DataValidationError):
        validate(raw.drop(columns=["PAY_1"]))
    dup = raw.copy()
    dup.loc[1, "ID"] = dup.loc[0, "ID"]
    with pytest.raises(DataValidationError):
        validate(dup)


def test_training_beats_baseline_and_is_approved(trained):
    models_dir, meta = trained
    assert meta["status"] == "approved", meta["gate"]
    m = meta["metrics_test"]["xgboost"]
    assert m["auc"] > 0.7
    assert (models_dir / "LATEST").read_text() == meta["version"]


def test_gate_rejects_a_model_that_does_not_beat_baseline():
    from credit_default_gbm.train import gate

    base = {"auc": 0.78, "brier": 0.135, "decile_inversions": 0}
    assert not gate({"auc": 0.782, "brier": 0.134, "decile_inversions": 0}, base)["approved"]
    assert not gate({"auc": 0.80, "brier": 0.140, "decile_inversions": 0}, base)["approved"]
    assert not gate({"auc": 0.80, "brier": 0.130, "decile_inversions": 3}, base)["approved"]
    assert gate({"auc": 0.80, "brier": 0.130, "decile_inversions": 1}, base)["approved"]


def test_reload_scores_identically(trained):
    models_dir, meta = trained
    model = model_store.load(models_dir=models_dir)
    raw = make_synthetic(200, seed=7)
    a = model.score(raw)["probability_of_default"]
    b = model_store.load(meta["version"], models_dir).score(raw)["probability_of_default"]
    assert np.allclose(a, b)
    assert a.between(0, 1).all()


def test_monotone_lateness_never_lowers_risk(trained):
    models_dir, _ = trained
    model = model_store.load(models_dir=models_dir)
    raw = make_synthetic(300, seed=3)
    worse = raw.copy()
    worse[PAY_STATUS] = worse[PAY_STATUS].clip(lower=0) + 2  # every month later
    f_base, f_worse = build_features(raw), build_features(worse)
    # Isolate the constrained features so the comparison is clean.
    for col in ("months_late", "worst_delay"):
        bumped = f_base.copy()
        bumped[col] = f_worse[col]
        assert (model.raw_score(bumped) >= model.raw_score(f_base) - 1e-9).all()


def test_drift_detects_shift(trained):
    models_dir, _ = trained
    model = model_store.load(models_dir=models_dir)
    same = build_features(make_synthetic(4000, seed=11))
    stable = drift_report(model.metadata["reference"], same, model.probability(same))
    assert (stable["status"] == "stable").mean() > 0.8

    shifted_raw = make_synthetic(4000, seed=11)
    shifted_raw[BILL_AMT] = shifted_raw[BILL_AMT] * 1.8  # balances balloon
    shifted = build_features(shifted_raw)
    report = drift_report(model.metadata["reference"], shifted, model.probability(shifted))
    assert report.set_index("feature").loc["util_avg", "status"] == "investigate"


def test_api_scores_applicants(trained, monkeypatch):
    models_dir, meta = trained
    monkeypatch.setattr(model_store, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_store.load, "__defaults__", (None, models_dir))
    from credit_default_gbm.api import app

    applicant = {
        "id": "a1", "LIMIT_BAL": 50000, "EDUCATION": 2,
        "pay_status": [2, 2, 0, 0, -1, -1],
        "bill_amt": [48000, 47000, 45000, 30000, 20000, 10000],
        "pay_amt": [0, 0, 1500, 1500, 2000, 2000],
    }
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == meta["version"]
        assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
        r = client.post("/score", json=[applicant])
        assert r.status_code == 200
        s = r.json()["scores"][0]
        assert 0 <= s["probability_of_default"] <= 1
        assert s["risk_band"] in list("ABCDE")
        assert s["reason_codes"]
        assert client.post("/score", json=[{**applicant, "pay_status": [0]}]).status_code == 422


def test_fairness_audit_table():
    from credit_default_gbm.fairness import audit_table

    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.9, 0.1, 0.1, 0.3, 0.8])
    declined = np.array([False, True, True, True, False, False, False, False])
    group = pd.Series(["a"] * 4 + ["b"] * 4)
    t = audit_table(y, p, declined, group).set_index("group")
    assert t.loc["a", "approval rate"] == 0.25 and t.loc["b", "approval rate"] == 1.0
    assert t.loc["a", "adverse impact ratio"] == 0.25
    assert t.loc["a", "good customers declined"] == 0.5
    assert t.loc["b", "defaults approved"] == 1.0
    assert "adverse impact < 0.8" in t.loc["a", "flags"]


def test_fairness_audit_runs_on_trained_model(trained):
    from credit_default_gbm.fairness import run_audit, to_markdown

    models_dir, _ = trained
    model = model_store.load(models_dir=models_dir)
    tables = run_audit(make_synthetic(20000, seed=1), model)
    assert set(tables) == {"sex", "age band"}
    assert tables["sex"]["customers"].sum() == 4000
    assert "## By sex" in to_markdown(tables, model)


def test_calibration_by_group_flags_real_gaps():
    from credit_default_gbm.fairness import calibration_by_group

    rng = np.random.default_rng(0)
    n = 20000
    group = pd.Series(np.where(rng.random(n) < 0.5, "a", "b"))
    y = (rng.random(n) < 0.2).astype(int)
    p = np.where(group == "a", 0.26, 0.20)  # group a over-predicted by 6 points
    t = calibration_by_group(y, p, group).set_index("group")
    assert t.loc["a", "significant"] and t.loc["a", "gap"] > 0.05
    assert not t.loc["b", "significant"]


def test_cross_fit_scores_every_customer():
    from credit_default_gbm.fairness import drivers, out_of_fold

    raw = make_synthetic(8000, seed=2)
    p, contrib = out_of_fold(raw, folds=3)
    assert len(p) == len(raw) and (p > 0).all() and (p < 1).all()
    assert contrib.shape == (len(raw), len(feature_names()))
    assert len(drivers(contrib, raw["AGE"] < 30, top=3)) == 3
