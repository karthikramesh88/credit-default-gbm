# Credit-card default model (XGBoost)

[![CI](https://github.com/karthikramesh88/credit-default-gbm/actions/workflows/ci.yml/badge.svg)](https://github.com/karthikramesh88/credit-default-gbm/actions/workflows/ci.yml)

Predicts whether a credit-card customer will miss next month's payment, and wraps the model in what production use needs: a release gate, calibrated probabilities, reason codes, a scoring API, and drift monitoring.

This is a personal portfolio project. The goal is hands-on experience in building, deploying and monitoring a gradient-boosted model end to end, not just training one in a notebook.

## What's in it

| Piece | File | What it shows |
|---|---|---|
| Data checks | `data.py` | Training stops on missing columns, duplicate IDs, nulls or a non-binary target, instead of quietly training on bad data. |
| Features | `features.py` | Lateness, utilisation, payment ratios and trends. The same code runs in training and in the API, so they can't drift apart. Sex, marital status and age are kept out of the model. |
| Business rules in the model | `features.py` (`MONOTONE`) | More lateness or utilisation can never *lower* predicted risk, and paying more of the bill can never *raise* it (XGBoost monotone constraints). |
| Baseline and release gate | `train.py` | A logistic-regression baseline is trained alongside. The XGBoost model is approved only if it beats the baseline's AUC by at least 0.005, is no worse calibrated (Brier score), and its default rate rises steadily across score deciles. A model that fails is saved as `rejected` and never served. |
| Calibration | `train.py` | Isotonic calibration on the validation set, so a score of 0.20 means roughly a 20% chance of default. |
| Versioned models | `model_store.py` | Each run saves to `models/<version>/` (model, calibrator, metadata with metrics, gate result and features). `models/LATEST` points at the approved version. |
| Scoring API | `api.py` | FastAPI `POST /score` returns probability of default, a risk band (A to E) and the top three reasons pushing risk up, from XGBoost's built-in TreeSHAP contributions. |
| Fairness audit | `fairness.py` | Calibration, ranking, approval rates, adverse impact ratio and error rates by sex and age band, which the model never uses. |
| Drift monitoring | `monitor.py` | Population stability index (PSI) for every feature and the score against the training population; 0.10 = watch, 0.25 = investigate or retrain. |
| Tests | `tests/` | Feature safety, data checks, gate logic, reload consistency, monotonicity, drift detection and the API. |

## Setup

Needs [uv](https://docs.astral.sh/uv/) (already installed on this Mac).

```bash
make setup
make test
```

On macOS, XGBoost needs the OpenMP runtime. The Makefile points it at the copy bundled with scikit-learn, so Homebrew isn't required. That's why the targets call `.venv/bin/python` directly: `uv run` drops the `DYLD_LIBRARY_PATH` variable.

## Data

**Default of Credit Card Clients** (Taiwan, 2005; 30,000 card holders; about 22% default), from the UCI Machine Learning Repository, dataset 350:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Download the zip, unzip it, and put `default of credit card clients.xls` in `data/raw/`. Data files are git-ignored.

## Run

```bash
make train-synthetic   # dry run on synthetic data, no download needed
make train             # real data
make serve             # API at http://127.0.0.1:8000, interactive docs at /docs
make monitor-sample    # drift check on a batch with inflated balances
make fairness          # fairness audit of the approved model
make fairness-crossfit # calibration gaps on all customers via cross-fitting, with drivers
```

### Results on the UCI data

Trained 17 Nov 2026 on 18,000 card holders, with early stopping and calibration on 6,000 and a held-out test set of 6,000 (22% default rate). Approved by the release gate; 231 trees.

| | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| XGBoost | 0.789 | 0.577 | 0.447 | 0.132 |
| Logistic baseline | 0.757 | 0.513 | 0.419 | 0.139 |

Default rate by score decile, lowest to highest risk:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5.0% | 5.5% | 8.8% | 10.8% | 14.7% | 15.8% | 18.8% | 26.3% | 43.8% | 71.5% |
| Baseline | 6.5% | 9.5% | 11.5% | 14.2% | 11.0% | 15.8% | 16.8% | 24.7% | 44.2% | 67.0% |

XGBoost ranks risk steadily across all ten deciles. The baseline's fifth decile is out of order. The highest-risk tenth defaults about 14 times as often as the lowest-risk tenth.

### Fairness audit

`make fairness` scores the held-out test set and compares groups by sex and age band (attributes the model never sees), under a policy of declining band E, the riskiest fifth (probability of default above 37.5%). Full tables are written to `reports/`.

| Group | Customers | Actual default | Predicted default | AUC | Approved | Good customers declined | Adverse impact ratio |
|---|---|---|---|---|---|---|---|
| Female | 3,628 | 20.8% | 21.8% | 0.792 | 80.0% | 10.8% | 1.00 |
| Male | 2,372 | 24.1% | 23.0% | 0.779 | 77.2% | 13.3% | 0.97 |
| Age 21–29 | 1,926 | 21.9% | 23.6% | 0.787 | 76.4% | 13.8% | 0.93 |
| Age 30–39 | 2,206 | 19.9% | 20.3% | 0.790 | 82.4% | 9.8% | 1.00 |
| Age 40–49 | 1,296 | 24.5% | 22.7% | 0.791 | 77.9% | 11.5% | 0.95 |
| Age 50+ | 572 | 26.0% | 24.6% | 0.767 | 76.4% | 13.2% | 0.93 |

**Findings**

- **No group crosses the four-fifths line.** The lowest adverse impact ratio is 0.93, well above the 0.8 red flag, and the model ranks risk about equally well in every group (AUC 0.77–0.79).
- **Calibration holds within ±2 points for every group** (flag threshold: 3 points).
- **Follow-up: is there a systematic tilt?** A single test set is small, so `make fairness-crossfit` scores all 30,000 customers with 5-fold cross-fitting (each customer is scored by a model that never saw them), and the audit was also repeated across 20 retrains on different random splits.

  | Group | Predicted minus actual default rate | z-score | Same direction in 20 retrains |
  |---|---|---|---|
  | Female | +0.6 points | 2.1 | 17 of 20 |
  | Male | −0.9 points | −2.2 | 19 of 20 |
  | Age 21–29 | +0.9 points | 2.0 | 19 of 20 |
  | Age 30–39 | +0.3 points | 0.8 | 11 of 20 |
  | Age 40–49 | −1.1 points | −2.2 | 17 of 20 |
  | Age 50+ | −1.1 points | −1.3 | 13 of 20 |

  The first test split overstated the under-30 gap (1.8 points; the true figure is about 0.9). But there is a small, consistent tilt: the model slightly **over-predicts risk for women and under-30s** and slightly **under-predicts it for men and people in their 40s**. Every gap is under 1.2 points, well inside the 3-point flag, and no adverse impact ratio comes near 0.8.

- **What drives it: the credit limit.** In the TreeSHAP contributions, credit limit is the largest single source of the difference for every group except the over-50s; for under-30s it contributes two to three times as much as the next feature. Under-30s hold lower limits (NT$124k on average vs NT$188k), yet at the same limit they default *less* often than older customers. In the lowest limit quartile, for example, 30.9% of under-30s default against 32.5% of older customers. So the model partly reads a low limit as higher risk when, for young customers, it often just reflects a short credit history. The limit is acting partly as a stand-in for age.

- **What I'd do in production.** Not adjust predictions by age or sex, since using protected attributes in decisions is generally not allowed. Instead: (1) record this in the model documentation as a known, monitored effect; (2) add `make fairness-crossfit` to every retraining run, and escalate if any gap passes 3 points or a ratio drops below 0.8; (3) with richer data, add credit-history length, so the limit stops standing in for it.

### Dry run on synthetic data

On 30,000 synthetic rows (held-out test set):

| | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| XGBoost | 0.846 | 0.692 | 0.576 | 0.074 |
| Logistic baseline | 0.814 | 0.628 | 0.533 | 0.083 |

The synthetic data is built with thresholds, an interaction and a U-shape, so a straight-line model can't capture it fully. On linear synthetic data, the gate correctly *rejected* XGBoost because it didn't beat the baseline.

## Deployment

- **Container.** The `Dockerfile` downloads the UCI data (checksum-pinned), trains the model during the build, and fails the build unless the release gate approves it. A running container therefore always serves an approved model, and no data or model files live in git.
- **CI.** On every push, GitHub Actions runs the tests, builds the container, starts it, and checks that `/score` ranks the risky example applicant above the safe one.
- **Hosting and continuous deployment.** After the tests and container check pass on `main`, CI pushes the code to a Hugging Face Space (Docker, free CPU), which rebuilds and serves it. A failing test or a gate rejection never reaches the live API.

Try it:

```bash
curl -X POST <service-url>/score -H 'content-type: application/json' -d @examples/applicants.json
```

Interactive API docs are at `<service-url>/docs`.

## Design notes

- **Ranking vs calibration metrics.** AUC, Gini, KS and decile ordering use the raw model score. Brier and log loss use the calibrated probability. Isotonic calibration rounds scores into flat steps, and ranking on those adds artificial ties. The mapping is monotone, so the raw score keeps the same order without them.
- **Noise-aware decile check.** A decile counts as out of order only if its default rate drops below the previous decile's by more than two standard errors, so a small test set doesn't fail the gate by chance.
- **Split.** The UCI data has no application date, so the 60/20/20 split is random and stratified. With real lending data, split by time (train on older months, test on newer) so the test mirrors how the model meets future customers.
- **Protected attributes.** Kept in the raw data for the fairness audit, never used as features.
- **Bounded probabilities.** Calibrated probabilities are kept between 0.1% and 99.9%, so no customer is ever scored as certain to pay or certain to default. A test caught isotonic calibration returning exactly 0.

## Roadmap

1. Dockerfile, and deploy the API to a free cloud tier.
2. Scheduled retraining that runs the gate and the fairness checks, and promotes only approved models; experiment tracking with MLflow.
3. A larger dataset with dates (e.g. Home Credit Default Risk on Kaggle) and a time-based split.
4. A LightGBM comparison.
