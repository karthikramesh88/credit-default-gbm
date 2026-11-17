"""Scoring API.

    uvicorn credit_default_gbm.api:app

POST /score with a list of applicants; each gets a calibrated probability of default,
a risk band (A = lowest risk to E) and the top reasons pushing their risk up.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import model_store
from .config import BILL_AMT, PAY_AMT, PAY_STATUS
from .data import DataValidationError

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = model_store.load()
    yield
    STATE.clear()


app = FastAPI(title="Credit-card default scoring", version="0.1.0", lifespan=lifespan)


class Applicant(BaseModel):
    id: str
    LIMIT_BAL: float = Field(gt=0, description="credit limit")
    EDUCATION: int = Field(ge=0, le=6)
    pay_status: list[int] = Field(min_length=6, max_length=6, description="repayment status, most recent month first (-2..8)")
    bill_amt: list[float] = Field(min_length=6, max_length=6, description="statement balances, most recent first")
    pay_amt: list[float] = Field(min_length=6, max_length=6, description="amounts paid, most recent first")


class Score(BaseModel):
    id: str
    probability_of_default: float
    risk_band: str
    reason_codes: list[str]


class ScoreResponse(BaseModel):
    model_version: str
    scores: list[Score]


def _to_frame(applicants: list[Applicant]) -> pd.DataFrame:
    rows = []
    for a in applicants:
        row = {"LIMIT_BAL": a.LIMIT_BAL, "EDUCATION": a.EDUCATION}
        row.update(zip(PAY_STATUS, a.pay_status))
        row.update(zip(BILL_AMT, a.bill_amt))
        row.update(zip(PAY_AMT, a.pay_amt))
        rows.append(row)
    return pd.DataFrame(rows)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    model = STATE["model"]
    return {"status": "ok", "model_version": model.version, "trained_on": model.metadata["data_source"]}


@app.post("/score", response_model=ScoreResponse)
def score(applicants: list[Applicant]) -> ScoreResponse:
    if not applicants:
        raise HTTPException(422, "send at least one applicant")
    model = STATE["model"]
    try:
        result = model.score(_to_frame(applicants))
    except DataValidationError as e:
        raise HTTPException(422, str(e)) from e
    return ScoreResponse(
        model_version=model.version,
        scores=[
            Score(id=a.id, probability_of_default=round(float(r.probability_of_default), 4),
                  risk_band=r.risk_band, reason_codes=r.reason_codes)
            for a, r in zip(applicants, result.itertuples())
        ],
    )
