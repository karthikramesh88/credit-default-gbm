"""Turn raw card-account fields into model features. Used identically in training and serving."""

import numpy as np
import pandas as pd

from .config import BILL_AMT, PAY_AMT, PAY_STATUS, PROTECTED

# Plain-language meaning of each feature, used for reason codes in the API.
DESCRIPTIONS = {
    "LIMIT_BAL": "credit limit",
    "EDUCATION": "education level",
    "PAY_1": "repayment status last month",
    **{f"PAY_{m}": f"repayment status {m} months ago" for m in range(2, 7)},
    "months_late": "number of months paid late",
    "worst_delay": "longest payment delay",
    "util_1": "card utilisation last month",
    "util_avg": "average card utilisation over six months",
    "util_trend": "rise in balance over six months",
    "pay_ratio_1": "share of last bill paid",
    "pay_ratio_avg": "average share of bills paid",
    "zero_pay_months": "months with no payment on a balance",
}

# Business logic the model must respect: more lateness or utilisation can never
# lower predicted risk; paying more of the bill can never raise it.
MONOTONE = {
    "months_late": 1,
    "worst_delay": 1,
    "util_avg": 1,
    "pay_ratio_avg": -1,
    "zero_pay_months": 1,
}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    limit = raw["LIMIT_BAL"].astype(float)
    pay = raw[PAY_STATUS].astype(float)
    bills = raw[BILL_AMT].astype(float)
    paid = raw[PAY_AMT].astype(float)

    f = pd.DataFrame(index=raw.index)
    f["LIMIT_BAL"] = limit
    # Source codes 0, 5 and 6 are undocumented; fold them into 4 ("other").
    f["EDUCATION"] = raw["EDUCATION"].where(raw["EDUCATION"].isin([1, 2, 3]), 4)
    for col in PAY_STATUS:
        f[col] = pay[col]
    f["months_late"] = (pay > 0).sum(axis=1)
    f["worst_delay"] = pay.max(axis=1).clip(lower=0)

    util = bills.div(limit, axis=0).clip(-1, 3)
    f["util_1"] = util["BILL_AMT1"]
    f["util_avg"] = util.mean(axis=1)
    f["util_trend"] = ((bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit).clip(-3, 3)

    # PAY_AMTm is paid against the previous month's statement, BILL_AMT(m+1).
    ratios = []
    for m in range(1, 6):
        bill = bills[f"BILL_AMT{m + 1}"]
        ratio = np.where(bill > 0, paid[f"PAY_AMT{m}"] / bill.where(bill > 0, 1), 1.0)
        ratios.append(np.clip(ratio, 0, 2))
    ratios = np.column_stack(ratios)
    f["pay_ratio_1"] = ratios[:, 0]
    f["pay_ratio_avg"] = ratios.mean(axis=1)
    owed = bills[[f"BILL_AMT{m + 1}" for m in range(1, 6)]].to_numpy() > 0
    f["zero_pay_months"] = ((paid[[f"PAY_AMT{m}" for m in range(1, 6)]].to_numpy() == 0) & owed).sum(axis=1)

    assert not set(PROTECTED) & set(f.columns), "protected attribute leaked into features"
    return f.astype(float)


def feature_names() -> list[str]:
    return list(DESCRIPTIONS)
