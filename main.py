"""
Credit Score API  -  Poor / Standard / Good

Run:      uvicorn main:app --reload --port 8000
Swagger:  http://localhost:8000/docs

Endpoints
  GET  /health          model info
  POST /predict         one customer record (JSON)  ->  credit_score, class_id, probabilities
  POST /predict/batch   many records (JSON)         ->  one prediction per record

The model comes from the "Credit Score Pipeline (fixed)" notebook (credit_score_model.pkl), and the
cleaning / feature steps in scoring.py are that notebook's own code.

Settings (optional environment variables)
  MODEL_PATH   path to the model file (.pkl or .pkl.xz)
  TRAIN_CSV    path to train.csv (default: train.csv next to this file)
"""
import lzma
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from scoring import CLASSES, score

BASE = Path(__file__).parent
TRAIN_CSV = Path(os.getenv("TRAIN_CSV", BASE / "train.csv"))
MAX_BATCH = 5000
state = {}


def find_model() -> Path:
    if os.getenv("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    for name in ("credit_score_model.pkl", "credit_score_model.pkl.xz"):
        if (BASE / name).exists():
            return BASE / name
    raise FileNotFoundError("credit_score_model.pkl not found next to main.py (or set MODEL_PATH)")


def load_bundle(path: Path) -> dict:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as fh:
        return pickle.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = load_bundle(find_model())
    state["train"] = pd.read_csv(TRAIN_CSV, low_memory=False)   # customer-history features need it
    yield


app = FastAPI(
    title="Credit Score API",
    version="1.0.0",
    description="Classifies a customer's credit score as **Poor**, **Standard** or **Good**.",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------------
MonthName = Literal["January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"]

EXAMPLE = {
    "Age": 28,
    "Amount_invested_monthly": 80.42,
    "Annual_Income": 19114.12,
    "Changed_Credit_Limit": 11.27,
    "Credit_History_Age": "22 Years and 9 Months",
    "Credit_Mix": "Good",
    "Credit_Utilization_Ratio": 26.82,
    "Delay_from_due_date": 3,
    "Interest_Rate": 3,
    "Month": "January",
    "Monthly_Balance": 312.49,
    "Monthly_Inhand_Salary": 1824.84,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Num_Credit_Inquiries": 4,
    "Num_of_Delayed_Payment": 7,
    "Num_of_Loan": 4,
    "Occupation": "Scientist",
    "Outstanding_Debt": 809.98,
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 49.57,
    "Type_of_Loan": "Auto Loan, Credit-Builder Loan, Personal Loan, Home Equity Loan",
}


class CustomerRecord(BaseModel):
    """One customer-month, with the same fields as a row of test.csv (minus ID, Name and SSN).

    Fields marked optional can be left out or sent as null when the value is unknown; the pipeline then
    fills them in (from the same customer's other months, or the overall typical value), exactly as it
    does for missing values in the training data.
    """
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})

    Customer_ID: Optional[str] = Field(
        None, description="Optional. Records that share a Customer_ID are treated as months of the same "
                          "customer (and joined with that customer's training history, if any).")
    Month: MonthName

    # ---- required: columns that are always filled in the data
    Annual_Income: float = Field(..., ge=0, le=200_000)
    Num_Bank_Accounts: int = Field(..., ge=0, le=20)
    Num_Credit_Card: int = Field(..., ge=0, le=20)
    Interest_Rate: float = Field(..., ge=0, le=40)
    Num_of_Loan: int = Field(..., ge=0, le=10)
    Delay_from_due_date: int
    Outstanding_Debt: float = Field(..., ge=0)
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100)

    # ---- optional: often missing in real data, so the pipeline can fill them in
    Age: Optional[int] = Field(None, ge=14, le=100)
    Occupation: Optional[str] = None
    Monthly_Inhand_Salary: Optional[float] = Field(None, ge=0)
    Type_of_Loan: Optional[str] = Field(None, description="Comma-separated loan types; empty means none.")
    Num_of_Delayed_Payment: Optional[int] = Field(None, ge=0, le=30)
    Changed_Credit_Limit: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = Field(None, ge=0, le=30)
    Credit_Mix: Optional[Literal["Bad", "Standard", "Good"]] = None
    Credit_History_Age: Optional[str] = Field(None, pattern=r"^\d+ Years? and \d+ Months?$",
                                              description='Format: "22 Years and 9 Months"')
    Payment_of_Min_Amount: Optional[Literal["Yes", "No", "NM"]] = None
    Total_EMI_per_month: Optional[float] = Field(None, ge=0, le=1500)
    Amount_invested_monthly: Optional[float] = Field(None, ge=0)
    Payment_Behaviour: Optional[str] = None
    Monthly_Balance: Optional[float] = None


class Prediction(BaseModel):
    credit_score: str
    class_id: int
    probabilities: dict[str, float]


class BatchPrediction(Prediction):
    index: int
    customer_id: str


class BatchRequest(BaseModel):
    records: list[CustomerRecord] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    count: int
    predictions: list[BatchPrediction]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
NUMERIC_FIELDS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
]


def run_model(records: list[CustomerRecord]) -> list[BatchPrediction]:
    rows = []
    for i, rec in enumerate(records):
        d = rec.model_dump()
        d["Customer_ID"] = d["Customer_ID"] or f"API_ROW_{i}"     # no ID -> independent new customer
        d["ID"] = f"api_{i}"
        rows.append(d)
    df = pd.DataFrame(rows)
    for col in NUMERIC_FIELDS:                                   # None -> NaN so the pipeline can impute
        df[col] = pd.to_numeric(df[col], errors="coerce")
    try:
        out = score(state["train"], df, state["bundle"]).set_index("ID")
    except Exception as exc:                                    # pragma: no cover
        raise HTTPException(500, f"Scoring failed: {exc}")

    preds = []
    for i, row in enumerate(rows):
        r = out.loc[row["ID"]]
        probs = {c: float(r[f"Prob_{c}"]) for c in CLASSES}
        preds.append(BatchPrediction(
            index=i, customer_id=row["Customer_ID"], credit_score=r["Credit_Score"],
            class_id=CLASSES.index(r["Credit_Score"]), probabilities=probs))
    return preds


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": state["bundle"]["model_name"], "classes": CLASSES}


@app.post("/predict", response_model=Prediction, summary="Predict")
def predict(record: CustomerRecord):
    """Score one customer record.

    The model also uses a customer's behaviour over several months. A record sent on its own is treated as
    a new customer with a single month, so it is less reliable than a full history. Send `null` (or leave out) any optional field whose value is unknown. For better results send
    the same `Customer_ID` on every month through `/predict/batch`.
    """
    p = run_model([record])[0]
    return Prediction(credit_score=p.credit_score, class_id=p.class_id, probabilities=p.probabilities)


@app.post("/predict/batch", response_model=BatchResponse, summary="Predict Batch")
def predict_batch(batch: BatchRequest):
    """Score many records at once (up to 5000). Records with the same `Customer_ID` are months of one customer."""
    preds = run_model(batch.records)
    return BatchResponse(count=len(preds), predictions=preds)
