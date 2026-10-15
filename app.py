import io
import lzma
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from scoring import CLASSES, REQUIRED_COLUMNS, score

BASE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "credit_score_model.pkl.xz"))
TRAIN_CSV = Path(os.getenv("TRAIN_CSV", BASE / "train.csv"))
DATA_CSV = Path(os.getenv("DATA_CSV", BASE / "test.csv"))
MAX_UPLOAD_MB = 30

PROB_COLS = [f"Prob_{c}" for c in CLASSES]
state = {}


def load_bundle(path: Path) -> dict:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as fh:
        return pickle.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = load_bundle(MODEL_PATH)
    state["train"] = pd.read_csv(TRAIN_CSV, low_memory=False)
    state["demo"] = None
    if DATA_CSV.exists():
        state["demo"] = score(state["train"], pd.read_csv(DATA_CSV, low_memory=False), state["bundle"])
    yield


app = FastAPI(title="Credit Score API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def per_customer(rows: pd.DataFrame) -> pd.DataFrame:
    """One bracket per customer: average the monthly probabilities, then take the most likely class."""
    g = rows.groupby("Customer_ID", sort=False)
    cust = g[PROB_COLS].mean().round(4)
    cust["Credit_Score"] = cust[PROB_COLS].idxmax(axis=1).str.replace("Prob_", "", regex=False)
    cust["Months"] = g.size()
    return cust.reset_index()[["Customer_ID", "Credit_Score", *PROB_COLS, "Months"]]


def counts(series: pd.Series) -> dict:
    vc = series.value_counts()
    return {c: int(vc.get(c, 0)) for c in CLASSES}


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": state["bundle"]["model_name"],
        "classes": CLASSES,
        "demo_customers": int(state["demo"]["Customer_ID"].nunique()) if state["demo"] is not None else 0,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception:
        raise HTTPException(400, "Could not read the file as a CSV.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(400, "Missing columns: " + ", ".join(missing))
    if df.empty:
        raise HTTPException(400, "The CSV has no rows.")

    rows = score(state["train"], df, state["bundle"])
    cust = per_customer(rows)
    return {
        "rows": int(len(rows)),
        "customers": int(len(cust)),
        "summary": counts(cust["Credit_Score"]),
        "results": cust.to_dict(orient="records"),
    }


@app.get("/api/customer/{customer_id}")
def customer(customer_id: str):
    demo = state["demo"]
    if demo is None:
        raise HTTPException(404, "No demo data loaded on the server.")
    rows = demo[demo["Customer_ID"].str.lower() == customer_id.strip().lower()]
    if rows.empty:
        raise HTTPException(404, f"Customer '{customer_id}' not found in the demo data.")
    overall = per_customer(rows).iloc[0].to_dict()
    return {"customer": overall, "months": rows.drop(columns=["Customer_ID"]).to_dict(orient="records")}
