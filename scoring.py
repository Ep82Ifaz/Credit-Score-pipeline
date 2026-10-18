"""
scoring.py - turns raw rows (same columns as test.csv) into Poor / Standard / Good predictions.

The cleaning and feature-engineering steps are the notebook's own code (cells 6-12), wrapped in
one function so the API can call it. Like the notebook, it builds each customer's history from
train + new rows together, so it needs the original training data.
"""
import re
import numpy as np
import pandas as pd

CLASSES = ["Poor", "Standard", "Good"]
REQUIRED_COLUMNS = [
    "ID", "Customer_ID", "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
    "Monthly_Balance",
]


def _quiet(*args, **kwargs):      # notebook print() calls are silenced on the server
    pass


def _display(*args, **kwargs):
    pass


def score(train: pd.DataFrame, test: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return one row per input row: ID, Customer_ID, Month, Credit_Score, Prob_Poor/Standard/Good."""
    display = _display
    TARGET = "Credit_Score"
    CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    # ---- notebook cell 6 ----
    def to_number(s):

        return pd.to_numeric(s.astype(str).str.strip().str.replace("_", "", regex=False)
                             .replace({"": np.nan, "nan": np.nan, "None": np.nan}), errors="coerce")

    def parse_history_age(s):

        m = s.astype(str).str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?").astype(float)
        return m[0] * 12 + m[1]

    # ---- notebook cell 7 ----

    train_part, test_part = train.copy(), test.copy()
    train_part["_is_train"], test_part["_is_train"] = 1, 0
    test_part[TARGET] = np.nan
    df = pd.concat([train_part, test_part], ignore_index=True)


    df["Month_Num"] = df["Month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    df["_row_id"] = df["ID"]
    cols_to_drop = ["ID", "Name", "SSN"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    _quiet("Dropped:", cols_to_drop, "->", df.shape)

    # ---- notebook cell 8 ----

    cols_to_clean = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                     "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Num_Credit_Inquiries"]
    for col in cols_to_clean:
        df[col] = to_number(df[col])


    df["Credit_History_Months"] = parse_history_age(df["Credit_History_Age"])
    df = df.drop(columns="Credit_History_Age")

    df.loc[df["Amount_invested_monthly"] >= 10000, "Amount_invested_monthly"] = np.nan
    df.loc[df["Monthly_Balance"].abs() > 1e6, "Monthly_Balance"] = np.nan

    # ---- notebook cell 9 ----

    bounds = {"Age": (14, 100), "Annual_Income": (0, 200_000), "Num_Bank_Accounts": (0, 20),
              "Num_Credit_Card": (0, 20), "Interest_Rate": (0, 40), "Num_of_Loan": (0, 10),
              "Num_of_Delayed_Payment": (0, 30), "Num_Credit_Inquiries": (0, 30),
              "Total_EMI_per_month": (0, 1500)}
    anomaly_rows = []
    for col, (lo, hi) in bounds.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        df.loc[bad, col] = np.nan
        anomaly_rows.append({"Column": col, "Valid range": f"[{lo}, {hi}]", "Anomalies -> NaN": int(bad.sum())})
    display(pd.DataFrame(anomaly_rows))

    # ---- notebook cell 10 ----

    df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)


    df = df.sort_values(["Customer_ID", "Month_Num"]).reset_index(drop=True)

    const_cols = ["Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
                  "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan", "Outstanding_Debt",
                  "Total_EMI_per_month", "Credit_Mix", "Payment_of_Min_Amount", "Credit_History_Months",
                  "Payment_Behaviour"]
    before = int(df[const_cols].isna().sum().sum())
    df[const_cols] = df.groupby("Customer_ID", sort=False)[const_cols].ffill()
    df[const_cols] = df.groupby("Customer_ID", sort=False)[const_cols].bfill()
    _quiet(f"ffill + bfill within customer: {before} -> {int(df[const_cols].isna().sum().sum())} missing cells")

    # ---- notebook cell 11 ----

    var_cols = ["Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
                "Credit_Utilization_Ratio", "Amount_invested_monthly", "Monthly_Balance"]
    for col in var_cols + const_cols:
        if df[col].dtype.kind in "fi" and df[col].isna().any():
            df[col] = df[col].fillna(df.groupby("Customer_ID")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())


    for col in ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["Type_of_Loan"] = df["Type_of_Loan"].fillna("No Loan")


    n_types = df["Type_of_Loan"].map(lambda s: 0 if s == "No Loan" else len(re.split(r",\s*(?:and\s+)?|\s+and\s+", s)))
    df["Num_of_Loan"] = np.maximum(df["Num_of_Loan"], n_types)
    df["Age"] = df["Age"].round().astype(int)

    _quiet("Remaining missing cells (features):", int(df.drop(columns=[TARGET]).isna().sum().sum()))
    clean = df.copy()
    display(clean[clean._is_train == 1].head())

    # ---- notebook cell 12 ----
    def engineer_features(df):

        f = df.copy()

        types = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan",
                 "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"]
        for t in types:
            f["Loan_" + t.replace(" ", "_").replace("-", "_")] = f["Type_of_Loan"].str.contains(t, regex=False).astype(int)
        f["Num_Loan_Types"] = f[[c for c in f.columns if c.startswith("Loan_")]].sum(axis=1)


        eps = 1e-6
        f["Debt_to_Income"] = f["Outstanding_Debt"] / (f["Annual_Income"] + eps)
        f["EMI_to_Salary"] = f["Total_EMI_per_month"] / (f["Monthly_Inhand_Salary"] + eps)
        f["Invest_to_Salary"] = f["Amount_invested_monthly"] / (f["Monthly_Inhand_Salary"] + eps)
        f["Balance_to_Salary"] = f["Monthly_Balance"] / (f["Monthly_Inhand_Salary"] + eps)
        f["Delay_per_Loan"] = f["Num_of_Delayed_Payment"] / (f["Num_of_Loan"] + 1)
        f["Credit_Cards_per_Account"] = f["Num_Credit_Card"] / (f["Num_Bank_Accounts"] + 1)
        f["Log_Annual_Income"] = np.log1p(f["Annual_Income"])

        g = f.groupby("Customer_ID")
        for c in ["Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Utilization_Ratio",
                  "Num_Credit_Inquiries", "Changed_Credit_Limit", "Monthly_Balance", "Amount_invested_monthly"]:
            f[f"Cust_Mean_{c}"] = g[c].transform("mean")
        f["Cust_Std_Delay"] = g["Delay_from_due_date"].transform("std").fillna(0)
        f["Cust_Max_Delay"] = g["Delay_from_due_date"].transform("max")
        f["Cust_Months_Seen"] = g["Month_Num"].transform("count")
        _quiet(f"Created {f.shape[1] - df.shape[1]} new features (loan flags, ratios, customer aggregates).")


        f["Credit_Mix_Ord"] = f["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 2})
        f["PB_Spend_High"] = f["Payment_Behaviour"].str.startswith("High").astype(int)
        f["PB_Value"] = f["Payment_Behaviour"].str.extract(r"(Small|Medium|Large)")[0].map({"Small": 0, "Medium": 1, "Large": 2})
        f["PayMin_Yes"] = (f["Payment_of_Min_Amount"] == "Yes").astype(int)
        f["PayMin_No"] = (f["Payment_of_Min_Amount"] == "No").astype(int)
        return f

    feat = engineer_features(clean)
    tr = feat[feat._is_train == 1].reset_index(drop=True)
    te = feat[feat._is_train == 0].reset_index(drop=True)
    y_all = tr[TARGET].map(CLASS_MAP).values.astype(int)
    _quiet(f"Modelling frame: train={tr.shape}  test={te.shape}")
    display(tr[["Debt_to_Income", "EMI_to_Salary", "Num_Loan_Types", "Cust_Mean_Delay_from_due_date",
                "Credit_Mix_Ord", "PB_Spend_High", "PB_Value", "PayMin_Yes"]].head())

    # ---- scoring with the saved model bundle ----
    te_r = te.reset_index(drop=True)
    X = te_r[bundle["base_cols"]].copy()
    for col in bundle["te_cols"]:
        mp, prior = bundle["te_maps"][col]
        arr = mp.reindex(te_r[col].values).fillna(dict(enumerate(prior))).values
        for k in range(3):
            X[f"TE_{col}_{CLASSES[k]}"] = arr[:, k]
    X = X[bundle["feature_names"]]
    proba = bundle["model"].predict_proba(bundle["scaler"].transform(X))
    pred = proba.argmax(axis=1)

    out = pd.DataFrame({"ID": te_r["_row_id"].values, "Customer_ID": te_r["Customer_ID"].values,
                        "Month": te_r["Month"].values, "Credit_Score": [CLASSES[i] for i in pred]})
    for k, c in enumerate(CLASSES):
        out[f"Prob_{c}"] = proba[:, k].round(4)
    return out
