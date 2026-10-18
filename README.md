# Credit Score API (FastAPI)

Predicts **Poor / Standard / Good** for a customer using the model from the fixed pipeline notebook.

## Files (keep them in one folder)

    main.py                  the API
    scoring.py               the notebook's cleaning + feature code, wrapped in score()
    requirements.txt
    credit_score_model.pkl   saved by the fixed notebook (or credit_score_model.pkl.xz)
    train.csv                the original training data (the model uses customer-history features)

## Run

    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Open **http://localhost:8000/docs** for the Swagger page. Startup takes a few seconds.
Each request takes about 2-3 seconds, because the pipeline re-builds customer history from train.csv.

## Endpoints

| Method | Path             | What it does                                   |
|--------|------------------|------------------------------------------------|
| GET    | `/health`        | model name and classes                         |
| POST   | `/predict`       | one customer record -> score                   |
| POST   | `/predict/batch` | `{"records": [...]}` (up to 5000) -> one score per record |

### POST /predict

Request body (this is also the example shown in Swagger):

    {
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
      "Type_of_Loan": "Auto Loan, Credit-Builder Loan, Personal Loan, Home Equity Loan"
    }

Response:

    {
      "credit_score": "Good",
      "class_id": 2,
      "probabilities": {"Poor": 0.091, "Standard": 0.2282, "Good": 0.6809}
    }

`class_id`: 0 = Poor, 1 = Standard, 2 = Good.

curl:

    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @record.json

Python:

    import requests
    r = requests.post("http://localhost:8000/predict", json=record)
    print(r.json())

### POST /predict/batch

    {"records": [ {...record...}, {...record...} ]}

Response:

    {"count": 2, "predictions": [
       {"index": 0, "customer_id": "API_ROW_0", "credit_score": "Good", "class_id": 2,
        "probabilities": {"Poor": 0.09, "Standard": 0.23, "Good": 0.68}},
       ...]}

## Important notes

- **History matters.** The model also uses each customer's behaviour over several months. A record sent
  without `Customer_ID` is treated as a brand-new customer with one month, which is less reliable.
  For best results send every month of a customer in `/predict/batch` with the same `Customer_ID`
  (for customers that are in train.csv, their history is included automatically).
- **Optional fields.** Age, Occupation, Monthly_Inhand_Salary, Type_of_Loan, Num_of_Delayed_Payment,
  Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Credit_History_Age, Payment_of_Min_Amount,
  Total_EMI_per_month, Amount_invested_monthly, Payment_Behaviour and Monthly_Balance may be left out or
  sent as `null`; the pipeline fills them in the same way it does for missing values in the data.
- **Validation.** Impossible values (for example Age 500 or a negative Num_of_Loan) are rejected with
  HTTP 422 and a message naming the field. In a batch, the error shows which record index failed.
  If you feed rows straight from the raw CSV, junk values like these must be cleaned or sent as null first.
- Use scikit-learn 1.9.1, the version the model was saved with.
- Settings (optional environment variables): `MODEL_PATH`, `TRAIN_CSV`.
