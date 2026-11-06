# Twitter Sentiment Analysis - Notebook + Flask API

Trains sentiment models (Naive Bayes, Logistic Regression, Linear SVM) on `twitter_training.csv`,
saves the best one, and serves it through a Flask API you can call from Postman.

Classes: **Positive, Negative, Neutral, Irrelevant**

## Files

| File | Purpose |
|---|---|
| `Twitter_Sentiment_Flask_API.ipynb` | Full pipeline: cleaning, EDA, TF-IDF, training, SMOTE comparison, word clouds, saving the model, starting the API |
| `app.py` | Standalone Flask API (also written by the notebook) |
| `Twitter_Sentiment_API.postman_collection.json` | Ready-made Postman requests |
| `requirements.txt` | Python dependencies |
| `twitter_training.csv` | Your dataset (not included - place it next to the notebook) |
| `best_model.pkl`, `tfidf_vectorizer.pkl` | Created by the notebook; `app.py` needs both |

## 1. Setup

Python 3.9+ is required.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

NLTK data (`punkt_tab`, `stopwords`) is downloaded automatically on first run.
If your network blocks it, run once: `python -m nltk.downloader punkt_tab stopwords`

## 2. Train and save the model

1. Put `twitter_training.csv` in the same folder as the notebook.
2. Open `Twitter_Sentiment_Flask_API.ipynb` and choose **Run All**.
3. The save cell prints the file names and full paths of `best_model.pkl` and `tfidf_vectorizer.pkl`.
4. The last cells write `app.py`, start the server on port 5000, and run a quick self-test.

Charts and word clouds are saved in the `wordclouds/` folder.

## 3. Run the API without the notebook

Keep `app.py`, `best_model.pkl` and `tfidf_vectorizer.pkl` in one folder:

```bash
python app.py
```

Optional environment variables: `PORT` (default 5000), `HOST` (default 0.0.0.0).
Windows example: `set PORT=5001 && python app.py`

If the `.pkl` files are missing, the server stops at startup with a clear message.

## 4. Use it from Postman

**Quick way:** Postman -> Import -> choose `Twitter_Sentiment_API.postman_collection.json`.
The collection uses a variable `base_url` (default `http://127.0.0.1:5000`).

**Manual way:** for POST requests set Body -> raw -> JSON.

| Method | URL | Body |
|---|---|---|
| GET | `http://127.0.0.1:5000/health` | none |
| POST | `http://127.0.0.1:5000/predict` | `{"text": "I love this game"}` |
| POST | `http://127.0.0.1:5000/predict/batch` | `{"texts": ["I love it", "worst update ever"]}` |

Form-data also works: key `text` for `/predict`, repeated key `texts` for `/predict/batch`.

### Example response - `POST /predict`

```json
{
  "text": "I absolutely love this game, it is amazing!",
  "cleaned_text": "absolutely love game amazing",
  "sentiment": "Positive",
  "confidence": 0.9606,
  "confidence_type": "probability",
  "scores": {"Irrelevant": 0.0095, "Negative": 0.0126, "Neutral": 0.0173, "Positive": 0.9606}
}
```

(Example values come from a synthetic test; yours will differ.)

If the best model is the Linear SVM, it has no real probabilities, so `confidence` is a softmax of its
decision scores and `confidence_type` says so.

### Errors

| Status | Cause |
|---|---|
| 400 | Missing or empty `text` / `texts`, or an item that is not a non-empty string |
| 404 | Unknown URL |
| 405 | Wrong method (e.g. GET on `/predict`) |
| 413 | More than 500 texts in one batch |

## Troubleshooting

- **Port 5000 in use** (macOS AirPlay Receiver uses it): set `PORT=5001`, or change `PORT` in the notebook's server cell.
- **Postman can't connect from Colab:** Colab's server isn't on your computer. Download `app.py` and both `.pkl` files, then run `python app.py` locally.
- **Model fails to load:** use the same scikit-learn version that created the `.pkl` files, or re-run the notebook.
- **`only integer scalar arrays...` error:** make sure you are using the latest notebook; the train/test split already uses index-based splitting.
- **Server log:** when started from the notebook, output goes to `flask_server.log`.
