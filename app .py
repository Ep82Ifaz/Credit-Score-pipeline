"""Flask API + web dashboard for the Twitter sentiment model (Postman or browser).

Run:   python app.py          (needs best_model.pkl + tfidf_vectorizer.pkl in this same folder)
Env:   HOST (default 0.0.0.0)  PORT (default 5000)
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

import nltk
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
DASHBOARD_DATA_PATH = BASE_DIR / "dashboard_data.json"
IMAGE_DIR = BASE_DIR / "wordclouds"
MAX_BATCH = 500


def ensure_nltk():
    """Make sure the NLTK data used by the notebook's clean_text() is available."""
    for resource, path in (("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource '{resource}' ...")
            nltk.download(resource, quiet=True)


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(
            f"'{path.name}' not found in {path.parent}. Run the notebook's save-model cell first "
            f"and keep the .pkl files in the same folder as app.py.")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load '{path.name}': {e}. Use the same scikit-learn version "
                           f"that created it.") from e


ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))

try:
    model = load_pickle(MODEL_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)
except Exception as err:
    print(f"[STARTUP ERROR] {err}", file=sys.stderr)
    sys.exit(1)


def clean_text(text):
    """Same cleaning as the notebook (lowercase, drop <unk>, URLs, symbols, stopwords)."""
    text = str(text).lower()
    text = text.replace("<unk>", "")
    text = re.sub(r"http\S+|www\S+|pic\.twitter\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in STOP_WORDS)


def predict_texts(texts):
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    labels = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs, kind = model.predict_proba(X), "probability"
    else:  # LinearSVC has no predict_proba -> softmax over decision scores
        d = model.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        probs, kind = e / e.sum(axis=1, keepdims=True), "softmax of SVM decision scores (not a calibrated probability)"
    results = []
    for text, clean, label, p in zip(texts, cleaned, labels, probs):
        item = {
            "text": text,
            "cleaned_text": clean,
            "sentiment": str(label),
            "confidence": round(float(p.max()), 4),
            "scores": {str(c): round(float(v), 4) for c, v in zip(model.classes_, p)},
        }
        if not clean.strip():
            item["warning"] = "Nothing left after cleaning (only URLs/symbols/stopwords) - prediction is unreliable."
        results.append(item)
    return results, kind

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment Dashboard</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--text:#1d2330;--muted:#667085;--line:#e4e7ec;--accent:#3b6fd4;
--pos:#2a9d5c;--neg:#d1495b;--neu:#7b6fd0;--irr:#8a8f98}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--card:#181e29;--text:#e8ecf3;--muted:#98a2b3;--line:#283042;--accent:#6c9bff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:22px 16px 6px;max-width:1100px;margin:0 auto}
h1{margin:0;font-size:24px}
.sub{color:var(--muted);margin-top:4px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;background:var(--accent);color:#fff;font-size:13px;margin-left:6px}
main{max-width:1100px;margin:0 auto;padding:12px 16px 40px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.card h2{margin:0 0 12px;font-size:16px}
.grid{display:grid;gap:16px}
.g4{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.kpi .v{font-size:28px;font-weight:650}
.kpi .l{color:var(--muted);font-size:13px}
textarea{width:100%;min-height:84px;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--text);font:inherit;resize:vertical}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;cursor:pointer}
button.ghost{background:transparent;color:var(--accent);border:1px solid var(--line);padding:5px 10px;font-size:13px}
button:disabled{opacity:.6;cursor:wait}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.bar{display:grid;grid-template-columns:90px 1fr 120px;gap:8px;align-items:center;margin:6px 0}
.track{background:var(--bg);border-radius:6px;height:14px;overflow:hidden}
.fill{height:100%;border-radius:6px}
.res-label{font-size:22px;font-weight:650}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.cm td,.cm th{text-align:center}
.clouds img{width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
.clouds figcaption{text-align:center;color:var(--muted);font-size:13px;margin-top:4px}
figure{margin:0}
.err{color:var(--neg)}
.note{color:var(--muted)}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<header>
  <h1>Twitter Sentiment Dashboard <span class="badge" id="model-badge">loading...</span></h1>
  <div class="sub" id="sub">Flask API &middot; try the model, then see how it performed on the test set.</div>
</header>
<main>
  <section class="card">
    <h2>Try the model</h2>
    <textarea id="text" placeholder="Type or paste a tweet..."></textarea>
    <div class="row">
      <button id="go">Predict sentiment</button>
      <span class="note">Examples:</span>
      <button class="ghost" data-ex="I absolutely love this game, it is amazing!">Positive</button>
      <button class="ghost" data-ex="This update is terrible and the servers keep crashing">Negative</button>
      <button class="ghost" data-ex="Just downloaded the new patch">Neutral</button>
    </div>
    <div id="result" style="margin-top:14px"></div>
  </section>

  <p id="stats-missing" class="card note" hidden>Performance charts are not available yet: run the notebook's "dashboard data" cell to create <code>dashboard_data.json</code>, then refresh this page.</p>

  <section class="grid g4" id="kpis"></section>

  <section class="grid g2">
    <div class="card"><h2>Class distribution (all tweets)</h2><div id="dist"></div></div>
    <div class="card"><h2>Per-class performance (test set)</h2><div class="scroll"><table id="perclass"></table></div></div>
  </section>

  <section class="grid g2">
    <div class="card"><h2>Confusion matrix <span class="note">(rows = actual, columns = predicted)</span></h2><div class="scroll"><table class="cm" id="cm"></table></div></div>
    <div class="card"><h2>Model comparison (weighted F1)</h2><div class="scroll"><table id="board"></table></div></div>
  </section>

  <section class="card">
    <h2>Word clouds</h2>
    <div class="grid g2 clouds" id="clouds"></div>
    <p class="note" id="clouds-missing" hidden>No word cloud images found in the <code>wordclouds</code> folder. Run the notebook's word cloud cell first.</p>
  </section>
</main>
<script>
const COLORS = {Positive:"#2a9d5c", Negative:"#d1495b", Neutral:"#7b6fd0", Irrelevant:"#8a8f98"};
const $ = id => document.getElementById(id);
function el(tag, cls, text){ const e = document.createElement(tag); if(cls) e.className = cls; if(text !== undefined) e.textContent = text; return e; }
function pct(x){ return (x*100).toFixed(1) + "%"; }
function bar(label, value, color, shown){
  const row = el("div","bar"); row.append(el("span",null,label));
  const t = el("div","track"), f = el("div","fill"); f.style.width = Math.max(0,Math.min(100,value*100)) + "%"; f.style.background = color; t.append(f);
  row.append(t, el("span",null,shown)); return row;
}
function fillTable(table, head, rows){
  table.replaceChildren();
  const thead = el("thead"), hr = el("tr"); head.forEach(h => hr.append(el("th",null,h))); thead.append(hr); table.append(thead);
  const tb = el("tbody"); rows.forEach(r => { const tr = el("tr"); r.forEach(c => tr.append(c instanceof Node ? c : el("td",null,String(c)))); tb.append(tr); }); table.append(tb);
}

async function predict(){
  const text = $("text").value.trim(), out = $("result");
  out.replaceChildren();
  if(!text){ out.append(el("div","err","Please type a tweet first.")); return; }
  $("go").disabled = true;
  try{
    const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text})});
    const d = await r.json();
    if(!r.ok){ out.append(el("div","err",d.error || "Request failed")); return; }
    const head = el("div","res-label", d.sentiment); head.style.color = COLORS[d.sentiment] || "inherit";
    out.append(head, el("div","note","Confidence " + pct(d.confidence) + " (" + d.confidence_type.split(" (")[0] + ")"));
    Object.entries(d.scores).sort((a,b)=>b[1]-a[1]).forEach(([k,v]) => out.append(bar(k, v, COLORS[k]||"#888", pct(v))));
    if(d.warning) out.append(el("div","err",d.warning));
  }catch(e){ out.append(el("div","err","Could not reach the API: " + e.message)); }
  finally{ $("go").disabled = false; }
}
$("go").addEventListener("click", predict);
document.querySelectorAll("[data-ex]").forEach(b => b.addEventListener("click", () => { $("text").value = b.dataset.ex; predict(); }));

function render(d){
  $("model-badge").textContent = d.model_name;
  $("sub").textContent = "Trained on " + d.n_train.toLocaleString() + " tweets, evaluated on " + d.n_test.toLocaleString() + " held-out tweets" + (d.generated_at ? " (" + d.generated_at + ")" : "") + ".";
  const k = $("kpis"); k.replaceChildren();
  [["Accuracy", d.metrics.accuracy],["Weighted F1", d.metrics.f1_weighted],["Weighted precision", d.metrics.precision_weighted],["Weighted recall", d.metrics.recall_weighted],["Macro F1", d.metrics.f1_macro]]
    .forEach(([l,v]) => { const c = el("div","card kpi"); c.append(el("div","v",pct(v)), el("div","l",l)); k.append(c); });

  const dist = $("dist"); dist.replaceChildren();
  const total = Object.values(d.class_distribution).reduce((a,b)=>a+b,0);
  d.labels.forEach(l => { const n = d.class_distribution[l] || 0; dist.append(bar(l, n/total, COLORS[l], n.toLocaleString() + " (" + pct(n/total) + ")")); });

  fillTable($("perclass"), ["Class","Precision","Recall","F1","Support"],
    d.per_class.map(r => [r.label, pct(r.precision), pct(r.recall), r.f1.toFixed(3), r.support]));

  const cm = $("cm"); cm.replaceChildren();
  const head = el("tr"); head.append(el("th",null,""));
  d.labels.forEach(l => head.append(el("th",null,l))); cm.append(head);
  d.confusion_matrix.forEach((row,i) => {
    const tr = el("tr"); tr.append(el("th",null,d.labels[i]));
    const sum = row.reduce((a,b)=>a+b,0) || 1;
    row.forEach((v,j) => { const td = el("td",null,String(v)); td.style.background = "rgba(59,111,212," + (0.08 + 0.7*v/sum).toFixed(2) + ")"; if(i===j) td.style.fontWeight = "700"; tr.append(td); });
    cm.append(tr);
  });

  fillTable($("board"), ["Model","Balancing","Accuracy","F1"],
    d.leaderboard.map(r => [r.Model, r.Balancing, pct(r.Accuracy), pct(r["F1-Score"])]));
}

async function loadStats(){
  try{
    const r = await fetch("/api/dashboard-data");
    if(!r.ok){ $("stats-missing").hidden = false; $("model-badge").textContent = "model ready"; return; }
    const d = await r.json(); render(d);
    const box = $("clouds"); let n = 0;
    (d.wordclouds || []).forEach(f => {
      const fig = el("figure"), img = el("img"); img.src = "/assets/" + encodeURIComponent(f); img.alt = f;
      const name = f.replace("wordcloud_","").replace(".png","");
      fig.append(img, el("figcaption",null,name.charAt(0).toUpperCase() + name.slice(1) + " word cloud")); box.append(fig); n++;
    });
    if(!n) $("clouds-missing").hidden = false;
  }catch(e){ $("stats-missing").hidden = false; }
}
loadStats();
</script>
</body>
</html>"""


app = Flask(__name__)


def error(message, status=400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return jsonify({
        "service": "Twitter Sentiment API (Flask)",
        "endpoints": {
            "GET /dashboard": "web dashboard (open in a browser)",
            "GET /health": "service + model status",
            "POST /predict": 'body: {"text": "your tweet"}',
            "POST /predict/batch": 'body: {"texts": ["tweet 1", "tweet 2"]}',
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__,
                    "classes": [str(c) for c in model.classes_],
                    "vocabulary_size": len(vectorizer.vocabulary_)})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form.to_dict() or request.args.to_dict()      # form-data / x-www-form-urlencoded / ?text=
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return error('Send a non-empty "text" field, e.g. {"text": "I love this game"} '
                     '(Postman: Body -> raw -> JSON).')
    results, kind = predict_texts([text])
    return jsonify({**results[0], "confidence_type": kind})


@app.post("/predict/batch")
def predict_batch():
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        texts = data.get("texts")
    else:
        texts = request.form.getlist("texts") or None                # form-data: repeat the key "texts"
    if not isinstance(texts, list) or not texts:
        return error('Send a non-empty "texts" list, e.g. {"texts": ["I love it", "worst update ever"]}.')
    if len(texts) > MAX_BATCH:
        return error(f"Too many texts ({len(texts)}). Maximum per request is {MAX_BATCH}.", 413)
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return error('Every item in "texts" must be a non-empty string.')
    results, kind = predict_texts(texts)
    return jsonify({"count": len(results), "confidence_type": kind, "predictions": results})

@app.get("/dashboard")
def dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


@app.get("/api/dashboard-data")
def dashboard_data():
    if not DASHBOARD_DATA_PATH.exists():
        return error("dashboard_data.json not found. Run the notebook's 'dashboard data' cell first.", 404)
    try:
        with open(DASHBOARD_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return error(f"Could not read dashboard_data.json: {e}", 500)
    data["wordclouds"] = [n for n in data.get("wordclouds", []) if (IMAGE_DIR / n).exists()]
    return jsonify(data)


@app.get("/assets/<path:filename>")
def assets(filename):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")) or not IMAGE_DIR.is_dir():
        return error("Image not found.", 404)
    return send_from_directory(IMAGE_DIR, filename)


@app.errorhandler(404)
def not_found(_):
    return error("Route not found. Try GET /, GET /dashboard, GET /health, POST /predict or POST /predict/batch.", 404)


@app.errorhandler(405)
def bad_method(_):
    return error("Method not allowed for this route (prediction routes need POST).", 405)


@app.errorhandler(500)
def server_error(_):
    return error("Internal server error - check the terminal / flask_server.log for details.", 500)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    print(f"Model: {type(model).__name__} | classes: {list(model.classes_)}")
    print(f"Dashboard (browser): http://127.0.0.1:{port}/dashboard")
    print(f"Open in Postman    : http://127.0.0.1:{port}/health")
    app.run(host=host, port=port, debug=False)
