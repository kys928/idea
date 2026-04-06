#!/usr/bin/env python3
# emotion_inference_app.py — Minimal DARK UI with animated probability bars (no charts, no API/footer).

from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
from scipy.sparse import hstack
import gradio as gr

# ----------------------------- PATHS -----------------------------
ART = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts")

CHAR_VEC_PATH  = ART / "tfidf_char_vectorizer.joblib"
WORD_VEC_PATH  = ART / "tfidf_word_vectorizer.joblib"
LABEL_ENC_PATH = ART / "label_encoder.joblib"

MODEL_CANDIDATES = {
    "Logistic Regression": [
        ART / "logreg_tfidf_model.joblib",
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\LogReg\LogReg_best.joblib"),
    ],
    "LinearSVC": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\LinearSVC\LinearSVC_best.joblib"),
    ],
    "KNN": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\KNN\KNN_best.joblib"),
    ],
    "Naive Bayes": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\NaiveBayes\ComplementNB_best.joblib"),
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\NaiveBayes\NaiveBayes_best.joblib"),
    ],
}

# ----------------------------- LOAD ARTIFACTS -----------------------------
def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def _normalize(s: str) -> str:
    if s is None:
        return ""
    return s.strip().lower()

_char_vec = joblib.load(CHAR_VEC_PATH) if CHAR_VEC_PATH.exists() else None
_word_vec = joblib.load(WORD_VEC_PATH) if WORD_VEC_PATH.exists() else None
assert _char_vec is not None or _word_vec is not None, "No TF-IDF vectorizer found."

_le = joblib.load(LABEL_ENC_PATH) if LABEL_ENC_PATH.exists() else None

def classes_to_names(clf_classes) -> list[str]:
    cc = np.asarray(clf_classes)
    if _le is None:
      return [str(x) for x in cc]
    if cc.dtype.kind in "OUS":
      return [str(x) for x in cc]
    return _le.inverse_transform(cc.astype(int)).tolist()

def vectorize_texts(texts: list[str]):
    texts = [_normalize(t) for t in texts]
    parts = []
    if _char_vec is not None:
        parts.append(_char_vec.transform(texts))
    if _word_vec is not None:
        parts.append(_word_vec.transform(texts))
    return parts[0] if len(parts) == 1 else hstack(parts, format="csr")

MODELS = {}
for name, cands in MODEL_CANDIDATES.items():
    p = _first_existing(cands)
    if p:
        try:
            MODELS[name] = joblib.load(p)
        except Exception as e:
            print(f"[warn] Failed to load {name} from {p}: {e}")
assert MODELS, "No models loaded. Check MODEL_CANDIDATES paths."

# ----------------------------- INFERENCE -----------------------------
def apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    eps = 1e-12
    t = max(0.1, float(temp))
    powered = np.power(np.clip(probs, eps, 1.0), 1.0 / t)
    s = powered.sum()
    return powered / (s if s > 0 else eps)

def probs_from_model(model, X, temp: float) -> tuple[np.ndarray, list[str]]:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
    else:
        if not hasattr(model, "decision_function"):
            yhat = model.predict(X)[0]
            classes = getattr(model, "classes_", np.arange(1))
            p = np.ones(len(classes), dtype=np.float64) * (1.0 / len(classes))
            try:
                idx = int(np.where(classes == yhat)[0][0])
                p[idx] = max(0.7, 1.0 / len(classes))
                p = p / p.sum()
            except Exception:
                pass
        else:
            logits = model.decision_function(X)
            logits = logits[0] if getattr(logits, "ndim", 1) > 1 else logits
            z = logits - np.max(logits)
            ez = np.exp(z / max(0.1, float(temp)))
            p = ez / np.sum(ez)
    p = apply_temperature(np.asarray(p, dtype=np.float64), temp)
    class_names = classes_to_names(getattr(model, "classes_", np.arange(len(p))))
    return p, class_names

def probs_html(labels: list[str], probs: np.ndarray, k: int) -> str:
    order = np.argsort(probs)[::-1][:k]
    rows = []
    for i in order:
        lbl = str(labels[i]).replace("<", "&lt;").replace(">", "&gt;")
        pct = f"{probs[i]*100:.1f}"
        width = f"{probs[i]*100:.2f}%"
        rows.append(f"""
          <div class="prob-row">
            <div class="prob-head">
              <div class="prob-label">{lbl}</div>
              <div class="prob-score">{pct}%</div>
            </div>
            <div class="prob-bar">
              <div class="prob-fill" style="width:{width};"></div>
            </div>
          </div>
        """)
    return f"<div class='prob-wrap'>{''.join(rows)}</div>"

# ----------------------------- THEME & CSS (FULL DARK + REQUESTED OVERRIDES) -----------------------------
THEME = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="rgb(20,24,29)",     # global dark bg (requested)
    background_fill_primary="rgb(20,24,29)",
    background_fill_secondary="#0f131a",
    block_background_fill="#0f131a",
    block_border_color="#2a3340",
    body_text_color="#e6edf3",
    body_text_color_subdued="#9aa7b1",
    link_text_color="#6ee7b7",
    slider_color="#34d399",
)

CSS = """
:root { color-scheme: dark; }
html, body, .gradio-container { background: rgb(20 24 29) !important; color:#e6edf3 !important; }

/* Hide footer, badges, and settings */
footer, .built-with-badge, [data-testid="settings-button"],
button[aria-label="Settings"], button[title="Settings"] { display:none !important; }

/* Panels */
.gradio-container .gr-block, .gradio-container .gr-panel, .gradio-container .container,
.gradio-container .gr-box, .gradio-container .form, .gradio-container .prose {
  background:#0f131a !important; color:#e6edf3 !important; border-color:#2a3340 !important;
}

/* Inputs */
textarea, input[type="text"], input[type="search"], select {
  background:#121821 !important; color:#e6edf3 !important; border:1px solid #2a3340 !important;
}
textarea::placeholder, input::placeholder { color:#7c8b98 !important; }
select option { background:#121821; color:#e6edf3; }

/* Sliders & toggles */
input[type="range"] { accent-color:#34d399; }
input[type="checkbox"], input[type="radio"] { accent-color:#34d399; }

/* Buttons — make them black */
button, .gr-button, .gradio-button {
  background:#000 !important; color:#e6edf3 !important; border:1px solid #2a3340 !important;
}
button:hover, .gr-button:hover, .gradio-button:hover { filter:brightness(1.1); }

/* Probability bars */
.prob-wrap { display:flex; flex-direction:column; gap:10px; }
.prob-row  { display:flex; flex-direction:column; gap:6px; }
.prob-head { display:flex; justify-content:space-between; align-items:center; }
.prob-label{ color:#e6edf3; font-size:14px; font-weight:600; }
.prob-score{ color:#9aa7b1; font-size:12px; }
.prob-bar  { height:12px; background:#2a3340; border-radius:999px; overflow:hidden; }
.prob-fill { height:12px; width:0; background:linear-gradient(90deg,#34d399,#22d3ee); transition:width 420ms ease; }
.legend small { color:#9aa7b1; }

/* Your explicit overrides for current Gradio build (class names may change between versions) */
.wrap-inner.svelte-1hfxrpf.svelte-1hfxrpf { background:#334155 !important; }
input.svelte-1hfxrpf.svelte-1hfxrpf { color:#dedfdf !important; }

input.svelte-1kajgn1.svelte-1kajgn1 { color:black !important; }

/* Fallbacks if those hashes change */
[class*="wrap-inner"] { background:#334155 !important; }
"""

# ----------------------------- UI -----------------------------
with gr.Blocks(theme=THEME, css=CSS, title="Emotion Inference") as demo:
    gr.Markdown("<div style='font-size:22px;font-weight:700'>Emotion Inference</div>")

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                label="Model",
                choices=list(MODELS.keys()),
                value=next(iter(MODELS.keys())),
            )
            text_in = gr.Textbox(
                label="Text",
                placeholder="Type an utterance to classify its emotion…",
                lines=8,
            )
            with gr.Row():
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-k")
                temp = gr.Slider(0.5, 1.6, value=1.0, step=0.1, label="Temperature")

            gr.Markdown(
                """
<div class="legend"><small>
<b>Top-k</b>: how many highest-probability emotions to show.<br>
<b>Temperature</b>: lower = sharper (more confident), higher = softer (more spread).
</small></div>"""
            )

            with gr.Row():
                run_btn   = gr.Button("Classify")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            pred_label  = gr.Markdown("**Prediction:** —")
            conf_md     = gr.Markdown("_Waiting for input…_")
            probs_panel = gr.HTML("")

    # ------------------ functions ------------------
    def do_classify(model_name: str, text: str, k: int, temperature: float):
        text = (text or "").strip()
        if not text:
            return ("**Prediction:** —", "_Please type some text._", "")
        model = MODELS.get(model_name)
        if model is None:
            return ("**Prediction:** —", f"_Model '{model_name}' not loaded._", "")
        X = vectorize_texts([text])
        probs, class_names = probs_from_model(model, X, temperature)
        order = np.argsort(probs)[::-1]
        top_idx  = int(order[0])
        top_name = class_names[top_idx]
        top_conf = float(probs[top_idx])
        html = probs_html(class_names, probs, int(k))
        return (f"**Prediction:** {top_name}", f"_Confidence:_ **{top_conf*100:.1f}%**", html)

    run_btn.click(do_classify, inputs=[model_dd, text_in, topk, temp],
                  outputs=[pred_label, conf_md, probs_panel])

    clear_btn.click(fn=lambda: ("",), outputs=text_in).then(
        fn=lambda: ("**Prediction:** —", "_Waiting for input…_", ""),
        outputs=[pred_label, conf_md, probs_panel]
    )

if __name__ == "__main__":
    try:
        demo.queue()
    except Exception:
        pass
    demo.launch(show_api=False)
