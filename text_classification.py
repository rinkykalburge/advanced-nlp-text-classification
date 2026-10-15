"""
Week 3 - Implementing and Evaluating Text Classification Models
================================================================
Dataset  : 20 Newsgroups (public, built into scikit-learn) - 5 topic classes
           (or your own CSV with columns `text,label` via --csv)
Pipeline : cleaning -> TF-IDF (word 1-2 grams) -> classifier
Models   : Multinomial Naive Bayes (baseline), Logistic Regression,
           Linear SVM, Multi-Layer Perceptron (neural network)
Evaluate : accuracy, macro precision / recall / F1, per-class report,
           confusion matrix, 5-fold cross-validation, training time

Run:
    pip install scikit-learn pandas numpy matplotlib
    python text_classification.py            # downloads 20 Newsgroups (~14 MB) once
    python text_classification.py --csv my_data.csv

Outputs (in ./outputs):
    results.json, class_distribution.png, model_comparison.png,
    confusion_matrix.png, top_features.png
"""
import argparse
import json
import os
import re
import time
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # also applies to parallel workers
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
CATEGORIES = ["rec.sport.hockey", "sci.space", "sci.med",
              "comp.graphics", "talk.politics.misc"]


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_data(csv_path=None):
    """Return (train_texts, train_labels, test_texts, test_labels, source_name)."""
    if csv_path:
        df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
        tr, te = train_test_split(df, test_size=0.2, stratify=df["label"],
                                  random_state=SEED)
        return (tr["text"].tolist(), tr["label"].astype(str).tolist(),
                te["text"].tolist(), te["label"].astype(str).tolist(),
                os.path.basename(csv_path))

    # headers/footers/quotes are removed so the model learns from the message
    # body and not from metadata that leaks the answer (e.g. e-mail signatures).
    kw = dict(categories=CATEGORIES, remove=("headers", "footers", "quotes"),
              random_state=SEED)
    train = fetch_20newsgroups(subset="train", **kw)
    test = fetch_20newsgroups(subset="test", **kw)
    names = train.target_names
    return (train.data, [names[i] for i in train.target],
            test.data, [names[i] for i in test.target],
            "20 Newsgroups (5 categories)")


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
EMAIL = re.compile(r"\S+@\S+")
URL = re.compile(r"(https?://|www\.)\S+")
NON_ALPHA = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")


def clean_text(text):
    """Lower-case, remove e-mails/URLs/digits/punctuation, collapse spaces."""
    text = text.lower()
    text = EMAIL.sub(" ", text)
    text = URL.sub(" ", text)
    text = NON_ALPHA.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def preprocess(texts, labels):
    cleaned = [clean_text(t) for t in texts]
    keep = [i for i, t in enumerate(cleaned) if len(t.split()) >= 3]
    return [cleaned[i] for i in keep], [labels[i] for i in keep], keep


# --------------------------------------------------------------------------
# 3. Models and hyper-parameter grids
# --------------------------------------------------------------------------
def make_pipeline(clf):
    # The vectoriser is INSIDE the pipeline, so during cross-validation it is
    # fitted only on the training folds -> no data leakage.
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.9, sublinear_tf=True,
                                  max_features=20000)),
        ("clf", clf),
    ])


MODELS = {
    "Naive Bayes": (MultinomialNB(),
                    {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=SEED),
                            {"clf__C": [0.1, 1, 10, 100]}),
    "Linear SVM": (LinearSVC(random_state=SEED),
                   {"clf__C": [0.05, 0.1, 0.5, 1, 5]}),
    "Neural Network (MLP)": (MLPClassifier(hidden_layer_sizes=(128,),
                                           early_stopping=False,
                                           max_iter=40, random_state=SEED),
                             {"clf__alpha": [1e-4, 1e-2],
                              "clf__hidden_layer_sizes": [(64,), (128,)]}),
}


# --------------------------------------------------------------------------
# 4. Main experiment
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="optional CSV with columns: text,label")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Xtr_raw, ytr_raw, Xte_raw, yte_raw, source = load_data(args.csv)
    n_raw_train, n_raw_test = len(Xtr_raw), len(Xte_raw)

    Xtr, ytr, _ = preprocess(Xtr_raw, ytr_raw)
    Xte, yte, kept_te = preprocess(Xte_raw, yte_raw)
    labels = sorted(set(ytr))

    # ---- descriptive statistics (used in the report) ----
    len_before = np.mean([len(t.split()) for t in Xtr_raw])
    len_after = np.mean([len(t.split()) for t in Xtr])
    example_raw = next((t for t in Xtr_raw if 200 < len(t) < 500), Xtr_raw[0])
    stats = {
        "source": source, "labels": labels,
        "n_train_raw": n_raw_train, "n_test_raw": n_raw_test,
        "n_train": len(Xtr), "n_test": len(Xte),
        "dropped_train": n_raw_train - len(Xtr),
        "dropped_test": n_raw_test - len(Xte),
        "avg_words_before": float(len_before), "avg_words_after": float(len_after),
        "train_class_counts": pd.Series(ytr).value_counts().sort_index().to_dict(),
        "test_class_counts": pd.Series(yte).value_counts().sort_index().to_dict(),
        "example_raw": example_raw.strip()[:350],
        "example_clean": clean_text(example_raw)[:350],
    }

    # ---- class distribution figure ----
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cc = stats["train_class_counts"]
    ax.bar(list(cc.keys()), list(cc.values()), color="#4C72B0")
    ax.set_ylabel("Training documents")
    ax.set_title("Class distribution (training set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/class_distribution.png", dpi=160)
    plt.close(fig)

    # ---- train / tune / evaluate every model ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = [], {}
    for name, (clf, grid) in MODELS.items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        gs = GridSearchCV(make_pipeline(clf), grid, cv=cv,
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, ytr)
        train_time = time.time() - t0

        t1 = time.time()
        pred = gs.predict(Xte)
        pred_time = time.time() - t1

        p, r, f1, _ = precision_recall_fscore_support(
            yte, pred, labels=labels, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(
            yte, pred, labels=labels, average="weighted", zero_division=0)
        rep = classification_report(yte, pred, labels=labels,
                                    output_dict=True, zero_division=0)
        res = {
            "model": name,
            "best_params": {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v)
                            for k, v in gs.best_params_.items()},
            "cv_f1_macro": float(gs.best_score_),
            "accuracy": float(accuracy_score(yte, pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1),
            "precision_weighted": float(pw), "recall_weighted": float(rw),
            "f1_weighted": float(fw),
            "per_class": {l: {k: float(rep[l][k]) for k in
                              ("precision", "recall", "f1-score", "support")}
                          for l in labels},
            "train_seconds": float(train_time), "predict_seconds": float(pred_time),
            "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        }
        results.append(res)
        fitted[name] = (gs.best_estimator_, pred)
        print(f"best params: {res['best_params']}")
        print(f"CV macro-F1 {res['cv_f1_macro']:.4f} | test acc {res['accuracy']:.4f} "
              f"| test macro-F1 {res['f1_macro']:.4f} | time {train_time:.1f}s")

    best = max(results, key=lambda r: r["f1_macro"])
    best_name = best["model"]
    best_est, best_pred = fitted[best_name]
    print(f"\nBest model on test macro-F1: {best_name}")
    print(classification_report(yte, best_pred, labels=labels, zero_division=0))

    # ---- model comparison figure ----
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in results]
    x = np.arange(len(names))
    w = 0.2
    for i, (key, lab) in enumerate([("accuracy", "Accuracy"),
                                    ("precision_macro", "Precision (macro)"),
                                    ("recall_macro", "Recall (macro)"),
                                    ("f1_macro", "F1 (macro)")]):
        ax.bar(x + (i - 1.5) * w, [r[key] for r in results], w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    lo = min(min(r["accuracy"], r["f1_macro"]) for r in results)
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test-set performance of all models")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/model_comparison.png", dpi=160)
    plt.close(fig)

    # ---- confusion matrix of best model ----
    cm = np.array(best["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{args.out}/confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ---- top features per class (Logistic Regression: most interpretable) ----
    lr_est, _ = fitted["Logistic Regression"]
    vocab = np.array(lr_est.named_steps["tfidf"].get_feature_names_out())
    coefs = lr_est.named_steps["clf"].coef_
    classes = list(lr_est.named_steps["clf"].classes_)
    top = {c: vocab[np.argsort(coefs[i])[-8:][::-1]].tolist()
           for i, c in enumerate(classes)}
    fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (i, c) in zip(axes, enumerate(classes)):
        idx = np.argsort(coefs[i])[-8:]
        ax.barh(vocab[idx], coefs[i][idx], color="#55A868")
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Top 8 weighted terms per class (Logistic Regression)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}/top_features.png", dpi=160)
    plt.close(fig)

    # ---- error analysis: a few misclassified test documents ----
    errors = []
    for t, true, pr in zip(Xte, yte, best_pred):
        if true != pr:
            errors.append({"true": true, "predicted": pr, "text": t[:220]})
        if len(errors) == 4:
            break

    out = {"stats": stats, "results": results, "best_model": best_name,
           "top_features": top, "errors": errors,
           "n_features": int(len(vocab)),
           "tfidf_settings": "word 1-2 grams, stop words removed, min_df=2, "
                             "max_df=0.9, sublinear TF, max 20,000 features"}
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results and figures to ./{args.out}/")


if __name__ == "__main__":
    main()
