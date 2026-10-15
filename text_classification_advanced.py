"""
Week 3 - ADVANCED experiments (run after, or instead of, text_classification.py)
===============================================================================
Adds to the baseline study:
  A. Feature ablation        : unigrams vs word 1-2 grams vs word + character n-grams
  B. Stronger models         : Logistic Regression, Linear SVM, Complement NB,
                               and a Stacking ensemble (LR + SVM + CNB -> LR meta-model)
  C. Robust evaluation       : repeated stratified 5x3 cross-validation (mean +/- std),
                               bootstrap 95% confidence interval on test macro-F1,
                               McNemar significance test (best model vs the others),
                               macro ROC-AUC, precision-recall curves, learning curve
  D. Optional pretrained NLP : Sentence-BERT embeddings + Logistic Regression (--embeddings)
  E. Optional scale-up       : all 20 newsgroups classes (--all20)

Usage:
    pip install scikit-learn pandas numpy matplotlib scipy
    python text_classification_advanced.py                 # 5 classes
    python text_classification_advanced.py --all20        # all 20 classes (slower)
    python text_classification_advanced.py --embeddings   # needs: pip install sentence-transformers
    python text_classification_advanced.py --csv my.csv   # own data, columns text,label

Outputs go to ./outputs (results_adv.json, learning_curve.png, pr_curves.png).
Run text_classification.py first if you also want the baseline figures/report sections.
"""
import argparse
import json
import os
import time
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     learning_curve)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

import text_classification as tc

SEED = tc.SEED


# ---------------------------------------------------------------- features
def word_tfidf(ngram=(1, 2)):
    return TfidfVectorizer(stop_words="english", ngram_range=ngram, min_df=2,
                           max_df=0.9, sublinear_tf=True, max_features=20000)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, max_features=30000)


FEATURE_SETS = {
    "Word unigrams": lambda: word_tfidf((1, 1)),
    "Word 1-2 grams": lambda: word_tfidf((1, 2)),
    "Word 1-2 grams + char 3-5 grams": lambda: FeatureUnion(
        [("word", word_tfidf((1, 2))), ("char", char_tfidf())]),
}


def scores_of(est, X):
    """Continuous class scores for ROC / PR curves."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)
    return est.decision_function(X)


def macro_prf(y, pred, labels):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels,
                                                 average="macro", zero_division=0)
    return float(p), float(r), float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--all20", action="store_true", help="use all 20 classes")
    ap.add_argument("--embeddings", action="store_true",
                    help="add Sentence-BERT embeddings + LR (needs sentence-transformers)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.all20:
        tc.CATEGORIES = None

    Xtr_raw, ytr_raw, Xte_raw, yte_raw, source = tc.load_data(args.csv)
    Xtr, ytr, _ = tc.preprocess(Xtr_raw, ytr_raw)
    Xte, yte, _ = tc.preprocess(Xte_raw, yte_raw)
    ytr, yte = np.array(ytr), np.array(yte)
    labels = sorted(set(ytr))
    if len(labels) < 3:
        raise SystemExit("The advanced script expects 3 or more classes.")
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {"source": source + (" - all 20 classes" if args.all20 else ""),
           "labels": labels, "n_train": len(Xtr), "n_test": len(Xte)}

    # ======================= A. feature ablation ==========================
    print("\n[A] Feature ablation (Linear SVM, C=0.5)")
    ablation = []
    for name, make in FEATURE_SETS.items():
        pipe = Pipeline([("feat", make()), ("clf", LinearSVC(C=0.5, random_state=SEED))])
        cv = cross_val_score(pipe, Xtr, ytr, cv=cv5, scoring="f1_macro", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        n_feat = int(sum(len(s.get_feature_names_out()) for _, s in
                         pipe.named_steps["feat"].transformer_list)
                     if hasattr(pipe.named_steps["feat"], "transformer_list")
                     else len(pipe.named_steps["feat"].get_feature_names_out()))
        row = {"features": name, "n_features": n_feat,
               "cv_f1_macro": float(cv.mean()),
               "accuracy": float(accuracy_score(yte, pred)),
               "f1_macro": float(f1_score(yte, pred, average="macro"))}
        ablation.append(row)
        print(f"  {name:36s} CV F1 {row['cv_f1_macro']:.4f} | test F1 {row['f1_macro']:.4f}")
    best_feat = max(ablation, key=lambda r: r["cv_f1_macro"])["features"]
    out["ablation"], out["best_features"] = ablation, best_feat
    print(f"  -> using '{best_feat}' for the models below (chosen by CV, not by test)")

    # ======================= B. models + ensemble =========================
    def pipe_with(clf):
        return Pipeline([("feat", FEATURE_SETS[best_feat]()), ("clf", clf)])

    svm_cal = lambda: __import__("sklearn.calibration", fromlist=["x"]).CalibratedClassifierCV(
        LinearSVC(C=0.5, random_state=SEED), cv=3)
    stack = StackingClassifier(
        estimators=[("lr", LogisticRegression(C=10, max_iter=2000, random_state=SEED)),
                    ("svm", svm_cal()),
                    ("cnb", ComplementNB(alpha=0.1))],
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        cv=3, n_jobs=1)
    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=SEED),
                                {"clf__C": [1, 10, 100]}),
        "Linear SVM": (LinearSVC(random_state=SEED), {"clf__C": [0.1, 0.5, 1, 5]}),
        "Complement NB": (ComplementNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
        "Stacking ensemble": (stack, {}),
    }
    print("\n[B] Training models")
    fitted, rows = {}, []
    for name, (clf, grid) in candidates.items():
        t0 = time.time()
        gs = GridSearchCV(pipe_with(clf), grid or {"clf": [clf]}, cv=cv5,
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, ytr)
        est = gs.best_estimator_
        pred = est.predict(Xte)
        p, r, f = macro_prf(yte, pred, labels)
        try:
            auc = float(roc_auc_score(label_binarize(yte, classes=est.classes_),
                                      scores_of(est, Xte), average="macro"))
        except Exception:
            auc = None
        rows.append({"model": name, "accuracy": float(accuracy_score(yte, pred)),
                     "precision_macro": p, "recall_macro": r, "f1_macro": f,
                     "roc_auc_macro": auc, "cv_f1_macro": float(gs.best_score_),
                     "train_seconds": time.time() - t0,
                     "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()
                                     if k != "clf"}})
        fitted[name] = (est, pred)
        print(f"  {name:22s} acc {rows[-1]['accuracy']:.4f} F1 {f:.4f} AUC {auc} "
              f"({rows[-1]['train_seconds']:.0f}s)")

    # ---------- D. optional pretrained sentence embeddings ----------
    if args.embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            print("\n[D] Sentence-BERT embeddings + Logistic Regression")
            t0 = time.time()
            enc = SentenceTransformer("all-MiniLM-L6-v2")
            Etr = enc.encode(Xtr, batch_size=64, show_progress_bar=True)
            Ete = enc.encode(Xte, batch_size=64)
            gs = GridSearchCV(LogisticRegression(max_iter=3000, random_state=SEED),
                              {"C": [1, 10, 100]}, cv=cv5, scoring="f1_macro", n_jobs=-1)
            gs.fit(Etr, ytr)
            pred = gs.predict(Ete)
            p, r, f = macro_prf(yte, pred, labels)
            auc = float(roc_auc_score(label_binarize(yte, classes=gs.classes_),
                                      gs.predict_proba(Ete), average="macro"))
            rows.append({"model": "SBERT embeddings + LR",
                         "accuracy": float(accuracy_score(yte, pred)),
                         "precision_macro": p, "recall_macro": r, "f1_macro": f,
                         "roc_auc_macro": auc, "cv_f1_macro": float(gs.best_score_),
                         "train_seconds": time.time() - t0,
                         "best_params": dict(gs.best_params_)})
            fitted["SBERT embeddings + LR"] = (gs.best_estimator_, pred)
            print(f"  acc {rows[-1]['accuracy']:.4f} F1 {f:.4f}")
        except ImportError:
            print("  sentence-transformers not installed; skipping (pip install sentence-transformers)")

    best = max(rows, key=lambda r: r["f1_macro"])
    best_name = best["model"]
    best_est, best_pred = fitted[best_name]
    out["models"], out["best_model"] = rows, best_name

    # ======================= C. robust evaluation =========================
    print("\n[C] Robust evaluation")
    # C1. repeated CV for every text-pipeline model (embedding model excluded: no pipeline)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for r in rows:
        est = fitted[r["model"]][0]
        if r["model"].startswith("SBERT"):
            continue
        sc = cross_val_score(est, Xtr, ytr, cv=rcv, scoring="f1_macro", n_jobs=-1)
        r["rcv_mean"], r["rcv_std"] = float(sc.mean()), float(sc.std())
        print(f"  repeated CV {r['model']:22s} {sc.mean():.4f} +/- {sc.std():.4f}")

    # C2. bootstrap CI for the best model's test macro-F1
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(yte))
    boots = [f1_score(yte[s], best_pred[s], average="macro")
             for s in (rng.choice(idx, len(idx)) for _ in range(1000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["bootstrap_ci"] = {"model": best_name, "low": float(lo), "high": float(hi)}
    print(f"  bootstrap 95% CI for {best_name}: [{lo:.4f}, {hi:.4f}]")

    # C3. McNemar exact test: best vs every other model
    tests = []
    right_best = best_pred == yte
    for name, (_, pred) in fitted.items():
        if name == best_name:
            continue
        right_other = pred == yte
        b = int(np.sum(right_best & ~right_other))
        c = int(np.sum(~right_best & right_other))
        pval = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
        tests.append({"other": name, "best_only_correct": b, "other_only_correct": c,
                      "p_value": pval})
        print(f"  McNemar {best_name} vs {name}: b={b} c={c} p={pval:.4f}")
    out["mcnemar"] = tests

    # C4. precision-recall curves (one-vs-rest) for best model
    if not best_name.startswith("SBERT"):
        sc = scores_of(best_est, Xte)
        Y = label_binarize(yte, classes=best_est.classes_)
        show = list(range(min(len(labels), 6)))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for i in show:
            pr, rc, _ = precision_recall_curve(Y[:, i], sc[:, i])
            ax.plot(rc, pr, label=str(best_est.classes_[i]))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"One-vs-rest precision-recall curves - {best_name}"
                     + (" (first 6 classes)" if len(labels) > 6 else ""), fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{args.out}/pr_curves.png", dpi=160)
        plt.close(fig)

        # C5. learning curve
        sizes, tr_s, va_s = learning_curve(best_est, Xtr, ytr, cv=3, n_jobs=-1,
                                           train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
                                           scoring="f1_macro", random_state=SEED)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sizes, tr_s.mean(1), "o-", label="Training macro-F1")
        ax.plot(sizes, va_s.mean(1), "o-", label="Validation macro-F1")
        ax.fill_between(sizes, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.2)
        ax.set_xlabel("Training documents")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Learning curve - {best_name}", fontsize=10)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{args.out}/learning_curve.png", dpi=160)
        plt.close(fig)
        out["learning_curve"] = {"sizes": [int(s) for s in sizes],
                                 "train": [float(v) for v in tr_s.mean(1)],
                                 "val": [float(v) for v in va_s.mean(1)]}

    with open(f"{args.out}/results_adv.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBest: {best_name}. Saved {args.out}/results_adv.json")


if __name__ == "__main__":
    main()
