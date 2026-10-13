#!/usr/bin/env python3
"""
Compare classifiers for Technology/Business section membership.

Input:
  data/raw/guardian_ai_articles.csv

Outputs:
  outputs/model_performance.csv
  outputs/best_model_metrics.csv
  outputs/top_model_features.csv
  data/derived/test_predictions.csv
  figures/figure4_model_performance.png
  figures/figure5_confusion_matrix.png
  figures/figure6_variable_importance.png
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as exc:
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except Exception:
    HAS_VADER = False

KEYWORD_FLAGS = {
    "has_chatgpt": r"\bchatgpt\b",
    "has_openai": r"\bopenai\b",
    "has_llm": r"\bllm(?:s)?\b|\blarge language model(?:s)?\b",
    "has_generative_ai": r"\bgenerative ai\b",
    "has_machine_learning": r"\bmachine learning\b",
    "has_deep_learning": r"\bdeep learning\b",
    "has_neural_network": r"\bneural network(?:s)?\b",
    "has_gemini": r"\bgemini\b|\bgoogle gemini\b",
    "has_claude": r"\bclaude\b|\banthropic\b",
    "has_dalle_midjourney": r"\bdall[- ]?e\b|\bmidjourney\b",
    "has_regulation": r"\bregulat(?:e|ion|ions|ory)\b|\blaw(?:s)?\b|\bpolicy\b|\bgovernance\b",
    "has_jobs_labor": r"\bjob(?:s)?\b|\blabou?r\b|\bworker(?:s)?\b|\bemployment\b|\bautomation\b",
    "has_education": r"\beducation\b|\bschool(?:s)?\b|\bstudent(?:s)?\b|\bexam(?:s)?\b|\buniversit(?:y|ies)\b",
    "has_copyright_creativity": r"\bcopyright\b|\bartist(?:s)?\b|\bwriter(?:s)?\b|\bcreative\b|\bmusic\b|\bfilm\b|\bbook(?:s)?\b",
    "has_risk_ethics": r"\brisk(?:s)?\b|\bethic(?:s|al)?\b|\bharm(?:s)?\b|\bdanger(?:s|ous)?\b|\bsafety\b",
    "has_election_misinformation": r"\belection(?:s)?\b|\bmisinformation\b|\bdisinformation\b|\bfake\b|\bdeepfake(?:s)?\b",
}

FORMAT_FLAGS = {
    "format_news": "news",
    "format_comment": "comment",
    "format_features": "features",
    "format_reviews": "reviews",
    "format_analysis": "analysis",
    "format_explainers": "explainers",
    "format_interviews": "interviews",
}


def parse_list_like(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(v) for v in val if str(v).strip()]
        except Exception:
            pass
    if ";" in s:
        return [v.strip() for v in s.split(";") if v.strip()]
    if "|" in s:
        return [v.strip() for v in s.split("|") if v.strip()]
    return [s]


def clean_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = re.sub(r"<[^>]+>", " ", str(x))
    return re.sub(r"\s+", " ", s).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def get_text(df: pd.DataFrame) -> pd.Series:
    if "headline_trail_text" in df.columns:
        return df["headline_trail_text"].fillna("").astype(str).map(clean_text)
    title_col = "headline" if "headline" in df.columns else "web_title"
    title = df[title_col].fillna("").astype(str).map(clean_text) if title_col in df.columns else pd.Series([""] * len(df))
    trail = df["trail_text"].fillna("").astype(str).map(clean_text) if "trail_text" in df.columns else pd.Series([""] * len(df))
    return (title + " " + trail).str.strip()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_text"] = get_text(df)

    if "tech_business_frame" not in df.columns:
        if "section_name" not in df.columns:
            raise RuntimeError("Need tech_business_frame or section_name.")
        df["tech_business_frame"] = df["section_name"].fillna("").astype(str).str.lower().isin(["technology", "business"]).astype(int)
    df["tech_business_frame"] = pd.to_numeric(df["tech_business_frame"], errors="coerce")

    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    elif "publication_datetime_utc" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_datetime_utc"], errors="coerce")
    else:
        df["publication_date"] = pd.NaT

    if "month" in df.columns:
        df["month_date"] = pd.to_datetime(df["month"], errors="coerce")
    else:
        df["month_date"] = df["publication_date"].dt.to_period("M").dt.to_timestamp()

    if "year" not in df.columns:
        df["year"] = df["month_date"].dt.year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "month_num" not in df.columns:
        df["month_num"] = df["month_date"].dt.month
    df["month_num"] = pd.to_numeric(df["month_num"], errors="coerce")

    if "post_chatgpt_period" not in df.columns:
        df["post_chatgpt_period"] = (df["publication_date"] >= pd.Timestamp("2022-11-30")).astype(int)

    if "wordcount" not in df.columns:
        df["wordcount"] = np.nan
    df["wordcount"] = pd.to_numeric(df["wordcount"], errors="coerce")

    if "headline_word_count" not in df.columns:
        head = df["headline"].fillna("").astype(str) if "headline" in df.columns else df.get("web_title", pd.Series([""] * len(df))).fillna("").astype(str)
        df["headline_word_count"] = head.map(count_words)
    if "trail_word_count" not in df.columns:
        trail = df["trail_text"].fillna("").astype(str) if "trail_text" in df.columns else pd.Series([""] * len(df))
        df["trail_word_count"] = trail.map(count_words)
    if "text_word_count" not in df.columns:
        df["text_word_count"] = df["model_text"].map(count_words)

    if "n_keyword_tags" not in df.columns:
        df["n_keyword_tags"] = df["keywords"].map(lambda x: len(parse_list_like(x))) if "keywords" in df.columns else 0
    if "n_format_tags" not in df.columns:
        df["n_format_tags"] = df["format_tags"].map(lambda x: len(parse_list_like(x))) if "format_tags" in df.columns else 0

    text_lower = df["model_text"].fillna("").astype(str).str.lower()
    for col, pat in KEYWORD_FLAGS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) if col in df.columns else text_lower.str.contains(pat, regex=True).astype(int)

    fmt_text = df["format_tags"].map(lambda x: " | ".join(parse_list_like(x)).lower()) if "format_tags" in df.columns else pd.Series([""] * len(df))
    for col, pat in FORMAT_FLAGS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) if col in df.columns else fmt_text.str.contains(pat, regex=True).astype(int)

    if "sentiment_compound" not in df.columns:
        if HAS_VADER:
            analyzer = SentimentIntensityAnalyzer()
            df["sentiment_compound"] = df["model_text"].map(lambda s: analyzer.polarity_scores(s or "")["compound"])
        else:
            raise RuntimeError(
                "VADER sentiment is required when sentiment_compound is not already present. "
                "Install the project dependencies with: pip install -r requirements.txt"
            )
    return df


def make_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    text = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=2500, sublinear_tf=True)
    nums = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
    return ColumnTransformer([("text", text, "model_text"), ("numeric", nums, numeric_cols)], sparse_threshold=0.8)


def scores(model: Pipeline, x_test: pd.DataFrame):
    y_pred = model.predict(x_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(x_test)[:, 1]
        except Exception:
            pass
    if y_score is None and hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(x_test)
        except Exception:
            pass
    return y_pred, y_score


def evaluate(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred, y_score = scores(model, x_test)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_score is not None:
        row["roc_auc"] = roc_auc_score(y_test, y_score)
    return row


def feature_importance(name: str, model: Pipeline) -> pd.DataFrame:
    try:
        names = model.named_steps["features"].get_feature_names_out()
    except Exception:
        return pd.DataFrame()
    clf = model.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        vals = np.ravel(clf.coef_)
        out = pd.DataFrame({"feature": names, "importance": np.abs(vals), "signed_coefficient": vals})
        out["importance_type"] = "absolute logistic coefficient"
    elif hasattr(clf, "feature_importances_"):
        out = pd.DataFrame({"feature": names, "importance": np.ravel(clf.feature_importances_)})
        out["signed_coefficient"] = np.nan
        out["importance_type"] = "tree impurity importance"
    else:
        return pd.DataFrame()
    out["model"] = name
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_performance(perf: pd.DataFrame, fig_dir: Path):
    plot_df = perf.sort_values("f1", ascending=True)
    y = np.arange(len(plot_df))
    h = 0.38
    plt.figure(figsize=(8.5, 4.8))
    plt.barh(y - h/2, plot_df["f1"], height=h, label="F1")
    plt.barh(y + h/2, plot_df["roc_auc"].fillna(0), height=h, label="ROC-AUC")
    plt.yticks(y, plot_df["model"])
    plt.xlim(0, 1)
    plt.xlabel("Test-set score")
    plt.title("Classification model performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure4_model_performance.png", dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, fig_dir: Path, name: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion matrix: {name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Broader public", "Tech/Business"], rotation=20, ha="right")
    ax.set_yticks([0, 1], labels=["Broader public", "Tech/Business"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, fig_dir: Path):
    if imp.empty:
        return
    top = imp.head(20).copy().iloc[::-1]
    labels = top["feature"].str.replace("text__", "text: ", regex=False).str.replace("numeric__", "", regex=False).str.replace("_", " ", regex=False)
    plt.figure(figsize=(9.5, 6.0))
    plt.barh(labels, top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top model features ({top['model'].iloc[-1]})")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure6_variable_importance.png", dpi=200)
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/raw/guardian_ai_articles.csv")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--derived-dir", default="data/derived")
    p.add_argument("--seed", type=int, default=370)
    p.add_argument("--test-size", type=float, default=0.20)
    args = p.parse_args()

    if not Path(args.data_file).is_file():
        p.error(
            f"Article data not found: {args.data_file}. Place the local corpus at this path "
            "or pass --data-file. See data/README.md for collection instructions."
        )
    if not 0 < args.test_size < 1:
        p.error("--test-size must be between 0 and 1.")

    outputs = Path(args.outputs_dir); outputs.mkdir(exist_ok=True, parents=True)
    figs = Path(args.fig_dir); figs.mkdir(exist_ok=True, parents=True)
    derived = Path(args.derived_dir); derived.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.data_file)
    df = add_features(df)
    df = df.dropna(subset=["tech_business_frame"])
    df = df[df["model_text"].fillna("").str.strip().ne("")]
    y = df["tech_business_frame"].astype(int)
    if y.nunique() != 2:
        raise RuntimeError("Target needs both 0 and 1 classes.")

    numeric_cols = [
        "year", "month_num", "post_chatgpt_period", "wordcount",
        "headline_word_count", "trail_word_count", "text_word_count",
        "n_keyword_tags", "n_format_tags", "sentiment_compound",
        *KEYWORD_FLAGS.keys(), *FORMAT_FLAGS.keys()
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x = df[["model_text", *numeric_cols]]
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, df.index.to_numpy(), test_size=args.test_size, random_state=args.seed, stratify=y
    )

    def pipe(clf):
        return Pipeline([("features", make_preprocessor(numeric_cols)), ("classifier", clf)])

    models = {
        "Majority baseline": pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=args.seed)),
        "Random forest": pipe(RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)),
    }
    if HAS_XGBOOST:
        pos = int(y_train.sum()); neg = int((y_train == 0).sum())
        models["XGBoost"] = pipe(XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.90,
            colsample_bytree=0.90, objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=(neg / pos if pos else 1.0), random_state=args.seed, n_jobs=-1
        ))
    else:
        print(f"XGBoost unavailable; comparing the remaining classifiers. {XGBOOST_IMPORT_ERROR}")

    fitted, rows = {}, []
    for name, model in models.items():
        print(f"Fitting {name} ...")
        model.fit(x_train, y_train)
        fitted[name] = model
        rows.append(evaluate(name, model, x_test, y_test))

    perf = pd.DataFrame(rows).sort_values(["f1", "roc_auc", "accuracy"], ascending=False).reset_index(drop=True)
    perf.to_csv(outputs / "model_performance.csv", index=False)
    plot_performance(perf, figs)

    cand = perf[perf["model"] != "Majority baseline"]
    best_name = str(cand.iloc[0]["model"] if len(cand) else perf.iloc[0]["model"])
    best_model = fitted[best_name]
    y_pred, y_score = scores(best_model, x_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion(cm, figs, best_name)

    perf[perf["model"] == best_name].assign(selected_model=best_name).to_csv(outputs / "best_model_metrics.csv", index=False)

    imp = feature_importance(best_name, best_model)
    if imp.empty and "Logistic regression" in fitted:
        imp = feature_importance("Logistic regression", fitted["Logistic regression"])
    imp.to_csv(outputs / "top_model_features.csv", index=False)
    plot_importance(imp, figs)

    pred_cols = [c for c in ["id", "publication_date", "section_name", "web_title", "headline", "trail_text", "web_url"] if c in df.columns]
    preds = df.loc[idx_test, pred_cols].copy()
    preds["true_tech_business_frame"] = y_test.to_numpy()
    preds["predicted_tech_business_frame"] = y_pred
    preds["predicted_score"] = y_score if y_score is not None else np.nan
    preds.to_csv(derived / "test_predictions.csv", index=False)

    print("\nSection classification complete.")
    print(perf.to_string(index=False))
    print("Selected model:", best_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
