#!/usr/bin/env python3
"""
Explore themes in AI coverage with LDA topic modeling.

Input:
  data/raw/guardian_ai_articles.csv

Outputs:
  outputs/topic_keywords.csv
  data/derived/document_topics.csv
  outputs/topic_prevalence_by_month.csv
  figures/figure7_topic_prevalence.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

LABEL_KEYWORDS = {
    "AI tools and ChatGPT": ["chatgpt", "openai", "chatbot", "tool", "tools", "model", "models", "generative", "llm", "claude"],
    "Business competition": ["business", "company", "companies", "google", "microsoft", "market", "startup", "investment", "tech", "apple", "amazon"],
    "Regulation and politics": ["government", "law", "laws", "regulation", "regulate", "policy", "politics", "election", "eu", "uk", "china"],
    "Education and work": ["school", "schools", "student", "students", "education", "exam", "university", "teacher", "job", "jobs", "workers", "work"],
    "Creative industries and copyright": ["copyright", "artist", "artists", "writer", "writers", "music", "film", "book", "books", "creative", "art"],
    "Risk, misinformation, and ethics": ["risk", "risks", "fake", "deepfake", "misinformation", "disinformation", "ethics", "safety", "danger", "bias", "harm"],
}
CUSTOM_STOPWORDS = {
    "guardian", "article", "articles", "says", "said", "say", "new", "latest", "live",
    "news", "world", "uk", "us", "australia", "australian", "today", "read", "best",
    "technology", "business", "opinion", "culture", "life", "style", "people", "year",
    "years", "time", "like", "make", "making", "use", "used", "using", "will", "could",
}


def clean_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = re.sub(r"<[^>]+>", " ", str(x))
    s = re.sub(r"[^A-Za-z0-9\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def get_text(df: pd.DataFrame) -> pd.Series:
    if "headline_trail_text" in df.columns:
        return df["headline_trail_text"].fillna("").astype(str).map(clean_text)
    title_col = "headline" if "headline" in df.columns else "web_title"
    title = df[title_col].fillna("").astype(str).map(clean_text) if title_col in df.columns else pd.Series([""] * len(df))
    trail = df["trail_text"].fillna("").astype(str).map(clean_text) if "trail_text" in df.columns else pd.Series([""] * len(df))
    return (title + " " + trail).str.strip()


def auto_label(words: list[str], used: set[str]) -> str:
    word_set = set(words)
    joined = " ".join(words)
    scores = {label: sum(1 for k in keys if k in word_set or k in joined) for label, keys in LABEL_KEYWORDS.items()}
    best, score = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[0]
    if score == 0 or best in used:
        return ""
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/raw/guardian_ai_articles.csv")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--derived-dir", default="data/derived")
    p.add_argument("--n-topics", type=int, default=6)
    p.add_argument("--max-features", type=int, default=3500)
    p.add_argument("--min-df", type=int, default=4)
    p.add_argument("--max-df", type=float, default=0.85)
    p.add_argument("--top-words", type=int, default=15)
    p.add_argument("--seed", type=int, default=370)
    args = p.parse_args()

    if not Path(args.data_file).is_file():
        p.error(
            f"Article data not found: {args.data_file}. Place the local corpus at this path "
            "or pass --data-file. See data/README.md for collection instructions."
        )
    if args.n_topics < 1 or args.top_words < 1:
        p.error("--n-topics and --top-words must both be positive integers.")

    outputs = Path(args.outputs_dir); outputs.mkdir(exist_ok=True, parents=True)
    figs = Path(args.fig_dir); figs.mkdir(exist_ok=True, parents=True)
    derived = Path(args.derived_dir); derived.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.data_file)
    df["topic_text"] = get_text(df)
    df = df[df["topic_text"].str.split().map(len) >= 4].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("The corpus contains no headlines and summaries with at least four words.")
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    elif "publication_date" in df.columns:
        df["month"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    else:
        df["month"] = pd.NaT

    vectorizer = CountVectorizer(
        lowercase=True, stop_words="english", min_df=args.min_df, max_df=args.max_df,
        max_features=args.max_features, ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
    )
    X = vectorizer.fit_transform(df["topic_text"])
    terms = np.array(vectorizer.get_feature_names_out())

    lda = LatentDirichletAllocation(
        n_components=args.n_topics, max_iter=25, learning_method="batch",
        random_state=args.seed, evaluate_every=-1
    )
    doc_topic = lda.fit_transform(X)

    rows = []
    for k, weights in enumerate(lda.components_):
        added = 0
        for idx in np.argsort(weights)[::-1]:
            term = str(terms[idx])
            if term in CUSTOM_STOPWORDS:
                continue
            if any(part in CUSTOM_STOPWORDS for part in term.split()) and not term.startswith("ai ") and not term.endswith(" ai"):
                continue
            rows.append({"topic_id": k, "rank": added + 1, "term": term, "weight": float(weights[idx])})
            added += 1
            if added >= args.top_words:
                break
    topic_keywords = pd.DataFrame(rows)

    labels, used = {}, set()
    for k in sorted(topic_keywords["topic_id"].unique()):
        words = topic_keywords.loc[topic_keywords["topic_id"] == k, "term"].head(15).tolist()
        label = auto_label(words, used)
        if not label:
            label = f"Topic {k + 1}: " + ", ".join(words[:3])
        labels[int(k)] = label
        used.add(label)

    topic_keywords["topic_label"] = topic_keywords["topic_id"].map(labels)
    topic_keywords.to_csv(outputs / "topic_keywords.csv", index=False)

    doc = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(args.n_topics)])
    doc["dominant_topic_id"] = doc_topic.argmax(axis=1)
    doc["dominant_topic_label"] = doc["dominant_topic_id"].map(labels)
    meta = [c for c in ["id", "publication_date", "month", "section_name", "web_title", "headline", "trail_text", "web_url"] if c in df.columns]
    pd.concat([df[meta].reset_index(drop=True), doc], axis=1).to_csv(derived / "document_topics.csv", index=False)

    topic_share = pd.DataFrame(doc_topic, columns=[labels[i] for i in range(args.n_topics)])
    topic_share["month"] = df["month"].to_numpy()
    monthly = topic_share.dropna(subset=["month"]).groupby("month").mean().reset_index()
    long = monthly.melt(id_vars="month", var_name="topic_label", value_name="topic_share")
    label_to_id = {v: k for k, v in labels.items()}
    long["topic_id"] = long["topic_label"].map(label_to_id)
    long = long[["month", "topic_id", "topic_label", "topic_share"]].sort_values(["month", "topic_id"])
    long.to_csv(outputs / "topic_prevalence_by_month.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    for label in monthly.columns:
        if label != "month":
            plt.plot(monthly["month"], monthly[label], marker="o", linewidth=1.3, label=label)
    plt.title("LDA topic prevalence over time in strict AI corpus")
    plt.xlabel("")
    plt.ylabel("Average topic share")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(figs / "figure7_topic_prevalence.png", dpi=200)
    plt.close()

    print("\nLDA topic modeling complete.")
    for k in sorted(labels):
        words = topic_keywords.loc[topic_keywords["topic_id"] == k, "term"].head(8).tolist()
        print(f"Topic {k + 1} - {labels[k]}: {', '.join(words)}")
    print("Check outputs/topic_keywords.csv and adjust topic names in the report if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
