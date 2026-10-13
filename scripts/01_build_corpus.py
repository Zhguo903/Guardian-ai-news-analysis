#!/usr/bin/env python3
"""Build an AI-focused corpus, features, audit sample, and summary figures.

Input: a local broad-corpus CSV in data/raw/guardian_ai_articles_broad.csv.
Article text is kept in ignored local directories; only aggregate results are public.
The historical search and filtering strategy is documented in data/README.md.
"""

from __future__ import annotations

import argparse
import ast
import html as ihtml
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AI_KEYWORD_PATTERNS = [
    r"\bai\b",
    r"artificial intelligence",
    r"machine learning",
    r"deep learning",
    r"neural network",
    r"large language model",
    r"large language models",
    r"llm",
    r"llms",
    r"generative ai",
    r"chatgpt",
    r"openai",
    r"google gemini",
    r"gemini ai",
    r"google deepmind",
    r"deepmind",
    r"claude ai",
    r"anthropic",
    r"dall-e",
    r"dalle",
    r"midjourney",
]

AI_TEXT_DIRECT_PATTERNS = [
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural network(?:s)?\b",
    r"\bgenerative ai\b",
    r"\blarge language model(s)?\b",
    r"\bllm(s)?\b",
    r"\bchatgpt\b",
    r"\bopenai\b",
    r"\bgoogle gemini\b",
    r"\bgemini ai\b",
    r"\bgoogle deepmind\b",
    r"\bdeepmind\b",
    r"\bclaude ai\b",
    r"\banthropic\b",
    r"\bdall[- ]?e\b",
    r"\bmidjourney\b",
]

AI_CONTEXT_WORDS_FOR_STANDALONE_AI = [
    "artificial", "intelligence", "machine", "learning", "model", "models",
    "algorithm", "algorithms", "chatbot", "chatbots", "generative", "automation",
    "automated", "technology", "software", "tool", "tools", "regulation", "regulate",
    "risk", "risks", "ethics", "copyright", "jobs", "workers", "education",
    "openai", "chatgpt", "llm", "neural", "deep learning",
]

KEYWORD_FLAG_PATTERNS = {
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


def strip_html(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = str(x)
    x = re.sub(r"<[^>]+>", " ", x)
    x = ihtml.unescape(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_list_like(x: Any) -> list[str]:
    """Convert list-like CSV/pickle cells into a clean list of strings."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, tuple) or isinstance(x, set):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [str(v) for v in out if str(v).strip()]
        except Exception:
            pass
    if ";" in s:
        return [v.strip() for v in s.split(";") if v.strip()]
    if "|" in s:
        return [v.strip() for v in s.split("|") if v.strip()]
    return [s]


def month_table(from_date: str, to_date: str) -> pd.DataFrame:
    start = pd.to_datetime(from_date)
    end = pd.to_datetime(to_date)
    month_starts = pd.date_range(start=start.replace(day=1), end=end.replace(day=1), freq="MS")
    month_ends = month_starts + pd.offsets.MonthEnd(0)
    month_ends = month_ends.where(month_ends <= end, end)
    return pd.DataFrame({"month_start": pd.to_datetime(month_starts), "month_end": pd.to_datetime(month_ends)})


def normalize_articles(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        raise RuntimeError("Article data is empty.")
    articles = articles.copy()
    if "id" not in articles.columns:
        raise RuntimeError("Data must contain an 'id' column for Guardian article IDs.")

    for col in ["headline", "trail_text", "web_title", "section_name", "section_id", "web_url"]:
        if col not in articles.columns:
            articles[col] = ""

    if "publication_date" not in articles.columns:
        if "publication_datetime_utc" in articles.columns:
            articles["publication_date"] = pd.to_datetime(articles["publication_datetime_utc"], errors="coerce").dt.date
        else:
            articles["publication_date"] = pd.NaT
    articles["publication_date"] = pd.to_datetime(articles["publication_date"], errors="coerce")

    if "month" not in articles.columns:
        articles["month"] = articles["publication_date"].dt.to_period("M").dt.to_timestamp()
    else:
        articles["month"] = pd.to_datetime(articles["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    if "year" not in articles.columns:
        articles["year"] = articles["month"].dt.year

    if "wordcount" not in articles.columns:
        articles["wordcount"] = np.nan
    articles["wordcount"] = pd.to_numeric(articles["wordcount"], errors="coerce")

    if "keywords" not in articles.columns:
        articles["keywords"] = [[] for _ in range(len(articles))]
    if "format_tags" not in articles.columns:
        articles["format_tags"] = [[] for _ in range(len(articles))]

    articles["keywords"] = articles["keywords"].apply(parse_list_like)
    articles["format_tags"] = articles["format_tags"].apply(parse_list_like)
    articles["headline"] = articles["headline"].fillna("").astype(str).apply(strip_html)
    articles["trail_text"] = articles["trail_text"].fillna("").astype(str).apply(strip_html)
    articles["web_title"] = articles["web_title"].fillna("").astype(str).apply(strip_html)
    articles["section_name"] = articles["section_name"].fillna("Unknown").replace("", "Unknown")
    articles["headline_trail_text"] = (articles["headline"] + " " + articles["trail_text"]).str.strip()
    articles = articles.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return articles


def has_ai_keyword(keywords: list[str]) -> bool:
    kw_text = " | ".join(str(k).lower() for k in keywords or [])
    return any(re.search(p, kw_text, flags=re.IGNORECASE) for p in AI_KEYWORD_PATTERNS)


def has_ai_text_signal(text: str) -> bool:
    raw = text or ""
    lower = raw.lower()
    if any(re.search(p, lower, flags=re.IGNORECASE) for p in AI_TEXT_DIRECT_PATTERNS):
        return True
    # Standalone AI is allowed only with context words to avoid overinclusive matching.
    if re.search(r"\bai\b", lower, flags=re.IGNORECASE):
        return any(cw in lower for cw in AI_CONTEXT_WORDS_FOR_STANDALONE_AI)
    # Also catch A.I. with context.
    if re.search(r"\ba\.i\.\b", lower, flags=re.IGNORECASE):
        return any(cw in lower for cw in AI_CONTEXT_WORDS_FOR_STANDALONE_AI)
    return False


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def add_features(articles: pd.DataFrame) -> pd.DataFrame:
    articles = articles.copy()
    articles["ai_keyword_hit"] = articles["keywords"].apply(has_ai_keyword)
    articles["ai_text_hit"] = articles["headline_trail_text"].apply(has_ai_text_signal)
    articles["ai_strict"] = articles["ai_keyword_hit"] | articles["ai_text_hit"]

    strict = articles[articles["ai_strict"]].copy().reset_index(drop=True)
    strict["section_name"] = strict["section_name"].fillna("Unknown").replace("", "Unknown")
    strict["section_name_lower"] = strict["section_name"].str.lower()
    strict["tech_business_frame"] = strict["section_name_lower"].isin(["technology", "business"]).astype(int)
    strict["frame_label"] = np.where(strict["tech_business_frame"] == 1, "technology_business", "broader_public_issue")

    strict["publication_date"] = pd.to_datetime(strict["publication_date"], errors="coerce")
    strict["month"] = pd.to_datetime(strict["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    strict["year"] = strict["month"].dt.year.astype("Int64")
    strict["month_num"] = strict["month"].dt.month.astype("Int64")
    strict["post_chatgpt_period"] = (strict["publication_date"] >= pd.Timestamp("2022-11-30")).astype(int)

    strict["headline_word_count"] = strict["headline"].apply(count_words)
    strict["trail_word_count"] = strict["trail_text"].apply(count_words)
    strict["text_word_count"] = strict["headline_trail_text"].apply(count_words)
    strict["n_keyword_tags"] = strict["keywords"].apply(lambda x: len(x or []))
    strict["n_format_tags"] = strict["format_tags"].apply(lambda x: len(x or []))

    text_lower = strict["headline_trail_text"].fillna("").str.lower()
    for col, pat in KEYWORD_FLAG_PATTERNS.items():
        strict[col] = text_lower.str.contains(pat, regex=True).astype(int)

    format_text = strict["format_tags"].apply(lambda xs: " | ".join(str(x).lower() for x in xs or []))
    for col, pat in FORMAT_FLAGS.items():
        strict[col] = format_text.str.contains(pat, regex=True).astype(int)

    # Section is the outcome source and is excluded from predictors.
    return strict


def save_csv_with_lists(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ["keywords", "format_tags"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda xs: "; ".join(str(x) for x in (xs or [])) if isinstance(xs, list) else xs)
    out.to_csv(path, index=False)


def full_months_from_data(articles: pd.DataFrame) -> pd.DataFrame:
    min_month = pd.to_datetime(articles["month"], errors="coerce").min()
    max_month = pd.to_datetime(articles["month"], errors="coerce").max()
    if pd.isna(min_month) or pd.isna(max_month):
        return pd.DataFrame({"month": []})
    months = pd.date_range(min_month.to_period("M").to_timestamp(), max_month.to_period("M").to_timestamp(), freq="MS")
    return pd.DataFrame({"month": months})


def make_monthly_counts(broad: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    full_months = full_months_from_data(broad)
    broad_counts = broad.groupby("month", as_index=False).agg(broad_articles=("id", "nunique"))
    strict_counts = strict.groupby("month", as_index=False).agg(strict_articles=("id", "nunique"))
    out = full_months.merge(broad_counts, on="month", how="left").merge(strict_counts, on="month", how="left")
    out[["broad_articles", "strict_articles"]] = out[["broad_articles", "strict_articles"]].fillna(0).astype(int)
    return out


def figure_monthly_counts(counts: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.plot(counts["month"], counts["broad_articles"], marker="o", linewidth=1.4, label="Broad corpus")
    plt.plot(counts["month"], counts["strict_articles"], marker="o", linewidth=1.4, label="Strict corpus")
    plt.title("Monthly AI-related article volume: broad vs strict corpus")
    plt.xlabel("")
    plt.ylabel("Unique articles in month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_monthly_volume_broad_vs_strict.png", dpi=200)
    plt.close()


def figure_top_sections(strict: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    top_sections = strict["section_name"].fillna("Unknown").value_counts().reset_index()
    top_sections.columns = ["section_name", "n"]
    top_sections["share"] = top_sections["n"] / top_sections["n"].sum()
    plot_sections = top_sections.head(12).iloc[::-1]
    plt.figure(figsize=(9, 5.5))
    plt.barh(plot_sections["section_name"], plot_sections["n"])
    plt.title("Most common sections in strict AI corpus")
    plt.xlabel("Number of articles")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_top_sections_strict.png", dpi=200)
    plt.close()
    return top_sections


def figure_section_share(strict: pd.DataFrame, top_sections: pd.DataFrame, fig_dir: Path) -> None:
    section_month = (
        strict.dropna(subset=["month", "section_name"])
        .groupby(["month", "section_name"])
        .size()
        .reset_index(name="n")
    )
    if section_month.empty:
        return
    section_month["share"] = section_month.groupby("month")["n"].transform(lambda s: s / s.sum())
    top_names = top_sections["section_name"].head(6).tolist()
    section_month_top = section_month[section_month["section_name"].isin(top_names)].copy()
    plt.figure(figsize=(10, 5.2))
    for sec in top_names:
        tmp = section_month_top[section_month_top["section_name"] == sec]
        plt.plot(tmp["month"], tmp["share"], marker="o", linewidth=1.4, label=sec)
    plt.title("Section share over time in strict AI corpus")
    plt.xlabel("")
    plt.ylabel("Share of monthly strict-corpus articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure3_section_share_over_time_strict.png", dpi=200)
    plt.close()


def make_manual_audit_sample(articles: pd.DataFrame, strict: pd.DataFrame, audit_n: int, seed: int) -> pd.DataFrame:
    strict_ids = set(strict["id"].dropna().astype(str))
    broad = articles.copy()
    broad["id_str"] = broad["id"].astype(str)
    broad["corpus_group"] = np.where(broad["id_str"].isin(strict_ids), "strict", "broad_not_strict")

    sample_frames = []
    rng = np.random.default_rng(seed)
    for group in ["strict", "broad_not_strict"]:
        dfg = broad[broad["corpus_group"] == group].copy()
        n = min(audit_n, len(dfg))
        if n == 0:
            continue
        idx = rng.choice(dfg.index.to_numpy(), size=n, replace=False)
        sample_frames.append(dfg.loc[idx].copy())
    audit = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    keep_cols = [
        "corpus_group", "id", "publication_date", "month", "section_name", "web_title",
        "headline", "trail_text", "web_url", "ai_keyword_hit", "ai_text_hit",
    ]
    for col in keep_cols:
        if col not in audit.columns:
            audit[col] = ""
    audit = audit[keep_cols].copy()
    audit["ai_keyword_hit"] = audit["id"].isin(strict.loc[strict["ai_keyword_hit"], "id"])
    audit["ai_text_hit"] = audit["id"].isin(strict.loc[strict["ai_text_hit"], "id"])
    audit.insert(0, "audit_id", range(1, len(audit) + 1))
    audit["reviewer_label"] = ""
    audit["reviewer_notes"] = ""
    audit["allowed_labels"] = "core_ai_article | incidental_ai_mention | unclear"
    return audit


def dataset_summary(broad: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    broad_n = int(broad["id"].nunique())
    strict_n = int(strict["id"].nunique())
    return pd.DataFrame({
        "metric": [
            "broad_unique_articles",
            "strict_unique_articles",
            "strict_share_of_broad",
            "months",
            "technology_business_articles_in_strict",
            "broader_public_issue_articles_in_strict",
        ],
        "value": [
            broad_n,
            strict_n,
            strict_n / broad_n if broad_n else np.nan,
            broad["month"].nunique(),
            int(strict["tech_business_frame"].sum()) if "tech_business_frame" in strict.columns else np.nan,
            int((strict["tech_business_frame"] == 0).sum()) if "tech_business_frame" in strict.columns else np.nan,
        ],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a corpus and aggregate results from a local CSV.")
    parser.add_argument("--input", default="data/raw/guardian_ai_articles_broad.csv")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--audit-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=370)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Missing local corpus: {input_path}. See data/README.md.", file=sys.stderr)
        return 2
    data_dir, fig_dir, outputs_dir = map(Path, (args.data_dir, args.fig_dir, args.outputs_dir))
    for directory in (data_dir, fig_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    articles = normalize_articles(pd.read_csv(input_path))
    strict = add_features(articles)
    if strict.empty:
        print("Strict corpus is empty. Check input data and AI relevance rules.", file=sys.stderr)
        return 3
    save_csv_with_lists(strict, data_dir / "guardian_ai_articles.csv")
    counts = make_monthly_counts(articles, strict)
    counts.to_csv(outputs_dir / "monthly_counts.csv", index=False)
    figure_monthly_counts(counts, fig_dir)
    top_sections = figure_top_sections(strict, fig_dir)
    top_sections.to_csv(outputs_dir / "top_sections_strict.csv", index=False)
    figure_section_share(strict, top_sections, fig_dir)
    section_month = strict.groupby(["month", "section_name"]).size().reset_index(name="n")
    section_month["month_total"] = section_month.groupby("month")["n"].transform("sum")
    section_month["share"] = section_month["n"] / section_month["month_total"]
    section_month.to_csv(outputs_dir / "section_share_by_month.csv", index=False)
    audit_path = data_dir / "audit_sample.csv"
    if not audit_path.exists():
        make_manual_audit_sample(articles, strict, args.audit_n, args.seed).to_csv(audit_path, index=False)
    summary = dataset_summary(articles, strict)
    summary.to_csv(outputs_dir / "dataset_summary.csv", index=False)
    print("Corpus preparation complete.")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
