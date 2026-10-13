#!/usr/bin/env python3
"""
Generate report_text/generated_results.md and generated_tables.md from CSV outputs.
This keeps report.qmd free of code chunks while still using actual model results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def get_metric(summary: pd.DataFrame, metric: str) -> float:
    if summary.empty or "metric" not in summary.columns:
        return np.nan
    vals = summary.loc[summary["metric"] == metric, "value"]
    return float(vals.iloc[0]) if len(vals) else np.nan


def pct(x, digits=1) -> str:
    return "NA" if pd.isna(x) else f"{100 * float(x):.{digits}f}%"


def num(x, digits=3) -> str:
    if pd.isna(x):
        return "NA"
    if float(x).is_integer():
        return f"{int(x):,}"
    return f"{float(x):.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = [list(df.columns), ["---"] * len(df.columns), *df.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--report-text-dir", default="report_text")
    args = p.parse_args()

    outputs = Path(args.outputs_dir)
    report_text = Path(args.report_text_dir)
    report_text.mkdir(exist_ok=True, parents=True)

    corpus_summary = read_optional(outputs / "dataset_summary.csv")
    audit = read_optional(outputs / "manual_audit_core_ai_share.csv")
    perf = read_optional(outputs / "model_performance.csv")
    if not perf.empty:
        perf = perf.sort_values("f1", ascending=False).reset_index(drop=True)
    topics = read_optional(outputs / "topic_keywords.csv")

    broad_n = get_metric(corpus_summary, "broad_unique_articles")
    strict_n = get_metric(corpus_summary, "strict_unique_articles")
    strict_share = get_metric(corpus_summary, "strict_share_of_broad")
    months = get_metric(corpus_summary, "months")
    tech_n = get_metric(corpus_summary, "technology_business_articles_in_strict")
    public_n = get_metric(corpus_summary, "broader_public_issue_articles_in_strict")

    corpus_table = pd.DataFrame({
        "Metric": [
            "Broad corpus articles", "Strict corpus articles", "Strict share of broad corpus",
            "Months covered", "Technology/Business section articles", "Articles in other sections"
        ],
        "Value": [num(broad_n), num(strict_n), pct(strict_share, 2), num(months), num(tech_n), num(public_n)]
    })

    audit_sentence = ""
    audit_table = pd.DataFrame()
    if not audit.empty:
        audit_table = audit.copy()
        audit_table["corpus_group"] = audit_table["corpus_group"].replace({"broad_not_strict": "Broad but not strict", "strict": "Strict corpus"})
        audit_table = audit_table.rename(columns={
            "corpus_group": "Audit group",
            "labeled_articles": "Labeled articles",
            "core_ai_articles": "Core AI articles",
            "core_ai_share": "Core AI share",
        })
        audit_table["Core AI share"] = audit_table["Core AI share"].map(lambda x: pct(x, 1))
        strict_vals = audit.loc[audit["corpus_group"].eq("strict"), "core_ai_share"]
        broad_vals = audit.loc[audit["corpus_group"].eq("broad_not_strict"), "core_ai_share"]
        if len(strict_vals) and len(broad_vals):
            audit_sentence = (
                f"The saved relevance audit labeled {pct(strict_vals.iloc[0], 1)} of sampled strict-corpus articles as core AI, "
                f"compared with {pct(broad_vals.iloc[0], 1)} of sampled broad-but-not-strict articles. "
                "This supports using the strict corpus as the main dataset for modeling."
            )

    perf_sentence = ""
    perf_table = pd.DataFrame()
    if not perf.empty:
        perf_table = perf[[c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"] if c in perf.columns]].copy()
        perf_table = perf_table.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"})
        for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if c in perf_table.columns:
                perf_table[c] = perf_table[c].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.3f}")
        best = perf.iloc[0]
        perf_sentence = (
            f"The highest observed test-set F1 model was **{best['model']}**, "
            f"with accuracy {float(best['accuracy']):.3f}, F1 {float(best['f1']):.3f}, "
            f"and ROC-AUC {float(best['roc_auc']):.3f}."
            if pd.notna(best.get("roc_auc", np.nan)) else
            f"The highest observed test-set F1 model was **{best['model']}**, "
            f"with accuracy {float(best['accuracy']):.3f} and F1 {float(best['f1']):.3f}."
        )

    topic_sentence = ""
    topic_table = pd.DataFrame()
    if not topics.empty:
        rows = []
        for label, grp in topics.groupby("topic_label", sort=False):
            rows.append({"Topic label": label, "Top terms": ", ".join(grp.sort_values("rank")["term"].head(8).astype(str).tolist())})
        topic_table = pd.DataFrame(rows)
        topic_sentence = "The six-topic LDA model was summarized using heuristic labels: " + "; ".join(topic_table["Topic label"].astype(str).head(6).tolist()) + "."

    generated_results = f"""## Results summary

The saved search strategy produced **{num(broad_n)}** broad-corpus articles and **{num(strict_n)}** strict-corpus articles across **{num(months)}** months. The strict corpus represented **{pct(strict_share, 2)}** of the broad corpus. Within the strict corpus, **{num(tech_n)}** articles appeared in Technology/Business sections and **{num(public_n)}** articles appeared in other sections.

{audit_sentence}

{perf_sentence}

{topic_sentence}
"""
    (report_text / "generated_results.md").write_text(generated_results, encoding="utf-8")

    table_parts = ["### Table 1. Corpus construction summary\n", md_table(corpus_table)]
    if not audit_table.empty:
        table_parts += ["\n\n### Table 2. Manual audit summary\n", md_table(audit_table)]
    if not perf_table.empty:
        table_parts += ["\n\n### Table 3. Classification model performance on the test set\n", md_table(perf_table)]
    (report_text / "generated_tables.md").write_text("\n".join(table_parts), encoding="utf-8")

    topic_md = "### LDA topic labels and top terms\n\n" + md_table(topic_table) + "\n" if not topic_table.empty else ""
    (report_text / "generated_topic_table.md").write_text(topic_md, encoding="utf-8")

    print("Generated report snippets:")
    print(f"  - {report_text / 'generated_results.md'}")
    print(f"  - {report_text / 'generated_tables.md'}")
    print(f"  - {report_text / 'generated_topic_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
