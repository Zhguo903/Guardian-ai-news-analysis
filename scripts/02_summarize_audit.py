#!/usr/bin/env python3
"""
Summarize saved article-relevance audit labels.

The public audit_labels.csv contains labels and article identifiers only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALLOWED_LABELS = {"core_ai_article", "incidental_ai_mention", "unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual audit labels.")
    parser.add_argument("--audit-file", default="data/audit_labels.csv")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--outputs-dir", default="outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_path = Path(args.audit_file)
    fig_dir = Path(args.fig_dir)
    outputs_dir = Path(args.outputs_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not audit_path.exists():
        fallback = Path("data/raw/audit_sample.csv")
        if fallback.exists():
            print(
                f"Could not find {audit_path}. I found {fallback}, but reviewer_label is probably blank. "
                "Fill reviewer_label and save as data/audit_labels.csv first."
            )
        else:
            print(f"Could not find {audit_path} or data/raw/audit_sample.csv.")
        return 2

    audit = pd.read_csv(audit_path)
    required = {"corpus_group", "reviewer_label"}
    missing = required - set(audit.columns)
    if missing:
        raise RuntimeError(f"Audit file is missing required columns: {sorted(missing)}")

    audit["reviewer_label"] = audit["reviewer_label"].fillna("").astype(str).str.strip()
    incomplete = audit["reviewer_label"].eq("").sum()
    invalid = sorted(set(audit.loc[audit["reviewer_label"].ne("") & ~audit["reviewer_label"].isin(ALLOWED_LABELS), "reviewer_label"]))
    if invalid:
        print(f"Invalid labels found: {invalid}")
        print(f"Allowed labels: {sorted(ALLOWED_LABELS)}")
        return 3
    if incomplete:
        print(f"Warning: {incomplete} rows have blank reviewer_label and will be excluded from proportions.")

    labeled = audit[audit["reviewer_label"].isin(ALLOWED_LABELS)].copy()
    if labeled.empty:
        print("No completed labels found.")
        return 4

    summary = (
        labeled.groupby(["corpus_group", "reviewer_label"])
        .size()
        .reset_index(name="n")
    )
    totals = labeled.groupby("corpus_group").size().reset_index(name="total_labeled")
    summary = summary.merge(totals, on="corpus_group", how="left")
    summary["proportion"] = summary["n"] / summary["total_labeled"]
    summary.to_csv(outputs_dir / "manual_audit_summary.csv", index=False)

    precision = (
        labeled.assign(is_core_ai=labeled["reviewer_label"].eq("core_ai_article").astype(int))
        .groupby("corpus_group", as_index=False)
        .agg(
            labeled_articles=("reviewer_label", "size"),
            core_ai_articles=("is_core_ai", "sum"),
            core_ai_share=("is_core_ai", "mean"),
        )
    )
    precision.to_csv(outputs_dir / "manual_audit_core_ai_share.csv", index=False)

    plot_df = precision.sort_values("corpus_group")
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(plot_df["corpus_group"], plot_df["core_ai_share"])
    plt.ylim(0, min(1.0, max(0.2, float(plot_df["core_ai_share"].max()) + 0.15)))
    plt.title("Manual audit: share labeled as core AI article")
    plt.xlabel("")
    plt.ylabel("Share of labeled audit sample")
    for i, row in plot_df.reset_index(drop=True).iterrows():
        label = f'{row["core_ai_share"]:.2f}\n(n={int(row["labeled_articles"])})'
        plt.text(i, row["core_ai_share"] + 0.02, label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure0_manual_audit_precision.png", dpi=200)
    plt.close()

    print("Manual audit summary complete.")
    print(precision.to_string(index=False))
    print("Created:")
    print(f"  - {outputs_dir / 'manual_audit_summary.csv'}")
    print(f"  - {outputs_dir / 'manual_audit_core_ai_share.csv'}")
    print(f"  - {fig_dir / 'figure0_manual_audit_precision.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
