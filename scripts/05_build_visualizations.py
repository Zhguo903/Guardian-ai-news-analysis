#!/usr/bin/env python3
"""
Create interactive Plotly visualizations.

Outputs:
  assets/viz_monthly_volume.html
  assets/viz_section_share.html
  assets/viz_topic_prevalence.html
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px


def read_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing aggregate data: {path}. Restore the saved outputs or rebuild the analysis."
        )
    df = pd.read_csv(path)
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing_columns))}.")
    if df.empty:
        raise ValueError(f"{path} contains no data to plot.")
    return df


def monthly(counts_file: Path, assets: Path) -> None:
    out = assets / "viz_monthly_volume.html"
    counts = read_table(counts_file, {"month", "broad_articles", "strict_articles"})
    counts["month"] = pd.to_datetime(counts["month"], errors="coerce")
    value_vars = [c for c in ["broad_articles", "strict_articles"] if c in counts.columns]
    long = counts.melt(id_vars="month", value_vars=value_vars, var_name="corpus", value_name="article_count")
    long["corpus"] = long["corpus"].replace({"broad_articles": "Broad corpus", "strict_articles": "Strict corpus"})
    fig = px.line(long, x="month", y="article_count", color="corpus", markers=True,
                  title="Monthly AI-related Guardian article volume",
                  labels={"month": "Month", "article_count": "Unique articles", "corpus": "Corpus"})
    fig.update_layout(hovermode="x unified")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)


def section_share(section_file: Path, assets: Path, top_sections: int) -> None:
    out = assets / "viz_section_share.html"
    df = read_table(section_file, {"month", "section_name", "n", "month_total"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["month"]).copy()
    df["section_name"] = df["section_name"].fillna("Unknown")
    df["n"] = pd.to_numeric(df["n"], errors="raise")
    df["month_total"] = pd.to_numeric(df["month_total"], errors="raise")
    if df.empty or (df["month_total"] <= 0).any():
        raise ValueError(f"{section_file} needs valid months and positive monthly totals.")
    if df.groupby("month")["month_total"].nunique().gt(1).any():
        raise ValueError(f"{section_file} has inconsistent totals within the same month.")
    top = df.groupby("section_name")["n"].sum().nlargest(top_sections).index.tolist()
    # The denominator describes all sections, before selecting the displayed sections.
    totals = df.groupby("month", as_index=False)["month_total"].first()
    index = pd.MultiIndex.from_product(
        [totals["month"].sort_values(), top], names=["month", "section_name"]
    )
    sm = (
        df.groupby(["month", "section_name"])["n"].sum()
        .reindex(index, fill_value=0).reset_index()
    )
    sm = sm.merge(totals, on="month", how="left")
    sm["share"] = sm["n"] / sm["month_total"]
    fig = px.line(sm, x="month", y="share", color="section_name", markers=True,
                  title="Section composition over time in the strict AI corpus",
                  labels={"month": "Month", "share": "Share of strict-corpus articles", "section_name": "Section"})
    fig.update_layout(hovermode="x unified")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)



def clean_topic_label(label: str) -> str:
    if pd.isna(label):
        return "Unknown topic"

    label = str(label).strip()
    lower = label.lower()

    direct_map = {
        "AI tools and ChatGPT": "AI tools & ChatGPT",
        "Topic 2: chatgpt, images, just": "ChatGPT & image generation",
        "Topic 3: openai, musk, media": "OpenAI, Musk & media",
        "Topic 4: intelligence, artificial, artificial intelligence": "General AI discourse",
        "Topic 5: nvidia, company, openai": "AI business competition",
        "Education and work": "Education & work",
    }

    if label in direct_map:
        return direct_map[label]

    if "nvidia" in lower or "company" in lower or "business" in lower:
        return "AI business competition"
    if "education" in lower or "school" in lower or "student" in lower or "work" in lower:
        return "Education & work"
    if "openai" in lower and ("musk" in lower or "media" in lower):
        return "OpenAI, Musk & media"
    if "chatgpt" in lower and ("image" in lower or "images" in lower):
        return "ChatGPT & image generation"
    if "chatgpt" in lower:
        return "AI tools & ChatGPT"
    if "intelligence" in lower and "artificial" in lower:
        return "General AI discourse"
    if "copyright" in lower or "artist" in lower or "music" in lower or "film" in lower:
        return "Creative work & copyright"
    if "regulation" in lower or "government" in lower or "law" in lower or "policy" in lower:
        return "Regulation & policy"
    if "risk" in lower or "misinformation" in lower or "election" in lower:
        return "Risk & misinformation"

    if len(label) > 36:
        return label[:33].rstrip() + "..."

    return label


def topics(topic_file: Path, assets: Path) -> None:
    out = assets / "viz_topic_prevalence.html"
    df = read_table(topic_file, {"month", "topic_id", "topic_share", "topic_label"})

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["topic_share"] = pd.to_numeric(df["topic_share"], errors="raise")
    df = df.dropna(subset=["month", "topic_id", "topic_share", "topic_label"]).copy()
    if df.empty:
        raise ValueError(f"{topic_file} contains no valid topic observations.")
    if df.duplicated(["topic_id", "month"]).any():
        raise ValueError(f"{topic_file} has more than one observation per topic and month.")

    df["topic_label_short"] = df["topic_label"].apply(clean_topic_label)
    # Keep distinct topics separate even when shortened display names coincide.
    label_counts = df[["topic_id", "topic_label_short"]].drop_duplicates()["topic_label_short"].value_counts()
    repeated = df["topic_label_short"].map(label_counts).gt(1)
    df.loc[repeated, "topic_label_short"] = (
        df.loc[repeated, "topic_label_short"] + " (topic "
        + df.loc[repeated, "topic_id"].astype(str) + ")"
    )

    df = df.sort_values(["topic_id", "month"]).copy()
    df["topic_share_smoothed"] = (
        df.groupby("topic_id")["topic_share"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    topic_order = (
        df.groupby("topic_label_short")["topic_share_smoothed"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig = px.line(
        df,
        x="month",
        y="topic_share_smoothed",
        color="topic_label_short",
        line_group="topic_id",
        category_orders={"topic_label_short": topic_order},
        title="LDA topic prevalence over time",
        labels={
            "month": "Month",
            "topic_share_smoothed": "Average topic share, 3-month rolling mean",
            "topic_label_short": "Topic",
        },
        hover_data={
            "topic_label": True,
            "topic_label_short": False,
            "topic_share": ":.3f",
            "topic_share_smoothed": ":.3f",
            "month": "|%Y-%m",
        },
    )

    fig.update_traces(
        mode="lines",
        line=dict(width=3),
        opacity=0.9,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        height=700,
        margin=dict(l=80, r=40, t=80, b=155),
        legend=dict(
            title="Topic",
            orientation="h",
            yanchor="top",
            y=-0.13,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(title_text="Month", title_standoff=22)
    fig.update_yaxes(
        title_text="Average topic share",
        tickformat=".0%",
        range=[0, min(1.0, max(0.1, float(df["topic_share_smoothed"].max()) * 1.10))],
        title_standoff=22,
        automargin=True
    )

    fig.write_html(
        out,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--section-shares", default="outputs/section_share_by_month.csv")
    p.add_argument("--monthly-counts", default="outputs/monthly_counts.csv")
    p.add_argument("--topic-prevalence", default="outputs/topic_prevalence_by_month.csv")
    p.add_argument("--assets-dir", "--docs-dir", dest="assets_dir", default="assets")
    p.add_argument("--top-sections", type=int, default=8)
    args = p.parse_args()
    if args.top_sections < 1:
        p.error("--top-sections must be a positive integer.")

    assets = Path(args.assets_dir)
    assets.mkdir(exist_ok=True, parents=True)
    try:
        monthly(Path(args.monthly_counts), assets)
        section_share(Path(args.section_shares), assets, args.top_sections)
        topics(Path(args.topic_prevalence), assets)
    except (FileNotFoundError, ValueError) as exc:
        p.error(str(exc))

    print("Interactive visualization files created:")
    for pth in ["viz_monthly_volume.html", "viz_section_share.html", "viz_topic_prevalence.html"]:
        print(f"  - {assets / pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
