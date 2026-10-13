# Data and provenance

## Public files

`audit_labels.csv` retains the 200 saved relevance labels and Guardian article IDs, without headlines, trail text, or reviewer notes. It has four columns: `audit_id`, `corpus_group` (`strict` or `broad_not_strict`), `id`, and `reviewer_label` (`core_ai_article`, `incidental_ai_mention`, or `unclear`). Labels come from the existing audit snapshot; no independent reviewer verification is recorded.

`../outputs/` contains corpus counts, monthly section shares, audit summaries, model metrics, feature importance, topic keywords, and monthly topic prevalence. Raw text and per-article model outputs are excluded from the public repository. The committed figures and model summaries are the saved analytical snapshot; rebuilding the website does not retrain models.

## Local analytical input

Full analysis uses a local CSV at `raw/guardian_ai_articles_broad.csv`. Supply a snapshot you are authorized to use for this analysis. The repository does not download article content or distribute the original text corpus. The original snapshot contains 40,232 unique articles, dated January 2021 through December 2025. After filtering, 3,120 articles remain.

Required columns are `id`, `publication_date` (ISO date), `headline`, `trail_text`, and `section_name`. To match the saved analysis, also supply `section_id`, `web_url`, `web_title`, `wordcount`, `keywords`, and `format_tags`. `keywords` and `format_tags` may be semicolon-separated strings or serialized lists. Missing tag/word-count fields receive empty or missing defaults and can change the strict corpus and model results. `month` and `year` can be derived from `publication_date`.

`01_build_corpus.py` writes the strict corpus to `raw/guardian_ai_articles.csv` and a new audit sample to `raw/audit_sample.csv`. It does not overwrite an existing sample. Completed labels for a different corpus should be passed explicitly to `02_summarize_audit.py --audit-file PATH`; the committed labels belong to the original snapshot. Per-article predictions and topic weights are written under `derived/`. Both `raw/` and `derived/` are ignored by Git.

## Historical collection strategy

The original corpus was retrieved from the [Guardian Content API](https://open-platform.theguardian.com/documentation/) in monthly windows for 2021–2025, deduplicating on article ID. Search terms were: artificial intelligence, machine learning, generative AI, large language model, ChatGPT, deep learning, neural network, LLM, OpenAI, Google Gemini, Google DeepMind, Claude AI, DALL-E, and Midjourney. The broad search did not query standalone “AI”.

The strict filter matches AI-related keyword tags or direct signals in headline/trail text. Standalone “AI” requires contextual terms. The implementation preserves the historical matching rules; it is a relevance heuristic, not a verified gold standard.

## Source rights

Guardian article content remains subject to its [Open Platform terms](https://www.theguardian.com/open-platform/terms-and-conditions). Access alone does not establish permission for analytical reuse or redistribution. This repository grants no rights to Guardian content.
