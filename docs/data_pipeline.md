# Data pipeline (Sumana)

This pipeline covers ONLY Sumana's responsibilities from the project guide: collecting and tagging PUBLIC university facts, tracking source URLs and collection dates, maintaining the campus-map update workflow, freezing a leakage-resistant test set, and comparing snapshots of changing facts. It does not train models or implement the chatbot.


## Quick start (from the repository root; Python 3.10+, no third-party packages)

```bash
python3 scripts/pipeline.py validate                          # rules below; 0 errors required
python3 scripts/pipeline.py split --version v0 --test-size 300  # train/test, stratified by tier
python3 scripts/pipeline.py snapshot --version v0              # freeze verified TERMLY pairs (D5 baseline)
python3 scripts/pipeline.py snapshot --version v1              # after week 13-14 re-verification
python3 scripts/pipeline.py compare --before v0 --after v1     # which TERMLY answers changed
python3 -m unittest discover -s tests -v
```

(Windows: `py` instead of `python3`, backslashes in paths.)

## The dataset: `data/raw/pairs.csv` and `data/raw/sources.csv`

Field names follow CLAUDE.md §6. These two files are the source of truth; the `data/*.csv` drafts
from 22 September are kept as history, and `scripts/migrate_drafts.py` (one-shot, refuses to re-run)
records how they were moved across.

`pairs.csv`: `id,group_id,instruction,response,tier,source_id,source_url,captured_at,verified_by,verified_at,legacy_id,notes`

- `tier` is `STATIC` or `TERMLY`. PRIVATE content never enters the file (D4).
  TERMLY = expires within a term or admission cycle: exam, registration and break dates, holidays,
  fee deadlines, event cards, the current admissions cycle. Leadership, contact details and
  headcounts drift annually at most and are STATIC.
- `group_id` ties paraphrases of one fact together; the split never separates a group. All rows of
  a group share a tier.
- `source_id` may list several registered sources separated by `;`.
- `legacy_id` is the draft `qa_id` (QA001...), so `qa_manual_review.csv` still lines up.
- **Verifying a pair** = check it against the source, correct it, then fill `verified_by` and
  `verified_at` together. `captured_at` is the date the source was read; for the 175 migrated drafts
  it is unknown, so the verifier sets it to the date they checked the live source.
  Only verified pairs reach `split` and `snapshot` (§12 rule 3).

`sources.csv`: `source_id,title,url,source_type,captured_at,public_access,notes`. `url` is http(s)
or an `assets/` path (campus-map photographs, the calendar PDF).

`validate` rejects: duplicate/missing IDs, unknown tier, unregistered source, half-filled
verification, verified pairs without `source_url`/`captured_at`, mojibake (`Ã`, `â€`), and text
matching the private-data screen. The privacy screen is a hard error, but it is a keyword list, so
human privacy review is still required.

## Versions

`split` and `snapshot` write to `data/<version>/` (e.g. `data/v0/`) and refuse to overwrite an
existing version. The split is stratified: each tier gets its proportional share of the test set,
so TERMLY facts cannot vanish from it. Give the training owner `train.jsonl` only; `test.jsonl` goes
to evaluation and is never trained or tuned on.


## UPDATED: supplied campus-map photographs
Two user-supplied photos are included in `assets/`. `data/raw/campus_map_observations.csv` contains 19 visible labels/sign directions, all marked `pending_university_confirmation`. This is a **transcription of the pictured sign**, not an official updated map. The directional minutes are from the sign location only. Do not use the observations as verified QA training facts until the map owner signs off. See `docs/campus_map_review.md`.

## Crawl -> clean corpus (added 22 September)

`scripts/mucrawl.py` deep-crawls the public site with crawl4ai (headless Chrome) to `mahindra_complete.json`.
That file is the RAW capture and is kept as provenance — do not generate QA pairs from it directly.

```bash
python3 scripts/mucrawl.py                                # raw capture, ~50 pages
python3 scripts/clean_crawl.py mahindra_complete.json
```

`clean_crawl.py` does three things and prints exactly what it dropped:

1. **Drops blocked hosts.** `muerp.mahindrauniversity.edu.in` is the Juno-powered student ERP. Juno
   is out of the project (CLAUDE.md §12 rule 6) and that login URL also carried a live `jsessionid`.
   `scripts/mucrawl.py` now restricts the crawl to `www.mahindrauniversity.edu.in` so it cannot be collected
   again; `clean_crawl.py` is the second line of defence for any crawl taken before that fix.
2. **Strips boilerplate** — any line appearing on 80%+ of pages. On this site that is site nav,
   footer and social links: **71% of the captured bytes**, and it is what put an ERP login link on
   all 50 pages.
3. **Stamps `captured_at`** on every page, which §6 requires and the raw crawl did not carry.

Outputs `data/raw/crawl_clean.json` (49 pages, boilerplate removed) and `data/raw/crawl_pdfs.csv`
(32 linked PDFs with the page each was found on).

**The PDFs are where the term-expiring facts live.** The `/calendar/` page contains no dates at all
— it links to `Academic-Calendar-2026-2027.pdf`, now saved in `assets/` and registered as `SRC-P-01`
with 50 draft pairs (`mu-000176` onward, unverified). The PDF has three date typos, recorded in those
pairs' `notes`. Next: the 2026-27 admission notifications. Ignore the 2020–2025 brochures,
superseded calendars and `covid-sop.pdf`.

Re-running either script is safe; both are deterministic and overwrite their own outputs.

