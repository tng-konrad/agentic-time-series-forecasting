# Module 1 implementation record

Built 23 October 2026 from the [revised plan](module-01-plan.md). The [notebook](../a01-forecasting-workflow.ipynb) and [Substack post draft](../a01-post.md) are implemented. The article remains a draft for editorial review; nothing was published.

## Delivered

- `agentic_ts/forecasting.py`: validated 16-series panel, five fixed candidates, three chronological folds, per-series selection, frozen final forecasts, failure handling, MASE and portfolio summaries.
- `agentic_ts/workflow.py`: deterministic template, one-call control, brief-to-report chain, immutable evidence, strict JSON contracts, one-attempt local Ollama adapter, matching replay, and injected unknown-reference example.
- `data/a01/abs_retail_panel.csv` and `manifest.json`: 4,896 observations with exact source IDs, source/checksum provenance, labels, frequency, units, and extraction command.
- `scripts/prepare_a01_data.py`: reproducible extraction using only Python's standard library. The XLSX reader was kept dependency-free because the pinned workbook needs only value/metadata extraction; no Excel authoring is involved.
- `scripts/capture_a01.py`: preserves exact prompts, schemas, responses, model digest/settings, server version, token counts, timings, and failures; existing captures cannot be overwritten by a new invocation.
- `scripts/execute_a01.py`: fresh-kernel execution with the invoking Python runtime.
- `examples/a01/numerical_run.json`, `recorded_run.json`, and [report audit](../examples/a01/report_audit.md): full numerical evidence, genuine live captures, and documented semantic inspection.
- `graphs/atsf01-graph1.png` through `atsf01-graph3.png`, and `graphs/atsf01-table1.png` through `atsf01-table9.png`: every graph and table, numbered separately in notebook order. The post contains explicit filename placeholders and a complete asset index.
- Graphs use the existing notebooks' `seaborn-v0_8` style, 20 × 10 default figure size, and seed 42. Table images contain only column headers and data, with no title or footer. Decimal results use two places, including numbers embedded in the report comparison; the source results and captured responses retain their original precision.
- Focused standard-library unit tests and README instructions.

## Numerical result

Source: the pinned June 2025 ABS Table 11 workbook, retrieved 22 October 2026. Four states × four industries; January 2000–June 2025. No imputation, shock removal, or post-score changes to membership/windows.

| Measure | Result |
|---|---:|
| Validation evaluations | 240 |
| Candidate/series combinations ineligible | 1 |
| Selected `hw_mul` | 9 |
| Selected `hw_mul_damped` | 4 |
| Selected `hw_add` | 1 |
| Selected `hw_add_damped` | 1 |
| Selected `seasonal_naive` | 1 |
| Final-refit fallbacks | 0 |
| Delivered macro-average MASE | 0.5301732997 |
| Seasonal-naive macro-average MASE | 0.8063839634 |
| Final wins / ties / losses | 10 / 1 / 5 |
| Initial numerical runtime | 7.765 seconds |

`hw_mul` failed to converge on one validation fold for Queensland clothing (`A3349477X`). All three fold attempts were retained, and that candidate was excluded for this series. The eligible damped variant won selection. No series was dropped.

Model selection used mean validation RMSE per series. Final evaluation used July 2024–June 2025. The portfolio headline uses equal-weight seasonal MASE, with denominators fitted only to development history. Forecast choices were frozen before scoring the final outcomes. A fixed historical vintage is not a real-time-vintage backtest.

## Genuine language capture and findings

The installed local `qwen3-coder-next:q4_K_M` model ran through Ollama 0.33.2. Model digest: `ca06e9e4087c714d44355bf954099187890e63084b4a632b8e9956c4b9492074`. Configuration: temperature 0, seed 42, context 32,768, output limit 4,096, 240-second request timeout. There was one attempt per planned stage and no automatic retries or repairs. Capture order: A, B, C brief, C report.

| Stage | Wall seconds | Prompt tokens | Output tokens |
|---|---:|---:|---:|
| B report | 34.584 | 6,356 | 540 |
| C brief | 16.616 | 6,313 | 681 |
| C report | 10.442 | 6,947 | 544 |

B's server metadata records about 19.47 seconds of model loading. C used a warm model and overlapping context. These timings do not support a controlled architectural speed comparison. No dollar costs or productivity benefits are inferred.

Both B and C passed the structural contract. Both confuse MASE and MAE in their Queensland hardware wording. C propagates the wording from its brief and omits the requested final win/tie/loss counts. The original responses are retained. “Successful live example” here means successful execution and structurally valid output; the semantic defects are explicit findings, not claims of correct reporting. The source facts are sufficient to identify them.

The implementing AI agent inspected the reports against their cited evidence and documented the defects. **Independent human editorial review remains pending**. This is one portfolio reporting case, with no repeated-run reliability estimate or reader study.

## Verification

- Reproduced the 16 IDs and 4,896 values from the checksum-pinned official workbook, using the dependency-free extractor; validated dates, labels, units, positivity, and uniqueness.
- Ran the focused unit suite covering calendar boundaries, seasonal-naive repetition, hand-calculated MASE, complete-fold eligibility, selection ties, refit fallback, complete-panel macro averages, contract rejection, immutable evidence, skipped stages, and replay invalidation.
- Executed all 19 code cells from a fresh Python kernel in default replay mode after the 24 October Setup/Utils reorganization. The numerical experiment was recomputed; B and C replay matched the archived inputs and returned their captured reports. No live model request is made in this mode.
- Generated and visually inspected the original publication figures for labels, data, layout, and clipping. The 23 October export revision extends this to all three graphs and nine tables, including development histories, source metadata, validation/candidate tables, full results, failures, report comparison, stage trace, and audit.
- Inspected the genuine live call records, verified the article's main numeric claims, and preserved captured failures/semantic errors.

Python 3.12.8; NumPy 2.4.6; pandas 3.0.3; statsmodels 0.14.6. The numerical JSON records SciPy and numerical package versions; the existing `uv.lock` was unchanged. Cross-platform optimizer differences can change evidence and cause replay to fall back visibly, as intended.

## Running and extending

From the repository root after `uv sync --locked`:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_a01*.py' -v
.venv/bin/python -m scripts.execute_a01
```

The executed notebook defaults to replay. Setup gathers imports and a `CFG` class containing paths, display/export settings, reporting mode, and the local model configuration. Utils contains the export, table-formatting, and diagram helpers. For new live output, run Ollama locally, switch `CFG.run_mode` to `"live"`, and select the installed model in `CFG.llm`. Live notebook execution does not replace the archival record. A new capture uses `python -m scripts.capture_a01 --output examples/a01/another_run.json` and the saved numerical run. To capture a changed numerical experiment, first explicitly save that experiment to a new numerical record and adapt the capture input; do not silently pair new narratives with old data.

The plan's initial draft and original independent critique were preserved. The only revision to its status is the link to these delivered artifacts and the distinction between structural success and semantic correctness.
