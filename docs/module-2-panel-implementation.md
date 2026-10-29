# Module 2 panel study implementation

Completed 27 October 2026, following the [panel-study plan](module-2-panel-study-plan.md).

[Executed notebook](../a02-routing-panel.ipynb) · [Companion article](../a02-post-new.md) · [Publication verification](../examples/module02_panel/publication_verification.json).

## Scope and result

The companion expands numerical routing evaluation to 100 new item IDs in CA_1, TX_1 and WI_1, across six 28-day forecast origins. Three earlier development origins on CA_1/TX_1 choose a fixed-family comparator. It uses the original seasonal naive, damped additive Holt–Winters and TSB candidates, three internal validation blocks, 365-day fitting windows and unchanged 50% zero-frequency rule. It makes no new LLM calls and does not train a router.

There are 600 development cases and 1,800 main cases. The development-selected family is intermittent (I), with 597 scale-defined cases; three zero scales remain recorded as N/A. In main evaluation all six arms deliver all 1,800 cases and all main scales are positive. Macro MASE is A 1.091838, B 0.968754, R 0.969959, I/F 0.973965 and D 0.967190. B improves on F by about 0.54%; always-regular R is close to B. D has the lowest overall error. These are retrospective cohort results rather than general guarantees.

The paired B−F mean is −0.005211 MASE; its item-block resampling interval is approximately [−0.008841, −0.001817]. B−D is +0.001564, interval [−0.003798, +0.007254]. Resampling keeps item IDs' stores and dates together across 2,000 replicates, seed 42. Intervals condition on these locations/dates; shared calendar shocks remain and an interval containing zero does not establish equivalence.

The mean B−F MASE is negative in every store and at every aggregate origin, but individual store/origin combinations can lose. February aggregate MAE favors F even though MASE favors B. All cases, bands, dates and both metrics remain in the evidence. The threshold was not tuned after scoring.

## Data and protocol

`scripts.prepare_module02_panel` verified the raw validation sales archive and calendar against the original pinned hashes. It selected by catalog membership, department quotas and SHA-256 ranking, excluding the original twelve products, before extracting numeric target values. The protocol was written before numerical evaluation. No new download was needed.

The bundled extract contains 210,300 rows: 300 series × 701 days, 25 May 2014–24 April 2016. Reader execution needs only `data/module02_panel`, never the raw source path. The source and manifest hashes are checked before loading. Nonnegative integer counts, complete unique daily calendars, membership and the exact case grid are validated.

Main origins are 8 November and 6 December 2015; 3/31 January, 28 February and 27 March 2016. Every case receives only the 449 days ending at its origin. Past test outcomes can legitimately become a later origin's history, while F and the routing policy stay fixed. WI_1 contributes no error to selecting F. This is geographic robustness of an existing rule, not a learned-router transfer experiment.

## Implementation and work accounting

`agentic_ts/module02_panel.py` adds the panel loader, restricted views, exact-fit bank, pure arm reconstruction, freezing/scoring, development-only selection, paired masks, grouped analysis and uncertainty. It imports existing numerical routines. `module02_panel_publication.py` only renders evidence. The shared PNG exporter derives relative paths from destinations and accepts nested directories; default Module 1/2 output paths remain the same.

The bank keys exact training dates/values, method, numerical configuration and library versions. It serves 28,800 requested fits through 9,900 actual attempts and 18,900 cache hits. Six Holt–Winters fits fail convergence; their warnings/errors remain recorded. Those candidates are ineligible where needed and no main forecast needs a final numerical fallback.

Actual bank work is distinct from logical standalone work. Across main cases B requests 14,291 fits, D 19,697: 27.45% fewer. The 36-case uncached benchmark uses two predeclared items across all stores/dates, alternates B/D order and warms imports/fitting on development data. B includes profiling/dispatch. It makes 286 fits in 2.37317 seconds; D makes 394 in 4.69514 seconds in the archived author session. These times are a limited local sample, not a full-panel speed estimate. Every uncached forecast is checked against its bank-based counterpart.

The author's numerical fit times sum to 128.14 seconds; the complete author script takes 152.24 seconds including orchestration, scoring, serialization and the separate benchmark. These are different scopes. The notebook recomputes numerical fits and timings from a fresh kernel; publication embeds the stable archived author work table and labels its session. Reader measurements are saved separately.

Compressed JSON/gzip CSV reduce bundled record size; the compression does not remove full-precision values. Numerical selection, warnings, failures and predictions remain inspectable. Reader files and origin checkpoints are distinct from author records, whose creation scripts reject overwriting. Default execution starts with an empty bank, not the author's fit cache.

## Notebook, post and exports

The notebook has 31 code cells, each immediately followed by explanation. It covers seven substantive sections matching the companion article. A clean-kernel runner uses the project environment, a 1,200-second per-cell ceiling and temporary Jupyter/Matplotlib/Numba paths. The full grid is the default; no smoke-test substitution is used.

Five graphs and four rendered tables are saved under `graphs/module02-panel/`, independently numbered from one as `atsf02-graph<N>.png` / `atsf02-table<N>.png`. Every displayed notebook image is the saved PNG, reused by the post. Tables have no title/footer/caption; floats display exactly two decimals and integer counts retain integer formatting. Small effects are presented as ΔMASE × 1,000 in a labeled table block/figures so required rounding does not erase them. Full-precision scores remain in prose and records.

The original Module 2 notebook, capture and results remain the introductory study. README adds companion instructions, and its post has a short companion link with refreshed editorial binding. Historical numerical results are not rewritten.

## Reproduce and validate

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_module02_panel
.venv/bin/python -m unittest tests.test_module02_panel tests.test_module02_routing tests.test_a02_routing tests.test_module01 -v
.venv/bin/python -m scripts.verify_module02_panel_publication
.venv/bin/python -m scripts.verify_module02_publication
```

Validation covers source corruption, exact grids/calendars, per-case future/masked-history leakage, the distinction between future values for early versus late origins, final-horizon mutations, WI_1 exclusion from fixed selection, zero scales, empty comparisons, ties, failed fits/fallback, independent executor equivalence, exact cache invalidation, paired masks, item-block resampling and both nested/default export paths. Existing Module 1/2 routing tests remain part of regression verification.

The initial implementation passed all 60 relevant tests: 16 panel-study tests and 44 existing routing/Module 1 tests. The notebook completed 31 sequential code cells in a fresh kernel; its exact numerical hashes match the author run. The publication check independently validates 13,800 final forecast metric records, 21,600 internal validation folds, 149 rendered table cells and the article's evidence mappings. The original Module 2 publication verifier also checks the refreshed companion link. Test hashes and inspection records are saved in `examples/module02_panel/`.

The publication verifier checks full fresh numerical hashes, independently recalculates final and internal-validation errors from bundled values, verifies ranking/chronology, checks fixed-family provenance, reconciles uncached forecasts and fit counts, and validates headings, image references, section placement, PNG bytes, DPI, typed table formats and empirical claim mappings. All nine images receive native and 760-pixel visual inspection. Recorded verification files bind these checks to actual source and output hashes.

## Publication refresh: a02-post-new.md

On 27 October 2026, the full notebook was executed again from a clean kernel for [a02-post-new.md](../a02-post-new.md). All 31 code cells completed; the predictions, scores and fit-bank records reproduce the author's numerical hashes, with zero LLM calls. All five graphs and four tables were regenerated. Their PNG bytes reproduce the earlier exports, and their manifest now identifies the requested article filename. Every image was inspected again at native resolution and 760 pixels wide. The article has seven matching substantive headings and 56 reviewed empirical paragraphs. [SEO and LinkedIn copy](module-2-post-new-promo.md) accompany the new article; the SEO description is 110 characters.

The post and notebook also share a subsection, “Why use routing in an agentic workflow?”, covering specialization, selective computation, branch maintenance, auditability, LLM-versus-rule routing and its failure modes. This prose-only addition preserves every executed code cell and output. The post explains the weekly MASE denominator and its distinction from future baseline performance, retains the competitive R and D controls, and separates process auditability from unmeasured human explainability. Its work table and quoted runtimes explicitly identify the archived author session. New timing measurements remain in the separate reader records.

The asset inventory below uses paths relative to the root-level article. Numbering follows first appearance in the notebook. Each image appears under the same substantive heading in both deliverables; captions remain outside the PNGs.

| Filename / post image path | Purpose | Shared notebook and post section |
| --- | --- | --- |
| `graphs/module02-panel/atsf02-graph1.png` | Workflow and evaluation boundaries | Why test routing beyond one store and one month? |
| `graphs/module02-panel/atsf02-graph2.png` | Two preselected products across all stores | Build a panel without looking at future outcomes |
| `graphs/module02-panel/atsf02-table1.png` | Department quotas and case counts | Build a panel without looking at future outcomes |
| `graphs/module02-panel/atsf02-graph3.png` | Development, main and inner-fold chronology | Separate workflow selection from forecast evaluation |
| `graphs/module02-panel/atsf02-table2.png` | Six comparison arms and evaluation protocol | Separate workflow selection from forecast evaluation |
| `graphs/module02-panel/atsf02-graph4.png` | Paired error by store and forecast origin | Where does routing help, and where does it hurt? |
| `graphs/module02-panel/atsf02-graph5.png` | Paired error in predefined zero-fraction bands | Where does routing help, and where does it hurt? |
| `graphs/module02-panel/atsf02-table3.png` | Coverage, accuracy, paired differences and resampling intervals | Where does routing help, and where does it hurt? |
| `graphs/module02-panel/atsf02-table4.png` | Shared work, standalone fit budgets and archived uncached timings | How much work does routing save? |

## Limits

The sample is balanced across departments, not representative by sales or revenue. It covers three stores, six adjacent main horizons and one fixed numerical configuration. Complete recorded histories do not establish product availability, real launch timing or true demand. No inventory utility, forecast-interval calibration, analyst explainability improvement or LLM-specific benefit is measured. Shared-bank speed is never substituted for an independent routing speed measurement. No implementation blocker remains after the recorded checks pass.
