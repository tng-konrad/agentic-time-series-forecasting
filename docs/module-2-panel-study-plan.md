# Module 2 panel study: implementation decision and plan

Status: proposed implementation plan, 26 October 2026. This task inspected the project and source metadata; it did not change the notebook, post, numerical code, datasets, captures or publication assets, and did not run forecasting experiments.

Implementation completed 27 October 2026: see the [implementation and validation record](module-2-panel-implementation.md). The planning statements below describe what was known before execution.

## 1. Decision: create a companion notebook using existing tools

Create `a02-routing-panel.ipynb`, with a companion `a02-routing-panel-post.md` after execution. Retain `a02-routing.ipynb` and `a02-post.md` as the introduction to four request types, bounded LLM decisions, failures and clarification.

This is the faster route to a validated study, based on dependency inspection rather than a measured development-time benchmark. Both options require a longer data extract, an experiment manifest, repeated-origin execution and panel aggregation. Reusing the existing numerical functions avoids rebuilding the forecasting methods.

| Work | Modify existing notebook/post | New companion notebook/post |
|---|---|---|
| Longer panel and repeated-origin evaluator | Required | Required |
| Numerical methods | Reuse | Reuse |
| Teaching narrative | Combine four request types/LLM contracts with a second evaluation study | Focus directly on generalization of numerical routing |
| Existing evidence and tests | Rework checks coupled to old cases and publication structure | Keep original evidence and checks intact |
| Post and assets | Revise existing article, asset map, image placement and reviewed claims | New focused article and independently inventoried outputs |
| Regression surface | Changes to the existing publication plus the new evaluator | New evaluator and a small optional exporter-path extension |

The existing verifier explicitly requires 27 executed code cells, eight matching section headings, three graphs, six tables, nine common scale-defined cases and particular empirical counts. Publication helpers also select named old cases and draw a fixed origin. Extending this would require deliberate reworking, not simply adding an outer loop. Removing those checks merely to make an expanded notebook pass would weaken verification.

A small exploratory appendix would be quicker to add if the requirement were only to illustrate two extra dates. The requested product/store/origin study has enough distinct experimental logic to justify a separate notebook.

## 2. Question, scope and interpretation

Central question: **Does the existing numerical routing policy remain useful across more products, stores and forecast origins, compared with a fixed workflow and all-candidates selection?**

Learning objectives:

- Recognize a product–store–forecast-origin case as the evaluation unit.
- Separate internal candidate-selection folds from outer evaluations of the whole workflow.
- Distinguish variation across products, transfer across stores and stability through time.
- Compare paired forecast errors with coverage, undefined scales and computation visible.
- Identify situations in which routing helps or harms, without selecting favorable examples after scoring.

Keep the current forecasting methods, numerical settings, 50% zero threshold, 365-day final training window, three internal 28-day folds and 28-day forecast horizon. No learned router, threshold tuning, new forecasting family, business-note interpretation or human explanation study is included. Those are separate improvements discussed in the conversation.

The expanded primary study covers full histories. The original notebook retains the short/new-product scenarios and missing-information tests. A full sales row does not establish a real product launch date, stock availability or unconstrained demand.

The expanded study makes zero new LLM calls by default. It evaluates the usefulness of the existing deterministic routing policy. The original genuine LLM failures and separate development success remain inspectable in the original notebook. Any future LLM panel capture needs its own predeclared cohort and immutable records; it cannot inherit old responses for different inputs. More cases would not, by itself, create an LLM advantage over the same explicit policy.

## 3. Inspection findings and conventions

No applicable `AGENTS.md` was found in the repository or its ancestors. Inspected `notebook-recipe.md`, the executed notebook, its builder/runner/verifier, shared workflow/numerical/publication code, preparation/evaluation scripts, dependencies, existing planning records and relevant data manifests.

Reuse Python 3.12 and the locked NumPy/pandas/Matplotlib/statsmodels/StatsForecast environment; no new forecasting dependency is needed. Use grouped imports, `CFG`, `Setup`/`Utils`, seed 42, small code cells immediately followed by explanations, raw-data plots before fitting, `seaborn-v0_8`, configuration-derived publication dimensions and 180 DPI. Keep full calculation precision. Existing explicit user requirements for primary-source support and two-decimal table display take precedence over contrary recipe suggestions.

Reusable numerical APIs include `ForecastConfig`, `validation_windows`, `fit_forecast`, `score_forecast`, `select_candidate`, `run_forecast_path` and `run_unrestricted_reference`. `request_view`, `gate`, `profile` and `rule_route` provide useful input boundaries for explicit per-case origins. Create a panel-specific loader/evaluator rather than weakening `load_routing_panel`, whose twelve-series/5,724-row assertions protect the original dataset. The current scorer already obtains outcome dates from each request; the new evaluator must avoid assuming the old `main`/`full` cohort or A/B/C/D arm set.

## 4. Data: concrete proposed cohort

The source is already available at `../time-series-with-Konrad/data/sales_train_validation.csv.zip`, with `calendar.csv` in the same folder. Planning inspected headers/calendar and streamed catalog metadata only. The archive contains 30,490 series with day columns `d_1`–`d_1913`; 3,049 item IDs have rows in all three proposed stores: CA_1, TX_1 and WI_1. Seven departments are available.

Verified SHA-256 values match the existing source manifest:

- Sales archive: `88cf53dbb3782b93dc64c038f7589a2677ff171df8857346a07745fb80f9803e`.
- Calendar: `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5`.

Use these sources explicitly during author preparation; bundle the resulting small extract for reader execution. Do not require the companion repository, raw archive or network for reader replay. Do not switch to the separately available evaluation archive during the study.

Select 100 distinct item IDs present in all three stores, excluding the item IDs used by the twelve original series. Allocate 15 items to each of FOODS_1 and FOODS_2 and 14 to each of the other five departments. Within a department sort by SHA-256 of the literal `module02-panel-v1|42|<item_id>` and then by item ID. Select the first quota. Catalog membership and historical validity are checked before outcomes; never select based on which method wins. Fail visibly if a quota cannot be filled rather than choosing replacements based on performance.

This is a department-balanced teaching sample, not a representative estimate of all Walmart sales. Macro averages describe this declared cohort. Do not present them as M5 leaderboard WRMSSE or revenue-weighted business impact.

Extract the 300 item–store series for `d_1213`–`d_1913`, 25 May 2014–24 April 2016: 701 days, expected 210,300 sales rows. Preserve item, department, store and state metadata separately from the dated sales panel. Validate exact membership, unique complete daily calendars and finite nonnegative integer values. Keep zero-only histories; undefined scales are an evaluation issue, not a reason to select favorable products. Treat complete recorded histories as the study input contract without inferring true product age from zeros.

## 5. Development and outer evaluation dates

Use three development origins on CA_1 and TX_1 only to select the fixed-workflow comparator. Do not use WI_1 development errors for that choice. Main evaluation contains all three stores and six non-overlapping 28-day horizons.

| Role | Origin | M5 day |
|---|---|---|
| Development | 16 August 2015 | d_1661 |
| Development | 13 September 2015 | d_1689 |
| Development | 11 October 2015 | d_1717 |
| Main | 8 November 2015 | d_1745 |
| Main | 6 December 2015 | d_1773 |
| Main | 3 January 2016 | d_1801 |
| Main | 31 January 2016 | d_1829 |
| Main | 28 February 2016 | d_1857 |
| Main | 27 March 2016 | d_1885 |

For every request, expose exactly the 449 days ending at its origin. Recompute evidence, choose the route, validate candidates on three internal folds and refit on the final 365 days. Freeze all main decisions and dated predictions before revealing main evaluation errors.

The development grid has 100 × 2 × 3 = 600 cases. The main grid has 100 × 3 × 6 = 1,800 cases. Each case produces up to 28 predicted values; those daily errors are not 28 independent experiments.

Lock cohort, dates, threshold, candidates, metrics and aggregation before forecast scoring. Select the fixed comparator using development outcomes only and freeze its choice before scoring the first main horizon. The last development target ends on 8 November, which is available at the first main origin. Earlier main outcomes legitimately become history for later main origins; they must not be used to retune the fixed policy or comparator.

WI_1 is a location excluded from selecting the fixed comparator. The existing routing rule is already fixed and unlearned: describe WI_1 as a geographic robustness check, not evidence that a newly trained router generalizes. This retrospective extension is not a prospective trial. Excluding the original item IDs reduces reuse of already-inspected examples but does not turn historical dates into unseen future data.

## 6. Arms and fair numerical comparisons

| Arm | Action | Purpose |
|---|---|---|
| A | Seasonal naive | Simple numerical reference |
| B | Existing zero-frequency routing rule | Policy under evaluation |
| R | Always use the regular candidate family | Fixed specialist control |
| I | Always use the intermittent candidate family | Fixed specialist control |
| F | Freeze either R or I using development macro MASE | Stronger preselected fixed-workflow comparator |
| D | Validate all three candidate methods | All-candidates reference |

R/I still select within their fixed family using the same internal folds. Choose F from R/I on identical delivered, positive-scale development cases, with a declared tie order R then I. Stop and report insufficient development evidence if that comparison has no eligible cases. Never choose F from main outcomes, separately for each main store, or after seeing the headline result.

Every arm has the same information, numerical failure rules, clipping and seasonal-naive fallback. No arm chooses the final-horizon winner. Differences between B and F test useful specialization more directly than B versus A alone. Differences between B and D show the consequences of pruning the candidate menu.

For feasible full-grid accuracy recomputation, build a shared numerical bank keyed by permitted-history hash, train/validation dates, method, numerical configuration, versions and schema version. Store each method's three validation fits and final refit. Keep outcomes outside the bank's decision inputs. Pure selection reconstructs each arm from the same numerical records. Test reconstruction against existing independent executors on fixed fixtures.

There are at most 12 unique scheduled fits per case: three methods × (three folds plus one final refit). Main bank budget is 21,600 fit attempts; development bank budget is 7,200. Record every attempted failure. These are shared computation counts, not measured independent per-arm execution budgets. F reuses its chosen fixed-family record and does not imply extra fitting.

Separately reconstruct each arm's logical fit requests under the existing execution contract. Measure actual uncached B/D runtime on a predeclared 36-case subset: two hash-selected item IDs for every store/origin pair. Alternate B/D execution order across cases, warm imports using development-only data, and record the session/environment and all failed attempts. Do not add bank timing to these independent timings or present cached selection time as production speed. Fit reduction can be established from logical work; a runtime claim is limited to the measured subset.

## 7. Metrics and analysis

Retain per-case MAE/RMSE, lag-7 training-only MASE, scale status, route, candidates, selected/delivered method, numerical fallback and work provenance. Undefined scales stay N/A; no epsilon or zero error for missing forecasts.

The main six-arm MASE table uses one explicitly listed common-delivered, positive-finite-scale case set for every arm. Display eligible, delivered, common and scale-defined counts and explain exclusions. Preserve all delivered MAE/RMSE outside that subset. Keep coverage/failure totals for the entire 1,800-case grid visible.

Predeclare primary contrasts B minus F and B minus D; negative error differences favor B. Report overall results and all store/origin breakdowns. Report MAE as a secondary check. Analyze the same contrasts within fixed zero-fraction bands [0,.25), [.25,.50), [.50,.75), [.75,1], showing each denominator. Band membership uses pre-origin evidence only. Do not search for the threshold or subgroup that makes B win.

For uncertainty conditional on the chosen stores/origins, resample the 100 item-ID blocks 2,000 times using seed 42, retaining each sampled item's stores and dates together and keeping paired arm comparisons. Never resample 28 daily errors as independent cases. Show all six origin results and leave-one-origin-out sensitivity. Three stores and six adjacent horizons cannot establish broad geographic or temporal population uncertainty; state that limitation rather than using narrow intervals to imply it.

Success is a possible outcome, not an implementation requirement: B could beat F, approach D with fewer logical fits, help only certain groups, or lose. This expansion alone cannot establish an LLM-specific advantage or improved explanation quality.

## 8. Notebook, post and publication assets

Use these identical substantive section titles in the new notebook and post. Notebook-only Setup/Utils and configuration remain separate.

| Section | Demonstration/evidence | Planned exported assets |
|---|---|---|
| Why test routing beyond one store and one month? | Link original study; explain specialization and its limits; establish context before results | Workflow diagram |
| Build a panel without looking at future outcomes | Inspect the selected cohort and raw pre-origin histories; explain balanced sampling | Raw-history graph; cohort table |
| Separate workflow selection from forecast evaluation | Show development/main calendar and one origin's internal folds; freeze F | Evaluation-timeline graph; protocol/arm table |
| Repeat routing across products, stores and origins | Recompute numerical bank, reconstruct arms, save frozen predictions, then score | Text/JSON execution summaries only |
| Where does routing help, and where does it hurt? | Paired B/F/D errors, coverage, store/origin results, fixed intermittency bands | Error-difference graph by origin/store; subgroup graph; accuracy/coverage table |
| How much work does routing save? | Shared-bank work, logical arm budgets and separate uncached timing subset | Work table |
| What does the larger experiment establish? | Interpret the actual evidence, including null/negative results and limits | No additional asset |

Proposed scope: five graphs and four rendered tables. Save them under `graphs/module02-panel/`, with graph filenames `atsf02-graph1.png` through `atsf02-graph5.png` and table filenames `atsf02-table1.png` through `atsf02-table4.png`, independently numbered in first notebook appearance order. The subdirectory keeps the original Module 2 assets intact while retaining the required filename patterns.

Exact new-post embeds, relative to root-level `a02-routing-panel-post.md`:

- Section 1: `![Routing and its evaluation boundaries](graphs/module02-panel/atsf02-graph1.png)`.
- Section 2: `![Two selected products across three stores](graphs/module02-panel/atsf02-graph2.png)` and `![The predefined product and store cohort](graphs/module02-panel/atsf02-table1.png)`.
- Section 3: `![Development, inner folds and outer forecast origins](graphs/module02-panel/atsf02-graph3.png)` and `![Comparison arms and chronological protocol](graphs/module02-panel/atsf02-table2.png)`.
- Section 5: `![Paired routing errors across stores and origins](graphs/module02-panel/atsf02-graph4.png)`, `![Routing differences within predefined intermittency bands](graphs/module02-panel/atsf02-graph5.png)` and `![Forecast errors with common denominators and coverage](graphs/module02-panel/atsf02-table3.png)`.
- Section 6: `![Shared computation and independently measured work](graphs/module02-panel/atsf02-table4.png)`.

Choose the two raw-history examples by item-ID hash before scoring, not by how well the methods perform. Plot only permitted historical values before the first fit. Error figures show all predefined stores/origins/bands; they must not hide negative effects or N/A cells.

Use the existing style and table renderer: no title/footer/caption/free text within table images, exactly two decimals for floating results, integer counts unchanged, full calculation precision. Save typed sidecars and an asset manifest binding source, cell ID, section, PNG hash, font, dimensions and resolution. Display the saved PNG bytes in the notebook and reuse them in the post. For the shared exporter, derive the manifest's relative image path from the actual destination instead of hardcoding `graphs/<filename>`; test unchanged original defaults and the new nested path.

## 9. Files and ordered implementation steps

| Step | Files/work | Completion criterion |
|---|---|---|
| 1. Freeze design | `examples/module02_panel/protocol.json`; document source/selection/splits/arms/metrics | No scored main outcomes have informed the protocol |
| 2. Prepare panel | `scripts/prepare_module02_panel.py`; `data/module02_panel/{sales.csv.gz,items.json,cases.json,manifest.json}` | Verified raw hashes, 100 item IDs, 300 series, 701 days, 600 development and 1,800 main cases; default reader needs only bundled files |
| 3. Implement boundaries | New `agentic_ts/module02_panel.py`; reuse numerical APIs | Explicit per-case origin, restricted histories, panel-specific validation and pure arm reconstruction; old APIs unchanged |
| 4. Numerical bank and development | `scripts/evaluate_module02_panel.py`; `examples/module02_panel/development_results.json`, `fixed_workflow.json`, immutable bank/checkpoints | F chosen only on allowed development cases; bank keys invalidate on any input/config/version change |
| 5. Main execution | Freeze records before scoring; write `frozen_forecasts.json`, `results.json`, `summary.json`, `work.json`, timing subset records | Full main grid accounted for; all missing outcomes/failures reported; no selection based on main errors |
| 6. Teaching notebook | `scripts/build_module02_panel_notebook.py`, `scripts/execute_module02_panel.py`, `a02-routing-panel.ipynb` | Clean-kernel execution recomputes the full numerical bank offline and verifies stable numerical outputs; origin batches/checkpoints bound individual cells |
| 7. Export and inspect | Minimal exporter-path extension; new panel plotting helpers; nine assets and sidecars | All files inventoried, sequential names, native/760-pixel inspection and table rules pass |
| 8. Post and verification | `a02-routing-panel-post.md`, panel claim map, `scripts/verify_module02_panel_publication.py`, `tests/test_module02_panel.py` | Matching sections, actual results only, all paths/hashes/numbers/denominators agree; original publication still passes its own checks |
| 9. Handoff | Add companion links/run commands to README and a brief link from original post; update original post hash/claim-review binding for that editorial addition | New experiment and original capture remain clearly distinguishable; implementation record contains measured runtime and limitations |

The full notebook is the default reproducible study; no live model/network access is needed after environment setup. Batch by origin and use an explicit runner cell timeout appropriate to measured pilot runtime, instead of assuming the current 300-second limit is adequate. Checkpoint by exact input hashes. A feasibility pilot should use separate unscored development-only cases and report runtime/storage before full execution. If a smaller smoke-test profile is supplied, name it visibly and never label its results as the 1,800-case study. Do not quietly shrink the full cohort to meet runtime expectations.

## 10. Validation and completion

- Validate calendar mapping, product/store membership, exact counts, sample exclusions, deterministic selection and all development/main boundaries.
- Mutate post-origin values for one case and assert unchanged restricted view, features, route, candidate selection and forecast; its score should change. Later-origin forecasts may legitimately change when the modified date is then part of their past. Mutating the last main horizon must change no prediction at any origin.
- Alter WI_1 development values in an in-memory fixture and assert unchanged fixed-workflow selection. Changing any main outcome must not change F.
- Test all-zero histories, undefined scales, failed validation fits, final fallback, ties, missing forecasts and scale-defined empty subsets.
- Check bank-based predictions against uncached existing functions; ensure actual shared work and logical per-arm requests are never confused.
- Check paired masks and product-block bootstrap resampling retain every sampled item's correlated rows and corresponding arms.
- Execute the notebook cleanly, verify independent final-score arithmetic and exact prediction dates, resolve every post image reference, compare headings, and visually inspect all assets.
- Re-run original Module 2 tests/publication verification after the small shared exporter change; preserve original capture hash and numerical semantics. Any original-post link addition requires refreshing its explicit editorial binding, not altering historical results.

The completed implementation should permit a negative conclusion. It is complete when its design, runs, outputs and limitations are inspectable and reproducible, not when routing wins.

## 11. Sources and open dependencies

Primary references: [M5 organizers' data/methods repository](https://github.com/Mcompetitions/M5-methods), [Forecasting: Principles and Practice on rolling-origin evaluation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation), and [Anthropic's routing-workflow explanation](https://www.anthropic.com/research/building-effective-agents) for the opening context. Any claim of popularity must be limited to documented use in primary guidance, without inventing adoption statistics.

No known source-access blocker exists: the needed archives, calendar and pinned dependencies are local. Full-grid runtime, actual numerical failures and resulting comparison quality remain implementation measurements; they were not established by metadata inspection. No forecasting, figure generation, notebook modification or post drafting was performed as part of this plan.
