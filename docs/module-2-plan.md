# Module 2 implementation plan: One forecasting request, several possible problems

Status: revised after independent critique; ready for implementation. Reviewed 26 October 2026. Planning only; no Module 2 notebook, post, model capture, or experimental results have been produced by this task. Review findings and decisions are in `docs/module-2-critique.md`; this plan is self-contained.

## 1. Scope, question, and inspected conventions

**Pattern: routing.** Build two connected deliverables at the repository root: `a02-routing.ipynb` and `a02-post.md`. The notebook demonstrates a bounded dispatcher choosing a numerical workflow; the Substack draft explains the same executed evidence. The central question is:

> Given the history and metadata actually available at a forecast origin, which workflow can handle this request—and does an LLM router add anything useful beyond explicit rules?

Readers should learn to distinguish established sales, intermittent sales, short observed histories, and products with no observed history; construct evidence before routing; separate route eligibility, policy agreement, and forecast quality; handle missing information without inventing it; and compare the extra work of LLM routing with a simple alternative. The audience knows basic Python but is new to these workflow distinctions.

The task is daily **recorded unit sales**, not unconstrained demand. Keep “demand” when explaining the module title and established forecasting terminology, but label the measured target as observed sales throughout. Zeros do not prove stockouts, obsolescence, or zero demand.

### Inspection record and decisions

- No applicable `AGENTS.md` was found in the repository, its ancestor directories, or the referenced companion repository during inspection. Recheck for new instructions when implementation begins.
- Read `module-list.txt`, `README.md`, `notebook-recipe.md`, `pyproject.toml`, `.gitignore`, Module 1's plan/implementation record, the executed `a01-forecasting-workflow.ipynb`, and `a01-post.md`. Inspected `agentic_ts/routing.py`, `routing_forecasting.py`, `workflow.py`, `module01_assets.py`, A02 preparation/capture/evaluation/runner scripts, and `tests/test_a02_routing.py`.
- Inspected the starting notebooks `../time-series-with-Konrad/m06-sales-and-demand.ipynb` and `m14-tsfm.ipynb`, especially Croston/TSB and the 60-day context experiment; also inspected the opening structure of `m16-post.md` and `m17-post.md`. No separate post recipe was found. The m14 experiment has a short context, not zero observations: explicitly teach that distinction.
- The active Module 1 publication is the closest style reference. Its configuration uses `seaborn-v0_8`; m06 uses `fivethirtyeight`, so do not silently mix styles. Visually inspected Module 1's forecast plot and two-block table. Reuse its fonts, palette, export settings, surrounding captions, and display of saved PNGs.
- Existing A02 code supplies valuable numerical routines, exact replay, a two-route dispatcher, and leakage/failure tests. Its gate requires 449 observations, so it cannot demonstrate all four requested cases unchanged. Extend it without changing legacy defaults; do not copy its forecasting code into the notebook. Existing A02 notebooks/posts/captures referenced by README are absent. No earlier results are evidence for the new benchmark.
- At initial inspection the working tree contained substantial edits and deletions. Preserve existing work and recheck current status before implementation. This planning task creates only the two requested Markdown documents. Later implementation should add the new A02 paths and update only relevant README links, without restoring deleted artifacts. Module 1's earlier instruction to start that module from scratch is historical scope, not a restriction on this module's reuse of the project.
- Reuse Python 3.12 and `uv.lock`. Current numerical pins are NumPy 2.4.6, pandas 3.0.3, Matplotlib 3.11.0, SciPy 1.15.3, statsmodels 0.14.6, and StatsForecast 2.0.3; nbformat/nbclient/ipykernel already support clean execution. No agent framework, neural forecaster, embedding model, or new numerical dependency is needed.
- Follow the recipe's grouped imports, `CFG`, `Setup`/`Utils`, chronological splits, small code cells followed by what/why/interpretation prose, and reuse of helpers. Use approximately 25–35 code cells, each new operation explained; avoid padding to meet a word count. Override the recipe's rounding advice: calculations and stored results retain full precision. Override its “no academic citations” advice with the user's explicit source requirement. Print preparation previews as text, not unexported HTML tables.

## 2. Example, data, and information boundaries

### Reuse the existing M5 subset

Use `data/a02/m5_routing_panel.csv`, `manifest.json`, and `requests.json` as read-only source material. The CSV hash was checked against its manifest during planning. It contains 12 CA_1 item series, 477 days each, 5,724 rows, from 2015-01-04 through 2016-04-24. Fields are `series_id`, `date`, and integer `sales_units`; item/department/store metadata are in the manifest.

The existing extraction selected four dense, four sparse, and two items on either side of a zero-fraction threshold, using only the 365 days ending 2016-01-03, with lexicographic source-ID tie breaking. Preserve that selection and its exact predicates/hashes. These are teaching strata, not population weights or diagnoses. Recompute current evidence at the final origin; a selection stratum must not dictate the later route.

Raw `sales_train_validation.csv.zip` and `calendar.csv` also exist in the companion repository's `data/`. Reader replay must need only this repository's small subset. If regeneration is required, pass that source directory explicitly to the preparation script, verify its pinned source hashes, and write to a new destination. Preserve the official M5 provenance and original usage-terms links; assert no new data license. Do not require a new competition download for the default run.

### A small, fixed request suite

Create `data/module02/cases.json` and a derived manifest referencing the original subset hash. Do not duplicate the sales CSV. Lock these cases before any new model capture or final scoring:

| Cases | Construction and information exposed | Purpose |
|---|---|---|
| 12 full-history requests | Existing items, all 449 dated observations through 2016-03-27; metadata marked as retrospective catalog information | Real-series comparison of regular/intermittent candidate workflows, including borderline examples |
| 2 short-history requests | `FOODS_1_018_CA_1_validation` and `FOODS_1_001_CA_1_validation`, restricted to the last 60 days through the same origin | Controlled information restriction; neither router nor forecaster can recover the hidden older target history |
| 2 zero-history requests | The same two anchors, with all target history masked and explicit synthetic `lifecycle=new`, `launch_date=2016-03-28` metadata | Simulated new-product dispatch and an honest distinction from short history |

For the zero-history cases, supply a deliberately simple, predeclared analogy assumption: the first three lexicographic remaining panel IDs in the same department/store, excluding both anchors, form a fixed donor set. This selects `FOODS_1_002_CA_1_validation`, `FOODS_1_003_CA_1_validation`, and `FOODS_1_005_CA_1_validation`. Their last 28 observed days through the origin may be used; their later sales may not. The scenario explicitly authorizes a same-scale donor prior. No target sales, hidden earlier target statistics, similarity fitted on the target, or target-based scaling is allowed.

This is an information-masking experiment on established products, **not a genuine launch dataset or a validation of launch forecasting**. Donor suitability is a teaching assumption, not an inferred business fact. Repeated anchors share outcomes and are not independent observations. Report the 12 real full-history requests separately from the four constructed views.

Assign main IDs `M02-F01`–`M02-F12` in the source manifest's item order, `M02-S01`/`M02-S02` for the dense/sparse short views, and `M02-N01`/`M02-N02` for their new-product views. `M02-F01` is the dense anchor, `FOODS_1_018_CA_1_validation`. Store the ID-to-source mapping explicitly.

Add these four input cases and two response fixtures with fixed derivations. Only X01/X02 reach the live router; X03/X04 are stopped by shared input validation. X05/X06 use a deterministic fixture client and make no real model call. They remain six E3 rows, separate from the 16-case accuracy cohort.

| ID | Exact derivation | Expected component behavior |
|---|---|---|
| `M02-X01` | Clone N01; set lifecycle and launch date to unknown, retain authorized donor information | Router should clarify whether this is a launch or missing historical data |
| `M02-X02` | Clone N01; retain launch metadata, remove donor IDs/data and set analogy permission to unknown | Router should request the missing analogy information/permission |
| `M02-X03` | Clone F01; set horizon to missing | Shared gate requests the horizon; zero model/forecaster calls |
| `M02-X04` | Clone F01; remove the interior observation dated 2016-03-17, retaining the declared visible span | Shared gate returns invalid input; no zero-fill and zero model/forecaster calls |
| `M02-X05` | Clone F01; fixture client returns the malformed JSON text `{` | Contract rejection and exactly one B fallback; no actual model call |
| `M02-X06` | Clone S01; fixture returns otherwise schema-valid `regular`, cites `history.available_span`, uses a short reason and null clarification question | Reject because 60 observations cannot run the regular workflow, then one B fallback to the short workflow |

Keep expected outcomes in the test/evaluation manifest, never in router evidence. X01/X02 are constructed requests with genuine model responses; X05/X06 responses are injected failures, not observed LLM behavior. Questions in the table are expected subjects, not exact wording that must be matched.

### Temporal protocol

- Forecast origin: 2016-03-27. Final target dates: 2016-03-28–2016-04-24, 28 days. Those outcomes are accessible only to the final scorer.
- Full-history numerical selection: three 28-day validation blocks, each preceded by 365 training days. Training ends 2016-01-03, 2016-01-31, and 2016-02-28; validation blocks end 2016-01-31, 2016-02-28, and 2016-03-27. Final fits use the last 365 days through the origin. These are candidate-selection folds inside the available history, not three independently evaluated routing decisions.
- Short-history selection: one 28-day validation block at the end of the 60-day available history, trained on the preceding 32 days. Refit the selected simple candidate on all 60 available days. Do not relax the full-history model's requirements to fit it on this case.
- Zero-history prior: no target validation, target MASE denominator, or target fitting. Donor observations stop at the origin, and the formula is fixed in advance.
- Construct a restricted request view first, then pass only that view to profiling, routing, and forecasting. The LLM receives summary evidence, never hidden target rows, final scores, benchmark stratum labels, expected routes, or another arm's decision.
- Freeze all main predictions and the diagnostic predictions before opening final outcomes. Keep whole-file hashes for provenance separate from input hashes derived only from the permitted request view. Changing held-out or masked target values must leave the pre-scoring decisions and predictions unchanged.

## 3. Numerical workflows and simplest baselines

Routing chooses a **workflow**, and deterministic validation selects a candidate inside that workflow. No LLM predicts values, invents model parameters, selects analogs, or reads held-out scores.

| Workflow | Eligibility and candidates | Selection/delivery |
|---|---|---|
| `regular` | At least 449 consecutive observations; seasonal naive with period 7 and damped additive Holt–Winters with period 7 | Lowest full-precision mean validation RMSE over all three folds |
| `intermittent` | Same history requirement; seasonal naive and TSB with `alpha_d=alpha_p=0.1` | Same folds and selection rule |
| `short_history` | 56–448 consecutive observations; seasonal naive and trailing-28-day mean | One 28-day validation block, at least 28 preceding training observations; lowest RMSE |
| `new_product` | No target history, explicit launch metadata, same-unit authorized donor set with complete pre-origin data | Constant forecast equal to the median of the three donor 28-day means; no fitted/tuned target model |
| `clarify` | Missing lifecycle/analogy information, inconsistent metadata, or too little history for the declared workflows | Structured question and no forecast |

The supported short-history interval is a declared teaching capability limit. A future deployment may offer simpler methods below 56 observations; this module asks for more evidence instead. Handle all-zero histories explicitly: they are observed data, not empty history. Seasonal naive and the mean return zero; TSB must return finite nonnegative values or be marked ineligible. Preserve undefined scaled errors as `N/A`, never replace their denominator with an epsilon.

Holt–Winters uses the existing additive trend, damped trend, additive seasonality, estimated initialization, no Box–Cox transform, `L-BFGS-B`, and `minimize_kwargs={"options": {"maxiter": 2000}}`. A convergence warning, explicit optimizer failure, nonfinite output, or wrong shape makes a fit unsuccessful. Clip negative point forecasts to zero, record clipping, and apply the same rule in all arms. Retain warnings; do not globally suppress evidence of failure.

Within a workflow, every required validation fold must succeed. Break exact selection ties by declared candidate order with seasonal naive first. If the chosen final fit fails, deliver the already-computed seasonal naive forecast with an explicit numerical-fallback status. If the baseline fails, return numerical failure; do not fabricate a forecast. The donor prior fails closed if its inputs are invalid.

Compare these paths:

- **A — Minimal forecast baseline:** seasonal naive with at least 56 valid observed days; the same fixed donor prior on eligible zero-history requests; otherwise clarification. It shares the metadata consistency checks and the 56-day support floor with B/C. That floor is this module's capability contract, not seasonal naive's mathematical minimum. This baseline has essential capability checks and is not called “no routing.” It does no model search and makes no LLM calls. Sharing the donor prior prevents attributing an otherwise unavailable forecast to LLM intelligence.
- **B — Explicit rules:** apply the routing policy below, then run the selected workflow.
- **C — One-call LLM router:** receive exactly the same measured evidence and declared policy, propose a workflow with evidence IDs and a short reason, then dispatch the validated proposal. A successful decision genuinely controls the executed path.
- **D — All-candidates diagnostic, full-history requests only:** after A/B/C decisions are frozen, evaluate the union of seasonal naive, Holt–Winters, and TSB on the same three validation folds and select by the same rule. This checks the consequence of pruning a candidate family. It is not an oracle, does not use final outcomes to select, and is not evidence available to the router. Keep its extra work separate.

No foundation models, Croston-family survey, inventory optimization, calibration bands, dynamic diagnostic loop, or multi-agent architecture is needed. The m06/m14 starting points supply the problem framing, not a requirement to bring all their dependencies into this module.

## 4. Routing policy, contracts, and capture

### Shared gate and ordered rule policy

The gate validates request ID, item ID, daily frequency, item-unit target, explicit 28-day horizon and issue date, date uniqueness, finite nonnegative values, and calendar completeness **within the declared visible span**. An explicitly zero-length history is allowed; a missing row inside an observed span is not a zero. Missing required request fields cause clarification; corrupt data cause `invalid_input`. The gate does not compute the expected route. Both B and C receive identical gate outcomes; a stopped case makes no model or forecaster call.

On structurally valid evidence, B and the policy supplied to C apply this precedence:

1. Resolve inconsistent lifecycle/history metadata to `clarify`; never guess whether missing history means a launch.
2. With zero observed target rows, choose `new_product` only when launch metadata and the authorized donor-prior requirements are satisfied; otherwise `clarify` with the missing requirement.
3. With 1–55 observed days, choose `clarify`; with 56–448, choose `short_history` before considering zero frequency.
4. With at least 449 days, use the last 365 days: choose `intermittent` when `2 * zero_count >= observed_count`; otherwise `regular`.

The 50% zero threshold is an intentionally simple, disclosed policy, not the Syntetos–Boylan–Croston ADI/CV² classification and not a claim of optimality. Use integer comparisons to avoid display rounding changing a route. Near-threshold cases test policy boundaries, not missing-information detection. Do not tune the threshold using final scores.

### Small contracts

- `RequestView`: request/issue IDs, target, units, horizon, visible history, lifecycle/launch metadata, donor authorization and permitted donor histories, explicit observed/missing field statuses, and a view hash. Scenario labels stay in the evaluation manifest.
- `Evidence`: history span/count, current-window zero and positive counts, mean sales, declared metadata, donor IDs/availability and computed prior, limitations about observed sales, plus stable evidence IDs. Every statistic is computed in Python. Omit unavailable quantities rather than inventing zeros.
- `RouteDecision`: exactly `route`, `evidence_ids`, `reason` (1–60 words), and `clarification_question`; allowed routes are the five above. Only `clarify` permits/requires a nonempty question. Unknown fields/IDs, duplicate IDs, malformed JSON, invalid enum values, or truncated responses fail validation.
- `ExecutionRecord`: gate result, raw response, proposed route, accepted route, executed route, decision source, policy route (added for evaluation, never sent in the evidence), evidence/prompt/schema/config hashes, timestamps, usage, status, failure reason, numerical selection/delivery, candidate traces and forecast hash.

Separate schema validation, capability validation, policy agreement, and semantic review. A valid eligible disagreement with the threshold policy must execute the proposed family; do not replace it merely to make C agree with B. An ineligible proposal is rejected. A schema/capability/API failure gets one deterministic B fallback, explicitly attributed to the fallback, with no LLM repair/retry. A valid C clarification produces no forecast and is evaluated as abstention, not an API failure. Replay-integrity failures are different: stop the publication run rather than silently substituting rules for a mismatched archive.

Manually audit every reason against its cited evidence as supported, contradicted, unsupported, or ambiguous. Existing evidence IDs alone do not establish support. Audit questions for whether they request an actual missing requirement. Publish actual failures and unnecessary clarifications; no selected “best response” replacements.

### Genuine inference and exact offline replay

Use the existing local Ollama adapter and the repository's configured `qwen3-coder-next:q4_K_M` as the planned backend. Verify availability and record the model digest/server version before implementation capture; it was not queried or run in planning. Reuse A02's explicit settings: temperature 0, seed 42, `num_ctx=32768`, `num_predict=4096`, and a 240-second request timeout. Record any backend thinking/default options as part of the immutable configuration. Seeds do not guarantee identical inference across versions/hardware.

Freeze protocol/prompt/case manifest first. Make one attempt for each of the 16 main cases and the two router-level ambiguity cases, in fixed case-ID order, with no hidden retries or model switching. Persist attempted failures incrementally. The two gate cases and two response fixtures require no live inference. Do not reuse an old capture whose contracts or evidence differ.

Archive raw responses, parsed decisions, exact messages/schema, permitted-view hash, model identity/configuration, numerical configuration, source provenance, token counts, finish reason, client wall time and server duration/load metadata when supplied. Preserve the first capture; supplementary development runs use new IDs and do not replace primary outcomes.

The default notebook recomputes numerical work offline and replays genuine captured responses only after exact input/response-integrity checks. Distinguish archived inference from current replay; never label a deterministic fixture as an agent response. Completion requires at least one genuine accepted LLM decision that reaches a numerical workflow, with all other primary outcomes retained. If no such decision is obtained, report the mechanism demonstration as incomplete; a labeled supplementary genuine run may demonstrate it without entering primary comparison metrics.

## 5. Experiments and evaluation

| Experiment | Question and control | Evidence and interpretation |
|---|---|---|
| E1: route the fixed suite | Compare B and C on 12 real requests and four constructed views using identical evidence; A establishes minimum forecasting behavior | Proposed/accepted/executed paths, supported evidence, clarification and fallback counts. Policy agreement is compliance with a declared rule, not ground-truth forecast correctness. |
| E2: measure downstream consequences | Execute A/B/C; freeze D for full-history cases; score the same final dates only after all forecasts are frozen | Per-request MAE/RMSE, training-only MASE where defined, coverage, paired differences, and whether pruning changes validation selection. Equal routes should produce equal numerical outputs, apart from runtime fields. |
| E3: incomplete evidence and bounded failures | Two ambiguity cases, two shared-gate cases, two injected response fixtures | Which component stops or clarifies, exact missing information, rejected proposals, fallback source, and calls made. Keep fixture outcomes separate from observed behavior. |
| E4: count the work already done | No additional model calls: account for E1/E2 traces and the genuine capture | Calls, input/output tokens, routing duration, candidate fits, numerical duration, coverage and fallbacks. Cost means measured work; no invented dollar charge or hardware-energy estimate. |

For full-history cases, report per-item final MAE and RMSE in item units and macro MASE with a per-item scale equal to the mean absolute lag-7 difference in that item's final 365-day training window. MASE below one compares against this historical scaling error, not necessarily the actual final seasonal-naive forecast. Compare A's measured final errors directly. Do not use MAPE on zeros.

For the two short cases, report MAE/RMSE separately; for the two zero-history cases, do likewise without a target-based scaled error. Do not average these four cases into the 12-item result or call them additional independent products. Raw records retain all metrics at full precision; the main table can prioritize MAE and MASE and leave secondary RMSE in the linked evidence record.

Show forecast coverage out of all eligible requests alongside errors. Compare B/C on the identical subset where both delivered, with its denominator; show each arm's missing outcomes and fallback deliveries separately. For the A/B/C/D headline full-history table, define `F_common` as requests delivered by all four arms, then `F_scaled` as members of `F_common` with a positive finite lag-7 training scale. Compute every arm's macro MASE over the same `F_scaled`. Show all three counts: common delivered, scale-defined, and excluded for undefined scale; the latter can be explained in adjacent prose to keep the image compact. If `F_scaled` is empty, macro MASE is `N/A`. Retain MAE/RMSE for all delivered cases, including undefined-scale cases. Never give abstention zero error, replace a zero scale with epsilon, or let DataFrame skip-NaN behavior silently choose the denominator. Preserve per-request errors so readers can see heterogeneity. D is not guaranteed to improve final performance.

Routing metrics: schema-valid count/attempted calls, accepted eligible decisions/attempts, policy agreements/accepted forecasting decisions, correct clarification/ambiguity cases, unnecessary clarification/main cases, input-gate stops, routing fallbacks, numerical fallbacks, and evidence-audit counts. Use integer numerator/denominator displays. B agreement with its own policy is a construction check, not a learned benchmark score.

Freeze Table 6's primary A/B/C cost cohort to the same **16 main requests**, including failures and clarifications. Report the two genuine ambiguity attempts separately; gate/fixture work is separately labeled and excluded from primary totals. A/B have zero model calls; C has one attempted call per main request in a complete capture, irrespective of success. Record actual attempts rather than assuming the target count was reached.

Capture A/B dispatch durations with a monotonic timer in the same author session as C; do not hardcode them to zero. C duration includes its failed attempt and any fallback dispatch; store inference and fallback components separately. Record numerical work independently for each arm with its own run/session ID and environment. Show archived dispatch/inference durations and current numerical execution durations as separate components: **do not sum timings from different sessions and label the sum measured end-to-end latency**. If caching is used, disclose reused results and distinguish actual fits from logical requested fits; prefer uncached numerical arm runs for this small suite. D's full logical fit budget and any actual additional/reused work are reported separately. Give every total/median its cohort and measured observation count; an incomplete total is labeled partial, not extrapolated. Include machine/runtime/model settings and load status. Missing token/load/timing data are `N/A`. Tiny positive durations may display `0.00` under the required formatting; preserve full precision and explain the rounding in prose. One local capture is not a latency benchmark or a general cost estimate.

## 6. Notebook/post correspondence and section outlines

The following eight titles are the canonical substantive `##` headings in both deliverables, in this order. The table is the section-by-section outline and evidence map. Asset numbers refer to the inventory in section 7.

| ID | Exact notebook and post section title | Concept and notebook demonstration | Evidence the post discusses | Assets |
|---|---|---|---|---|
| S1 | Why route a forecasting request? | Contrast Module 1's fixed chain with one decision selecting among predefined branches; show the gate, router, numerical workflows, and clarification exit | Source-supported origins, problem, practical interest, complexity trade-off, and bounded authority | Graph 1 |
| S2 | Four requests, one sales dataset | Load/check data, build restricted views, immediately plot permitted raw histories, inspect typed profiles and dated splits; explicitly empty new-product target history | Real sales versus constructed information restrictions; zeros versus missing observations; donor assumption and time boundary | Graph 2, Table 1 |
| S3 | Define the workflows before routing | Walk through A, regular/intermittent candidate sets, short-history validation, and the fixed donor prior; explain each method at its first call | Eligibility before sophistication; baseline role; why a route is a candidate workflow, not a guaranteed best model | Table 2 |
| S4 | Let rules and an LLM choose a path | Apply B; inspect C's exact evidence, archived genuine response, validation and dispatch; run the 16 main cases | Proposed versus accepted versus executed route, evidence support, visible agreements/disagreements and fallback provenance | Table 3 |
| S5 | Measure forecasts and routing separately | Complete selected workflows and D, freeze forecasts, reveal outcomes, calculate paired metrics; show four predetermined representative requests | Forecast accuracy separately from policy agreement; coverage; candidate pruning; the limited meaning of synthetic cold start | Graph 3, Table 4 |
| S6 | When the request is incomplete | Run two ambiguity cases, two input-gate cases and two injected response failures; inspect actual statuses | Asking a useful question, shared validation versus agent judgment, and bounded deterministic recovery | Table 5 |
| S7 | Count the work, including failures | Aggregate archived inference work and current numerical work; audit reason support and call/fit counts | Overhead, failed work, missing measurements, fallback attribution and replay-versus-inference distinction | Table 6 |
| S8 | When is routing worth the complexity? | Read a compact text checklist and reconcile outcomes with the question | Conditional conclusions: rules may suffice; justify extra machinery by useful handling of requests plus measured consequences; limitations and transition to Module 3 | Reuse references to Tables 3, 4 and 6; no new image |

Notebook-only opening sections are `# Setup` (grouped imports; root-path assertion; `CFG`; seed/style; pinned versions; offline mode/capture identity) and `# Utils` (import reusable profiling, splitting, dispatch, evaluation, export helpers; explain their contracts). Substantive sections then follow S1–S8. Keep the notebook body readable and expose a small request/evidence/decision record as text; link to reusable implementation details. Utility code must not quietly run live inference or export unnumbered diagnostic figures.

The post starts with its title, a short concrete forecasting-request hook, and a link to `a02-routing.ipynb`, then **S1 as a dedicated opening context section before results**. Plan roughly 1,800–2,500 words, adjusted for clarity. For each later section, explain the concept, cite its notebook evidence, interpret the exact outputs, and state the relevant limitation. Use concise source links; a post-only `## Sources` is allowed as non-substantive back matter. This plan is an outline, not a draft of the post.

### Opening context and source plan

Sources were inspected during planning; recheck links and date-sensitive descriptions during writing. Historical/source claims below are distinct from hypotheses about this module.

| Planned claim/use | Primary or original source | Boundary on interpretation |
|---|---|---|
| An antecedent for combining language models with specialist symbolic/numerical modules | [Karpas et al., MRKL Systems, submitted May 2022](https://arxiv.org/abs/2205.00445) | Related architectural lineage; do not claim it invented all routing or is identical to this dispatcher |
| Definition of routing as classifying an input and directing it to a specialized path; why distinct tasks benefit from separation | [Anthropic, Building effective agents, 19 December 2024](https://www.anthropic.com/research/building-effective-agents) | Attribute the documented pattern and date; do not treat later edits/model examples as present in the original 2024 version |
| Practical visibility in current workflow tooling, with structured routing and conditional branches | [LangGraph official workflow guide](https://docs.langchain.com/oss/python/langgraph/workflows-agents) | Alongside the above guide, supports documented prominence, not market share or measured adoption; this module needs no LangGraph dependency |
| Dataset provenance and interpretation as daily unit sales | [M5 organizers' repository and competitor-guide links](https://github.com/Mcompetitions/M5-methods) | Preserve original data terms; do not infer real launch dates from leading zeros |
| TSB implementation and parameter meanings | [StatsForecast TSB documentation](https://nixtlaverse.nixtla.io/statsforecast/docs/models/tsb.html) and the installed pinned `statsforecast/models.py` | Use for API behavior, not popularity claims. The web page contains conflicting historical dates; omit a TSB invention-date claim unless the original paper is verified. |
| Chronological forecast evaluation, naive/mean baselines, and scaled errors | [Hyndman and Athanasopoulos, Forecasting: Principles and Practice, forecaster's toolbox](https://otexts.com/fpp3/toolbox.html) | Supports the evaluation method, not the chosen threshold or an expected winning model |
| Structured chat outputs and returned usage/duration fields | [Ollama chat API](https://docs.ollama.com/api/chat) | Availability of a field does not guarantee every backend returns it or validate a model's reasoning |

S1 must answer all four context questions: the architectural antecedents and named workflow guidance above; the problem of heterogeneous requests being forced through one path; the appeal of specialization, inspectable dispatch and bounded work that helps explain the pattern's popularity; and when this added complexity is justified. Phrase the appeal as a reasoned explanation supported by those guides, not a measured causal account of adoption. Say that the benchmark will test this small setting, and that reliable structured fields may make explicit rules sufficient. Benefits with messy natural-language metadata remain a motivation, not an evaluated result here.

## 7. Asset inventory and exact post placeholders

For module number **2**, the literal prefix is **`atsf02`**. Save every diagram, graph and rendered table under `graphs/`. Number graphs and tables independently in order of first appearance in the notebook. There are three graphs and six tables planned. The strings below are literal future post embeds, relative to the intended root-level `a02-post.md`, **not relative to this plan in `docs/`**.

| Filename | Purpose and contents | First notebook section | Post section and exact image placeholder |
|---|---|---|---|
| `atsf02-graph1.png` | Shared gate → B/C routing → four numeric paths or clarification; show A as the simple reference and D as a separate diagnostic | S1 | S1: `![The bounded routing workflow](graphs/atsf02-graph1.png)` |
| `atsf02-graph2.png` | Four panels: full observed dense/sparse histories; the dense anchor's visible last 60 days; new-product target history explicitly absent with donor histories shown distinctly. No final outcomes. | S2 | S2: `![The information available for four forecasting requests](graphs/atsf02-graph2.png)` |
| `atsf02-table1.png` | Four case classes, number of cases, visible target rows, validation scheme and shared final dates; compact columns with dates explained in prose | S2 | S2: `![Request views and chronological evaluation](graphs/atsf02-table1.png)` |
| `atsf02-table2.png` | Four workflows plus clarify: eligibility, candidates and selection rule; baseline A and diagnostic D described in surrounding prose | S3 | S3: `![Workflow eligibility and candidate methods](graphs/atsf02-table2.png)` |
| `atsf02-table3.png` | All 16 main cases: short request ID, observed-row/zero-count evidence, B route, C proposed route, C executed path/source. Full reasons and evidence IDs remain in linked traces and adjacent examples. | S4 | S4: `![Routing evidence and actual dispatches](graphs/atsf02-table3.png)` |
| `atsf02-graph3.png` | Four preselected cases: full dense anchor, full sparse anchor, short dense anchor, new dense anchor. Show allowed history, final actuals, A/B/C forecasts; label shared curves if identical, and show an explicit no-forecast status for abstention. | S5 | S5: `![Frozen forecasts and their final observations](graphs/atsf02-graph3.png)` |
| `atsf02-table4.png` | Two table blocks: full-history A/B/C/D coverage, common-delivered and scale-defined counts, and macro MASE on the identical scale-defined subset; then four constructed cases' A/B/C MAE, with missing results as N/A. State excluded-scale count in adjacent prose. Per-item MAE/RMSE and paired differences remain in the numerical record. | S5 | S5: `![Forecast errors with coverage and scenario boundaries](graphs/atsf02-table4.png)` |
| `atsf02-table5.png` | Six cases: two ambiguity, two gate, two response fixtures; expected behavior, observed B/C or fixture status, and calls/fallback source. Label constructed/injected provenance in cells. | S6 | S6: `![Missing information and bounded failures](graphs/atsf02-table5.png)` |
| `atsf02-table6.png` | Stacked blocks: primary 16-case audit/coverage/fallback counts; archived A/B/C dispatch seconds, calls and input/output tokens; current numerical duration/fit counts with separate D rows. Identify cohort/session in column headers/values, not titles. Put two-case ambiguity costs in separate rows and prose; exclude gate/fixture work from primary totals. | S7 | S7: `![Evidence quality and measured routing work](graphs/atsf02-table6.png)` |

S8 refers to the existing Tables 3, 4 and 6 by number; if an image is repeated, reuse its exact path, never allocate a new number. Predetermine representative IDs as above, so illustrative plots are not selected for success. Full traces and per-request scores remain inspectable even when only four curves are illustrated.

Write `examples/module02/assets.json` containing filename/kind/order, exact section title, generating cell ID, source-result hash, PNG hash, dimensions, DPI, resolved font, caption and intended post path. Each table gets a JSON sidecar containing typed raw values and displayed strings. Display the saved PNG in the notebook; the post reuses those bytes. Any additional plotted figure or rendered DataFrame requires export, inventory and sequential numbering; prefer ordinary text/JSON for small diagnostics to keep the nine-asset scope.

## 8. Figure and table specification

Reuse the actual Module 1 configuration:

```python
class CFG:
    img_dim1 = 20
    img_dim2 = 10
    publication_scale = 0.65
    export_dpi = 180
    table_fontsize = 13
    table_decimals = 2
    table_header_color = "#25455d"
    table_alternate_color = "#eaf0f4"
    table_wrap_chars = 100
    table_line_height = 0.22
    table_min_height = 1.5
    SEED = 42

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (CFG.img_dim1, CFG.img_dim2)
np.random.seed(CFG.SEED)
```

Let the style establish the sans-serif family and normal font/grid settings; record the resolved font file (Arial, Liberation Sans, or DejaVu Sans fallback) and Matplotlib version. Use history blue `#4C72B0`, holdout actuals dark gray `#222222`, baseline A purple `#8172B3` dotted, B green `#55A868` dashed, and C orange `#DD8452` dash-dot with sparse circle markers. Observed lines have width 2. Give donor histories a visibly distinct pale line/legend label. Use line styles/markers as well as color; do not conceal identical curves. Any fitted curve introduced would use red and require explicit inventory.

Every size derives from CFG: use publication width `img_dim1 * publication_scale`, appropriate height multiples of `img_dim2 * publication_scale`, and row-based table heights. Use explicit units, readable legends and constrained/tight layout. Save with white background, `dpi=CFG.export_dpi`, `bbox_inches="tight"`; close figures after saving. Graph titles/axis labels are allowed; captions remain in prose.

For **every rendered table**, including multi-block images:

1. No title inside the image. No subtitle/block title, footer, footnote, caption, source line, or other free text at the bottom. Identify blocks with meaningful column headers only; all explanations belong in the surrounding prose.
2. Format all floating-point values with `f"{value:.2f}"`, including floats numerically equal to an integer; retain full precision in calculations and JSON. Keep integer counts as integers, dates as dates, categories as text, and unavailable values as `N/A` with their reason in prose/records.
3. Apply the navy header, white header text, alternate pale rows, white borders and left alignment already used by `PublicationAssets`. Wrap long text and size rows from CFG. Never shrink text just to fit an overcrowded table.
4. Reuse/refactor the existing table exporter to parameterize module prefix, post path and sidecar location; its current `atsf01` constants must not overwrite Module 1. Preserve its title/free-text guard and typed-format sidecars.
5. Keep Tables 3 and 5 to six compact columns; Tables 4 and 6 use stacked blocks instead of a wide spreadsheet. If Table 6's work block is too wide, split it into routing and numerical blocks within the same PNG. No added titles are permitted.
6. Inspect all nine PNGs at native size and approximately 760-pixel post width. Fix clipping, wrapping, tiny text, ambiguous legends and excessive whitespace before delivery. Numerical sidecars do not replace visual inspection.

## 9. Ordered implementation steps and files

These are future actions, not work performed during planning. All paths below are relative to this repository unless stated otherwise.

| Step and dependency | Files and work | Completion criterion |
|---|---|---|
| 1. Freeze scope and environment | Read current instructions/status; `uv sync --locked`; inspect existing APIs; write `examples/module02/protocol.json` with source hashes, dates, route order, candidate settings, model settings, case order and representative IDs | No unresolved choice affects routing/evaluation; versions and source manifest verified; no unrelated working-tree change |
| 2. Prepare restricted views; after 1 | Add `scripts/prepare_module02_data.py`, `data/module02/cases.json`, `manifest.json`; read existing `data/a02` subset without copying or changing it | Exactly 16 primary requests plus four labeled input cases; metadata origin and synthetic launch/donor assumptions recorded; target/donor cutoffs validated |
| 3. Extend reusable methods/contracts; after 2 | Extend `agentic_ts/routing_forecasting.py` with short-history/donor procedures; keep existing numerical functions; extend `agentic_ts/routing.py` via a versioned Module 2 policy/adapter while preserving legacy defaults. Reuse generic JSON/hash/HTTP utilities; remove hardcoded two-route assumptions only in the new contract. Add `tests/test_module02_routing.py` | All four paths and clarification are executable; no model-generated arithmetic; failures and ineligible dispatches are explicit; new and affected legacy tests pass |
| 4. Author capture and replay; after 3 | Add `scripts/capture_module02.py`; create immutable `examples/module02/recorded_run.json`; measure A/B dispatches in that session; preserve protocol, cohorts and incremental failure records; exact replay implementation in the routing module | All 18 planned model attempts or their actual failed-attempt statuses recorded; at least one genuine numeric dispatch or a declared mechanism blocker; replay rejects altered inputs |
| 5. Execute numerical benchmark and audit; after 4 | Add `scripts/evaluate_module02.py`; write `examples/module02/frozen_forecasts.json`, `numerical_results.json`, `routing_audit.json`, `work.json`, and `failure_fixtures.json` | Forecasts frozen before scoring; observed and injected failures separated; per-request metrics, paired denominators and actual work reconstructible |
| 6. Build notebook and assets; after 5 | Create `a02-routing.ipynb`, `scripts/execute_module02.py`; generalize `agentic_ts/module01_assets.py` with backwards-compatible parameters or extract a shared exporter with Module 1 wrapper; create the nine named PNGs and sidecars/asset manifest | S1–S8 matched; explanatory cells complete; fresh-kernel offline replay succeeds; exported images displayed and visually inspected |
| 7. Write companion post; after executed 6 | Create `a02-post.md`, `examples/module02/post_claims.json`; update only A02 README entry/run instructions; `.gitignore` currently needs no change | Dedicated opening context; exact headings and image references; actual outcomes and costs, not expected wins; evidence mapped to every empirical claim |
| 8. Verify and record; after 7 | Add `scripts/verify_module02_publication.py`, `examples/module02/publication_verification.json`, `docs/module-2-implementation.md` | All checks below pass, including semantic/visual review; report any failed completion gate instead of claiming a finished module |

If a new configuration cannot be added cleanly without altering legacy behavior, introduce a thin `agentic_ts/module02_workflow.py` adapter that composes existing functions. Do not create a parallel forecasting implementation. Update dependency/lock files only if a demonstrated need appears; the planned work uses installed dependencies.

Expected reader commands after implementation: `uv sync --locked`, `.venv/bin/python -m scripts.execute_module02`, `.venv/bin/python -m unittest discover -s tests -p 'test_module02_routing.py' -v`, and `.venv/bin/python -m scripts.verify_module02_publication`. Author capture/preparation commands are separate from offline reader execution and refuse to overwrite original archives.

## 10. Validation and completion gates

### Meaningful numerical/workflow checks

- Verify calendar, duplicates, target/units, horizon, view masks and donor metadata. A missing date must not become a zero; zero history must not equal an all-zero series. Missing horizon/data errors make zero model/forecast calls.
- Perform leakage mutations on in-memory copies **after verified loading**, bypassing only the file-integrity layer for this test. Change only post-origin outcomes and assert unchanged permitted views, features, prompts, route proposals under replay, candidate selection, and frozen predictions. Change masked older target observations for short/new cases and assert the same. Donor post-origin changes cannot affect the new-product prior. Whole-file provenance hashes stay out of pre-evaluation request digests.
- Any changed authorized pre-origin donor value must change its view hash and invalidate that request's exact replay, but need not change a median forecast. Use a hand-calculable fixture where changing the median donor mean changes the prior by a known amount, plus a nonmedian mutation that leaves the prior unchanged. Separately corrupt a copied on-disk source file and verify rejection by its whole-file manifest hash. Provenance rejection and absence of temporal leakage are different checks.
- Test route precedence, 55/56 and 448/449 observation boundaries, and the exact zero threshold using integer fixtures. Test zero/positive counts, all-zero data, unknown lifecycle, missing analogy permission, invalid evidence IDs, unknown routes, wrong question fields, and output truncation.
- Verify seasonal-naive and mean forecasts against hand-calculable arrays; donor prior against a small three-donor example; short and full fold dates against expected windows; finite/nonnegative length-28 forecasts; TSB edge behavior against the pinned API. Test optimizer/fold failures, tie order, final fallback and baseline failure.
- Test that an eligible valid C disagreement genuinely executes the proposed branch; an ineligible proposal is rejected; valid clarification executes no forecaster; malformed response gets exactly one deterministic fallback; failures never trigger hidden retries.
- Test exact replay input and response hashes, model/schema/policy/config mismatch rejection, and distinction between observed capture failures and replay-integrity errors. Recompute metrics independently from frozen forecasts and dated outcomes. Preserve full precision before formatting.
- Run affected existing tests for shared routing/numerical/export changes. Avoid broad unrelated environment repair or refactoring. If the documented macOS SciPy loader issue recurs, follow the existing repair procedure only for the actual observed failure.

### Clean execution and publication checks

- Run the notebook from a fresh kernel, repository-root working directory, using the locked project interpreter. No hidden state, source-repository path dependency, network request, Ollama server, or stale saved numerical results may be required for default replay. Recompute forecasts; cached language responses must be genuine and input-matched.
- Require executed output for every code cell without traceback. Check default replay made zero live calls and that current numerical hashes match the audited records; timings are allowed to change and are excluded from semantic hashes.
- Parse Markdown headings outside fenced code: substantive level-two titles in notebook and post must equal S1–S8 exactly, including order, spelling, punctuation and capitalization. Permit notebook-only Setup/Utils and optional post-only Sources; do not compare document titles or lower-level explanatory subheadings.
- Require exactly `atsf02-graph1.png`–`atsf02-graph3.png` and `atsf02-table1.png`–`atsf02-table6.png` for this plan, with separate contiguous sequences matching first notebook appearance. All displayed graph/table images must be exported. Verify PNG validity, source-result/PNG hashes, settings, manifest and notebook displayed bytes.
- Verify typed table sidecars: every float displays exactly two decimals, every integer count remains integer, missing values are `N/A`. Inspect table artists and PNGs for forbidden title/footer/free text and for legibility. Generalized export must leave Module 1 outputs/configuration unaffected.
- Parse every post image reference; resolve it relative to `a02-post.md`; verify intended file/hash, section placement and inventory membership. Repeated references resolve to the same file. Verify links to notebook, evidence records and README entries.
- Map each empirical post claim/caption to a result field or audited response in `post_claims.json`. Check all numbers, directions, denominators, dates, workflow/model names, timing provenance, missing results, coverage and fallback attribution against the executed run. Conduct a manual semantic read, not only string/number matching.
- Check that source links support the historical/context claims actually made, no unmeasured adoption/productivity/superiority claim appears, and synthetic views/fixtures are clearly distinguished from genuine data/model observations.

Completion means the four request types are represented, B and genuine C routing are inspectable, their forecasting results and actual work are available with coverage/failures, the offline notebook runs, and the post explains the same evidence with matching headings and reused inspected assets. It does not require C to outperform B, avoid errors, or save computation.

## 11. Assumptions and blockers

Routine choices resolved from the project: English text, root-level notebook/post, M5 observed sales, 28-day forecasts, 60-day short views, a deliberately simple donor prior, local Ollama, no framework, and nine publication assets. Existing data are reused but numerical results and model decisions will be newly generated under the revised contract.

There is no unresolved blocker to this **planning** deliverable. Implementation prerequisites are an available configured local model for a genuine capture, successful locked numerical execution, and verification of the existing data/provenance. These were not executed during planning. If the backend is unavailable or all genuine decisions fail to dispatch, report that concrete implementation blocker; do not fabricate responses, silently change the primary model, or present fixtures as evidence of successful routing.
