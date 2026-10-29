# Module 2 implementation record

Completed 26 October 2026 from the [reviewed plan](module-2-plan.md). The [executed notebook](../a02-routing.ipynb) and [Substack draft](../a02-post.md) share eight substantive section headings and nine exported PNGs. [SEO and LinkedIn copy](module-2-promo.md) are ready for publication. Nothing has been published externally.

## What was built

The numerical baseline A, explicit rules B and one-call LLM router C receive the same permitted data. Four numerical workflows cover full regular histories, full intermittent histories, short histories and synthetic zero-history products. Python checks inputs, computes evidence, validates proposed routes, selects candidates chronologically, fits models and scores outcomes. A valid eligible LLM proposal controls dispatch; an invalid response gets one explicit rule fallback. A valid clarification delivers no forecast.

Reusable code lives in `agentic_ts/module02_workflow.py`, `module02_publication.py` and the extended `routing_forecasting.py`. The shared `module01_assets.py` exporter now accepts an explicit prefix and output locations while preserving its Module 1 defaults. The notebook contains 27 code cells, each followed by explanatory Markdown. It imports shared code rather than duplicating numerical implementation.

Reader execution replays exact genuine model responses and recomputes all numerical forecasts. Captured inputs bind the restricted view, prompt, schema, numerical configuration and model configuration; altered inputs or responses stop replay. There is no incidental live inference. `reader_frozen_forecasts.json` is saved before final scoring, then checked against the author run, excluding durations/session timestamps from semantic equality.

## Data and numerical design

The original bundled `data/a02/m5_routing_panel.csv` is unchanged and verified against its manifest: twelve CA_1 M5 items, 477 daily sales observations each, 4 January 2015–24 April 2016. Forecast origin is 27 March 2016; the final horizon is 28 March–24 April. The twelve full requests expose 449 days. Two fixed anchors additionally expose 60 days or no target history, giving sixteen main cases. Four additional requests cover genuine ambiguity and shared input failures. Two labeled response fixtures test malformed and ineligible proposals.

Full histories use three rolling 28-day validation folds with 365 training days. Regular candidates are seasonal naive and damped additive Holt–Winters; intermittent candidates are seasonal naive and TSB with both smoothing parameters fixed at 0.10. Short histories use one 32/28 chronological split and seasonal naive versus a trailing-28-day mean, followed by final refitting. The fixed zero-history prior is the median of three authorized same-unit donors' pre-origin 28-day means. The baseline receives that same prior. No target-based scaling or hidden target validation is used.

D is a separate all-candidates comparison on the twelve full histories. It runs after main dispatch decisions and uses the same validation folds. It does not select with final outcomes or feed back to routing. Every arm is recomputed independently without a numerical cache. Fit counts include failed attempts and final fallbacks. Donor arithmetic counts as no model fit.

Selection uses full-precision validation RMSE, with complete-fold eligibility and deterministic tie order. Nonconvergence, nonfinite output and date mismatch remain failures. Negative numerical forecasts are clipped with a recorded count. Final numerical failure can recover to seasonal naive; failure of that baseline remains explicit failure. Final scoring reports MAE/RMSE and training-only lag-7 MASE for full histories. Undefined MASE scales remain N/A.

## Genuine capture and observed results

The immutable [primary capture](../examples/module02/recorded_run.json) uses local Ollama `qwen3-coder-next:q4_K_M`, temperature 0, seed 42, context 32768, output limit 4096 and timeout 240 seconds. Model identity, digest, backend details, machine, ordered inputs, raw responses, usage, load time and request durations are archived. The protocol was saved before primary inference and final scoring. Eighteen genuine calls comprise sixteen main and two ambiguity requests; two shared-gate failures make no call.

All sixteen supported main requests ask unnecessary questions. Eleven forecasting proposals violate the null-question contract and fall back to rules; five valid clarifications abstain. Zero primary LLM forecasting decisions are accepted. C delivers eleven of sixteen main forecasts, including nine of twelve full histories, entirely through rules. The [manual audit](../examples/module02/routing_audit.md) rates the sixteen reasons: three supported, eleven contradicted, one unsupported and one ambiguous. Valid JSON/evidence IDs do not establish semantic support.

The plan explicitly permits a separate genuine development run if the primary run contains no accepted forecasting decision. One [development call](../examples/module02/development_run.json) clarifies the null-question/established-product instructions and uses an equivalent nullable-schema encoding. Its accepted regular proposal controls a real numerical fit. Its response, time and numerical outputs remain separate from primary counts and scores. Two changes together and one case cannot identify the effective change or establish a robust repair. No primary response was replaced.

On the nine full-history items delivered by every arm, all with positive finite scales, macro MASE is A 1.04, B 0.89, C 0.89, D 0.84. These C results belong to the rule fallback. A/B/D deliver all twelve full histories; the common subset excludes C's three abstentions. Constructed short/new views are reported individually and never pooled into this macro score. All per-case full-precision errors and paired differences remain in the [numerical record](../examples/module02/numerical_results.json) and [summary](../examples/module02/summary.json).

Primary C dispatch uses 16 calls, 60.09 seconds, 8,372 input tokens and 2,054 output tokens; median 2.56 seconds. The first call includes model loading. Two ambiguity calls use 4.72 seconds separately. Current numerical timings come from reader execution and are not added to archived dispatch as an end-to-end latency measurement. B makes 103 fit attempts over sixteen requests, including 96 on the twelve full histories; D makes 132 over those same twelve histories. Lower C computation comes with abstention. No dollar or energy estimate is inferred.

## Reproduction and validation

Run from the repository root with Python 3.12:

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_module02
.venv/bin/python -m unittest tests.test_module02_routing tests.test_a02_routing tests.test_module01 -v
.venv/bin/python -m scripts.verify_module02_publication
```

The lockfile supplies the captured numerical versions. Reader replay needs neither Ollama nor network access after environment installation. Jupyter needs permission to bind its local kernel socket in a restricted execution environment. Author preparation/capture/evaluation scripts reject overwriting their original records; they are not prerequisites for reader execution. Rebuilding the notebook source with `scripts.build_module02_notebook` requires executing it again afterward.

Validation completed:

- 27 code cells execute sequentially from a clean kernel, with no errors and zero new model calls. Numerical semantic hashes match the author record; the separate genuine dispatch is replayed and recomputed too.
- 44 tests pass: 16 Module 2 contract/leakage/numerical tests, 15 legacy routing regression tests and 13 Module 1 tests. Leakage mutations change final outcomes without changing restricted views, prompts, routes, candidate selection or frozen predictions. Masked old target values cannot affect short/new requests. Donor sensitivity, replay invalidation and corrupted-source rejection are checked separately.
- Three graphs and six tables use exact sequential `atsf02` filenames, configuration-derived dimensions, the project style and 180 DPI. Every displayed notebook PNG is byte-identical to its saved file and the post references that file. Table sidecars preserve raw types and precision; rendered floats have exactly two decimals, counts remain integers, and table images contain no title/footer/caption.
- All nine images are inspected at native size and approximately 760-pixel post width. Overlapping date ticks/diagram text were corrected; shared forecast curves and C fallback provenance are explicit. Captions and notes remain surrounding prose.
- The publication verifier checks matching substantive headings, relative links, section placement, asset hashes, first-appearance numbering, table formatting, independently calculated final errors, source provenance, reason-audit bindings, work counts, empirical prose mappings and SEO length. [Verification](../examples/module02/publication_verification.json), [visual review](../examples/module02/visual_qa.json) and [claim mapping](../examples/module02/post_claims.json) preserve the evidence.

The old archive-dependent routing test now constructs a minimal dated forecast fixture instead of relying on the absent legacy `examples/a02/forecast_run.json`; its freeze/score assertions are retained. No deleted legacy notebook or archive has been fabricated.

The broader Module 1 publication verifier stops at a pre-existing mismatch: it expects 26 code cells, while the committed Module 1 notebook has 25. Current counts match `HEAD`. Its 13 regression tests pass and all nine existing Module 1 PNG hashes still match their manifest. This Module 2 task does not rebuild that earlier publication.

## Limits and handoff

This is a bounded workflow demonstration using one store, one final horizon, an illustrative 50% zero threshold, fixed model settings, synthetic short/new views, assumed donors and one ordered local-model capture. Repeated anchor outcomes are not independent new products. Recorded sales do not reveal stockouts or unconstrained demand. No uncertainty calibration or inventory-value claim is made. Primary failures and the separate development success are both retained.

No implementation blocker remains. The LinkedIn draft only needs the eventual public article URL when the user publishes it.
