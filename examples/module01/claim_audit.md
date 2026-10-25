# Module 1 editorial evidence audit

Codex-assisted editorial inspection against source facts plus independent arithmetic checks; not a blinded human study or model self-grade.

Each numerical estimate, difference, model component/status, date/window, and scope assertion is an atomic factual claim. Hypotheses and completeness omissions are separate.

The original capture is immutable. No wording below repairs a model response. Counts describe this case, not population reliability.

| Path | Supported / total | Contradicted | Unsupported | Ambiguous |
|---|---:|---:|---:|---:|
| A | 16 / 17 | 0 | 1 | 0 |
| B | 18 / 18 | 0 | 0 | 0 |
| C | 19 / 19 | 0 | 0 | 0 |

## Path A

Required structure, selection rule, final date range and units are explicit. One true date claim lacks its direct supporting citation; the original template is preserved.

- **A.1 — supported:** hw_add was selected. Evidence: forecast.delivery, validation.results. Directly supported by the cited evidence.
- **A.2 — supported:** Selection used validation RMSE. Evidence: protocol.selection, validation.results. Directly supported by the cited evidence.
- **A.3 — supported:** The criterion was mean RMSE across three folds. Evidence: protocol.selection. Directly supported by the cited evidence.
- **A.4 — unsupported:** Validation years were 2022, 2023 and 2024. Evidence: protocol.validation. The dates are correct in the evidence packet, but this template claim does not cite protocol.validation; its cited validation.results contains scores without year labels. Citation completeness fails.
- **A.5 — supported:** hw_add was delivered. Evidence: forecast.delivery. Directly supported by the cited evidence.
- **A.6 — supported:** Delivery succeeded without fallback. Evidence: forecast.delivery. Directly supported by the cited evidence.
- **A.7 — supported:** The final evaluation covers January–December 2025. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.8 — supported:** Delivered RMSE rounds to 3.00. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.9 — supported:** Delivered MAE rounds to 2.46. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.10 — supported:** Baseline RMSE rounds to 3.44. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.11 — supported:** Baseline MAE rounds to 2.83. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.12 — supported:** Errors are expressed in index points. Evidence: data.profile. Directly supported by the cited evidence.
- **A.13 — supported:** Delivered RMSE is lower than the baseline. Evidence: evaluation.final. Directly supported by the cited evidence.
- **A.14 — supported:** The revised snapshot does not recreate historical data availability. Evidence: limitations.vintage. Directly supported by the cited evidence.
- **A.15 — supported:** Only one series was evaluated. Evidence: limitations.scope. Directly supported by the cited evidence.
- **A.16 — supported:** There is one final evaluation year. Evidence: limitations.scope. Directly supported by the cited evidence.
- **A.17 — supported:** The case does not establish general performance. Evidence: limitations.scope. Directly supported by the cited evidence.

## Path B

Valid structure and correct values; does not explicitly state index-point units or three-fold averaging in the report text.

- **B.1 — supported:** hw_add was selected. Evidence: forecast.delivery, validation.results. Directly supported by the cited evidence.
- **B.2 — supported:** Selection used validation RMSE. Evidence: protocol.selection, validation.results. Directly supported by the cited evidence.
- **B.3 — supported:** hw_add has a damped additive trend. Evidence: catalog.hw_add. Directly supported by the cited evidence.
- **B.4 — supported:** hw_add has additive seasonality. Evidence: catalog.hw_add. Directly supported by the cited evidence.
- **B.5 — supported:** The seasonal period is twelve. Evidence: catalog.hw_add. Directly supported by the cited evidence.
- **B.6 — supported:** Delivery succeeded without fallback. Evidence: forecast.delivery. Directly supported by the cited evidence.
- **B.7 — supported:** The final evaluation covers January–December 2025. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.8 — supported:** The delivered model is hw_add. Evidence: forecast.delivery. Directly supported by the cited evidence.
- **B.9 — supported:** Delivered RMSE rounds to 3.00. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.10 — supported:** Delivered MAE rounds to 2.46. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.11 — supported:** Baseline RMSE rounds to 3.44. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.12 — supported:** Baseline MAE rounds to 2.83. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.13 — supported:** Delivered RMSE is lower than the baseline. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.14 — supported:** Absolute RMSE reduction rounds to 0.43. Evidence: evaluation.final. Directly supported by the cited evidence.
- **B.15 — supported:** Only one series was evaluated. Evidence: limitations.scope. Directly supported by the cited evidence.
- **B.16 — supported:** There are three validation years. Evidence: limitations.scope. Directly supported by the cited evidence.
- **B.17 — supported:** There is one final evaluation year. Evidence: limitations.scope. Directly supported by the cited evidence.
- **B.18 — supported:** The case does not establish general performance. Evidence: limitations.scope. Directly supported by the cited evidence.

## Path C

Valid structure and correct values; does not state index-point units. Adds an unrequested, one-decimal derived percentage.

- **C.1 — supported:** hw_add was selected. Evidence: forecast.delivery, validation.results. Directly supported by the cited evidence.
- **C.2 — supported:** Selection used validation RMSE. Evidence: protocol.selection, validation.results. Directly supported by the cited evidence.
- **C.3 — supported:** The criterion was mean RMSE across three folds. Evidence: protocol.selection. Directly supported by the cited evidence.
- **C.4 — supported:** hw_add mean validation RMSE rounds to 3.61. Evidence: validation.results. Directly supported by the cited evidence.
- **C.5 — supported:** hw_mul mean validation RMSE rounds to 4.10. Evidence: validation.results. Directly supported by the cited evidence.
- **C.6 — supported:** seasonal_naive mean validation RMSE rounds to 3.74. Evidence: validation.results. Directly supported by the cited evidence.
- **C.7 — supported:** Delivery succeeded without fallback. Evidence: forecast.delivery. Directly supported by the cited evidence.
- **C.8 — supported:** The final evaluation covers January–December 2025. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.9 — supported:** Delivered RMSE rounds to 3.00. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.10 — supported:** Delivered MAE rounds to 2.46. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.11 — supported:** Baseline RMSE rounds to 3.44. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.12 — supported:** Baseline MAE rounds to 2.83. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.13 — supported:** Delivered RMSE is lower than the baseline. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.14 — supported:** Absolute RMSE reduction rounds to 0.43. Evidence: evaluation.final. Directly supported by the cited evidence.
- **C.15 — supported:** Relative RMSE reduction rounds to 12.6%. Evidence: evaluation.final. Derived arithmetic independently recomputed from full-precision RMSE: 12.58456964%. Not a precomputed evidence field, not proof of a chaining benefit. The one-decimal addition was not requested.
- **C.16 — supported:** Only one series was evaluated. Evidence: limitations.scope. Directly supported by the cited evidence.
- **C.17 — supported:** There are three validation years. Evidence: limitations.scope. Directly supported by the cited evidence.
- **C.18 — supported:** There is one final evaluation year. Evidence: limitations.scope. Directly supported by the cited evidence.
- **C.19 — supported:** The case does not establish general performance. Evidence: limitations.scope. Directly supported by the cited evidence.

## Intermediate brief

- **brief.1 — unsupported:** Seasonal naive repeats the last twelve months. The statement is true, but its sole citation data.profile contains no algorithm definition. catalog.seasonal_naive would support it.
- **brief.2 — hypothesis:** A simple baseline for strong seasonal patterns. Candidate-suitability rationale, not an observed performance claim. Lag-12 correlation alone does not prove seasonality.
- **brief.3 — supported:** The additive model uses damped trend and additive seasonality. catalog.hw_add explicitly specifies these components.
- **brief.4 — hypothesis:** It may capture additive seasonal shifts and trend decay. Prospective rationale; no evidence yet that it fits or wins.
- **brief.5 — hypothesis:** Multiplicative seasonality may better handle proportional variation. A prospective suitability claim, not a measured advantage.
- **brief.6 — supported:** Only one series is evaluated. limitations.scope.
- **brief.7 — supported:** Only point forecasts are produced. limitations.uncertainty.
- **brief.8 — supported:** The experiment does not use real-time vintages. limitations.vintage.
- **brief.9 — ambiguous:** Catalog selection is fixed outside modeler control. The language model cannot alter the catalog, but the human modeler sets it. The word modeler obscures that distinction.

## What the writer did with the brief

- **seasonal-naive rationale — ignored:** The final writer supplies its measured score but does not reuse or repair the rationale or bad citation.
- **additive rationale — ignored:** The writer reports the validation winner. It does not discuss the earlier component hypothesis.
- **multiplicative rationale — ignored:** The writer gives hw_mul validation RMSE without discussing the proportional-seasonality hypothesis.
- **one-series qualification — retained:** The final limitation repeats limited scope. B states the same qualification without receiving the brief, so this is not proof of its causal contribution.
- **point-only, vintage and fixed-catalog notes — ignored:** These brief limitations are absent from C’s final report.

## Interpretation

The final numeric statements are correct. A has one citation-completeness failure; B and C have supported factual claims but omit units, and B abbreviates the selection rule. Supported claim counts do not prove that the chain adds value.

The report counts exclude brief hypotheses and separate completeness omissions. A useful-looking sentence or a valid reference ID is not sufficient evidence of correctness. The model reports were not changed after inspection.
