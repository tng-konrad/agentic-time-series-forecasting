# Module 2: independent critique

Reviewed 24 October 2026 by a separately launched AI critic after the [initial plan](module-02-plan-initial.md) was written. This is a design review, not a forecasting experiment, model benchmark, or independent human review. The critic read the plan and A01's numerical/reporting modules. The primary agent will add its analysis and integration decisions after this review; the critic has not edited the plan.

## Overall assessment

The architectural progression is right. A01 used a fixed numerical path followed by alternative reports; A02 should let a bounded routing decision change the numerical work actually executed. The proposed M5 panel gives that decision a concrete consequence, and two numerical paths plus a clarification outcome are sufficient.

The main weakness is the size of the claim relative to the router's job. In the proposed controlled experiment, Python has already resolved request completeness and target suitability, and the LLM is told to reproduce a zero-fraction threshold on twelve clean structured requests. That can teach conditional dispatch and its failure modes. It supplies little reason to expect an LLM advantage. The plan acknowledges this, but its central question and publication outline still risk promising a richer problem diagnosis than the experiment measures.

Keep the module. Make its purpose narrower and more explicit, add a few genuinely borderline histories without enlarging the panel, and keep operational correctness separate from the question of forecast quality. Do not add natural-language parsing, anomaly agents, inventory decisions, or another reporting chain to rescue the LLM's usefulness.

## Findings and recommended changes

### C1 — High: the LLM is primarily a policy-execution control

The 0.50 policy determines the requested family from one supplied number. Giving both that policy and number to C makes the main comparison a policy-compliance exercise. Schema-valid deviations can affect accuracy, but a lucky departure from an explicit instruction does not demonstrate better problem understanding.

**Recommendation:** state that A02 teaches **bounded dispatch with numerical consequences**, using the LLM as an inspectable alternative to a simple classifier. Make these separate questions explicit:

1. Does each implementation honor the requested policy and dispatch exactly one permitted path?
2. What forecasts does that policy deliver on this historical panel relative to the fixed path and an unrestricted numerical candidate selector?
3. What extra failures, abstentions, and inference work does C introduce?

Do not describe threshold agreement as diagnosis accuracy or imply that C's knowledge of product names improves routing. Keep the null-result conclusion available: rules are sufficient for this particular structured problem. A later module can introduce semantic objectives or information gathering.

### C2 — High: extreme-only examples make the decision look easier than it is

Six almost-always-positive histories and six strongly sparse histories put the teaching panel far from the decision boundary. A router can look convincing by recognizing two extremes. The lexicographic selection also yields twelve FOODS items; this is not a representative cross-category M5 sample.

**Recommendation:** keep twelve series, but use **four dense, four sparse, and four boundary cases**, chosen at the first training origin by the same deterministic procedure. Define the boundary stratum as zero fraction 0.40–0.60 and retain the existing extreme predicates. The primary agent's additional read-only audit reports 763 eligible boundary series, including `FOODS_1_003_CA_1_validation`, `FOODS_1_005_CA_1_validation`, `FOODS_1_014_CA_1_validation`, and `FOODS_1_016_CA_1_validation`. Verify and freeze those during preparation, without inspecting validation or final errors.

Report the original sampling stratum separately from the final-origin route: a later rolling window can change zero fraction. Use integer `zero_count` and `observed_count` in the evidence so the threshold is reproducible; display a rounded percentage only for readers. Keep one exact-threshold unit fixture even though 365 observations cannot produce precisely 50% zeros. No extra real series are needed.

Do not select for strong weekly seasonality merely to force Holt–Winters wins. Dense sales need not be weekly seasonal, and sparse sales can also have a weekly pattern. Call these **candidate workflows**, not proven diagnoses of the underlying demand process.

### C3 — High: distinguish model-validation folds from a router backtest

The plan calls for twelve genuine LLM decisions at the final issuance origin, while the three earlier folds select numerical candidates. That is a coherent, bounded design. It is not a three-origin evaluation of the entire routing system.

**Recommendation:** say explicitly that the router runs **once per series at the final origin**, using only the current 365-day history. The earlier folds evaluate fixed numerical candidates; C does not replay its final decision into earlier origins and does not get their losses. The chosen family then restricts the candidate set from which Python selects a model. This can reuse previously computed fold results without giving the router access to them.

If an implementation later runs the router at every fold, that is a different experiment with more LLM calls and must be labeled separately. No such extension is necessary for the module. Assert exact dated slices and that selection/profiling never sees `d_1886` onward before all final forecasts are frozen.

### C4 — High: clarify which clarification decisions belong to Python

All four authored request/data variants are already handled by deterministic checks. They are useful interface demonstrations, but cannot establish that an LLM recognizes unsupported requests. Meanwhile, a `clarify` decision from C on the fully specified core requests is an avoidable abstention unless it identifies a real unmodeled defect.

**Recommendation:** define a small status table and precedence rather than leaving “missing required fields,” “units inconsistency,” and “invalid input” overlapping:

| Situation | Result | LLM calls | Numerical paths |
|---|---|---:|---:|
| Invalid calendar, duplicate key, nonfinite or negative value | `invalid_input` | 0 | 0 |
| Missing horizon/target, or requested target/units not supported by the supplied data | `clarify` with one precise question | 0 | 0 |
| Fully supported request | Router decides a permitted path | At most 1 for C | At most 1 |
| C asks an unnecessary question on a core case | `abstained`, retaining the model's question | 1 | 0 |

Separate external request corrections from model retries. A corrected request is a new versioned request, not an automatic second model attempt. The shared gate should verify `as_of` and exclude rows after it before computing any profile; otherwise a correct final-holdout seal can coexist with a request-date leak.

Three teaching outcomes are enough: regular-candidate path, sparse-candidate path, or clarification. `invalid_input` is a gate error status, not a fourth specialist.

### C5 — High: fallback can conceal the LLM's own failure rate

When C times out or produces malformed JSON, the rules deliver a forecast. A combined C-system forecast table can therefore look as accurate and complete as B while the LLM contributed no accepted decision. Valid clarification decisions, by contrast, produce no forecast and reduce coverage. These are materially different outcomes.

**Recommendation:** retain one result row per original request with separate columns for `proposed_route`, `accepted_route`, `decision_source`, `dispatch_status`, `fallback_reason`, and `forecast_status`. Publish at least accepted-C decisions out of 12, C abstentions out of 12, rule fallbacks out of 12, and delivered forecasts out of 12. Count shared-gate examples separately from those 12.

For error comparisons, show the deployed result with rule fallbacks clearly labeled and show accepted-C cases separately. If C abstains, compare errors on the common delivered subset and state its exact size; do not rank methods on different silently selected denominators. No invented numeric loss is needed for a missing forecast. A successful fallback is a property of the surrounding workflow, not evidence of successful LLM routing.

### C6 — Medium: the all-candidate reference is valuable, but should stay a diagnostic

The union of the two candidate sets has only three models. A numerical selector may be simpler and cheaper than routing through an LLM. Omitting that comparison would hide an obvious alternative. Presenting it as a fourth full architecture, however, risks crowding the routing lesson with another experiment.

**Recommendation:** retain the all-candidate validation selector, deriving it from the same candidate/fold records already computed for inspection. Give it a column in the forecast table and a short explanatory paragraph. Do not require another prompt, capture, dedicated diagram, or full notebook narrative for it.

Distinguish **live dispatch work** from **offline counterfactual evaluation work**. Computing every candidate in advance is fine for a study, but an executor that just reads the resulting forecasts has not demonstrated runtime savings from invoking only a branch. Require at least the two fixed walkthroughs to show actual branch invocation and a call trace. Count validation/final fit attempts, including failed ones, separately from LLM requests. Keep end-to-end speed comparisons outside the core.

### C7 — Medium: simplify evidence and make unsupported interpretations visible

Mean positive size and lag-7 autocorrelation can help explain a history, but neither participates in the declared threshold policy. Extra profile fields create more opportunities for unsupported narrative rationalization, and zero fractions alone do not identify the causes of zero sales.

**Recommendation:** the routing packet needs request metadata, history span/count, `zero_count`, `positive_count`, a derived zero fraction, and a statement that the target is observed sales with no availability evidence. Keep any extra descriptive features in the numerical inspection table, not in the required routing schema, unless the routing policy actually uses them.

Enforce conditional fields: nonempty known evidence IDs for every decision; a nonempty question only for `clarify`; null question for a forecasting route; bounded reason text; no unknown keys. Audit the reason for unsupported seasonality, stockout, obsolescence, or latent-demand claims. Evidence-ID validity alone cannot establish those interpretations.

The nonnegative clamp should be a fixed numerical output policy applied identically within every control and included in replay/configuration fingerprints. Do not call fractional point forecasts predicted integer transactions.

### C8 — Medium: pin a short implementation path and avoid another framework

A01 already has generic `OllamaClient.complete(messages, schema)`, `ModelConfig`, immutable JSON, hashing, and strict JSON parsing. Its report contracts and ABS-specific loader are unsuitable for direct reuse. Writing a new network client, generic agent runtime, or forecasting registry would consume effort without teaching routing.

**Recommendation:** reuse those generic helpers without changing A01 if possible. Two small new modules are acceptable, but make their interfaces concrete and keep them limited to the fixed daily problem. Default replay should recompute profiles and numerical results, match the routing inputs exactly, and label archived versus current execution times. Preserve all twelve first-attempt raw responses in stable order and never overwrite the original capture.

Add a consistency check between the request's structured fields and text by fixing the core sentence from those fields. Otherwise the same fixture could say “28 days of recorded sales” while requesting a different target or horizon in structured form. Unrestricted intent extraction is outside this module; do not introduce an LLM parser to resolve it.

## Smallest coherent revised version

1. **One dataset and one common task:** twelve daily CA_1 series, four dense/four sparse/four boundary, and a fixed 28-day recorded-sales request.
2. **One shared gate:** reject invalid data or request clarification for missing/unsupported request details before any LLM or forecasting call.
3. **Two fixed numerical workflows:** seasonal naive plus weekly additive Holt–Winters; seasonal naive plus fixed-parameter TSB. Use the existing three candidate-validation folds and one final holdout.
4. **Three dispatch implementations:** fixed regular path, explicit zero-fraction rule, one LLM routing attempt per clean final-origin request. C executes an allowed disagreement; schema failures use labeled B fallback; valid abstentions remain visible.
5. **One inexpensive numerical reference:** all-candidate validation selection from the shared numerical records, reported as a comparison column.
6. **Three illustrative traces:** predetermined dense and sparse requests with real dispatch, plus one shared-gate clarification case. Keep the remaining gate variants and one out-of-menu injection compact or in tests.
7. **One audit and one forecast table:** route/source/status coverage on all twelve, matched-case numerical results, and no claim that policy labels are optimal model classes.
8. **A restrained article:** request and data → contract/gate → allowed workflows → A/B/C → observed decisions and errors → overhead/limits. Use the same main titles in the notebook and article from the start.

The numerical reference, replay record, and complete exported asset index remain useful, but they need not each become another article section. The plan should target a readable worked example rather than a cell count or a fixed number of publication images.

## What to retain

- The progression from A01's reporting chain to a routed numerical workflow.
- A real heterogeneous daily dataset already present in the repository and a small offline bundle.
- Selection before validation/final outcomes, exact calendar boundaries, and preservation of every selected series.
- A shared conventional baseline, fixed numerical tie/failure policies, and no holdout-based route labels.
- Visible dispatch authority: a schema-valid disagreement must not be silently changed into the rule's answer.
- Observed-sales terminology, no claims that zeros establish stockouts, and no unsupported inventory recommendations.
- One attempt per eligible request, exact replay matching, genuine failures retained, and no LLM reporter added after routing.
- Matching notebook/post titles, the existing CFG/Utils structure, two-decimal tables, clean images, and precise asset placeholders.

## Review limits

No models, LLM calls, new data extraction, or implementation were executed for this critique. Source counts and the boundary-stratum audit are evidence supplied by the primary agent; the critic did not independently reproduce them. This review identifies design risks and a coherent reduction in scope. It cannot establish routing quality, numerical performance, local runtime, or whether a human reader will prefer the resulting article.

---

## Primary-agent analysis and integration

Completed 24 October 2026. The initial plan was saved before the critic was launched. The critic independently wrote the review above and did not edit either plan. The primary agent then assessed the recommendations and wrote the [revised build plan](module-02-plan.md). The [initial draft](module-02-plan-initial.md) remains unchanged. This is one critique-and-revision cycle, not a second independent review of the final plan.

### Main design judgment

The central criticism is correct: providing a zero-count threshold to an LLM does not create an empirical test of forecasting diagnosis. The module is still useful as the next architectural step because a decision now controls an actual numerical branch, whereas A01's language stages only reported frozen numerical results. The revised plan makes that difference explicit and separates policy compliance, numerical consequences, and the added operational failure modes.

Adding open-ended intent extraction or anomaly investigation would make the task less artificial, but would introduce another architecture and obscure the lesson. That expansion is deferred. The rule-based router and the three-candidate numerical selector remain serious alternatives. A null result for LLM usefulness is an acceptable and informative conclusion.

### Disposition of the findings

| Finding | Decision and rationale | Changes in the revised plan |
|---|---|---|
| C1: policy execution versus diagnosis | Accepted. The architecture can be demonstrated without claiming additional forecasting intelligence. A policy deviation's final error cannot retroactively make it compliant or demonstrate understanding. | Sections 1, 4, 5, and 7 define three distinct questions, an explicit integer-count policy, and separate compliance/forecast labels. |
| C2: extreme-only teaching sample | Accepted with a small modification. Keep 12 histories, replacing four extremes with boundary histories. Select two boundary cases below and two at/above the threshold, rather than the first four boundary IDs regardless of side. This improves the illustration without adding cases or selecting on forecast outcomes. | Section 2 gives the four strata, exact IDs, counts, initial selection origin, and the one-store/FOODS sampling limitation. The original sampling stratum is distinct from the final route. |
| C3: routing versus candidate backtesting | Accepted. Twelve final-origin routing calls are not a three-origin router evaluation. The final profile can use data known at issuance, but not final outcomes. | Section 6 states one C attempt per series at the final origin, gives every exact training/evaluation slice, and seals family/model decisions before final scoring. |
| C4: precheck and clarification responsibilities | Accepted, with explicit as_of resolution before dated-prefix validation. Missing/unsupported requests and invalid histories have different statuses. Constructed gate cases do not evaluate LLM understanding. | Sections 3–4 specify precedence, supported request fields, prefix-only validation, one concrete clarification question, conditional response fields, and a new versioned request for an external correction. |
| C5: fallback hides LLM failure | Accepted. An accepted LLM route, an LLM abstention, a rule fallback, and a numerical-fit fallback must remain independently observable. | Section 7 defines proposal/accepted/dispatched/source/status fields; publishes counts out of all 12; separates accepted-C and delivered-system results; and requires matched-case error denominators. |
| C6: all-candidate reference and real dispatch | Accepted. The reference is necessary because the entire numerical catalog has only three cheap models, but a fourth full architecture narrative would crowd the lesson. Cached counterfactuals cannot establish computational savings. | Sections 5–7 retain one derived diagnostic column, require selected-path execution and uncached walkthroughs, isolate offline counterfactual work, and remove end-to-end speed claims from the core. |
| C7: excess evidence and unsupported interpretation | Accepted. The declared policy uses counts, not autocorrelation or mean positive size. Extra fields would encourage unsupported explanations. | Section 4 limits the required packet, enforces conditional JSON fields, and distinguishes semantic support from schema/reference validity. Sections 5 and 8 fix the output clamp and fingerprint numerical policy. |
| C8: implementation burden and generic reuse | Accepted. The local client/JSON helpers already exist; the ABS and report contracts do not generalize directly. | Sections 8 and 10 reuse ModelConfig, OllamaClient, FrozenJSON, hashing, and strict parsing, while limiting new code to the fixed daily task. Core text is generated from structured fields; no extra parser or framework is introduced. |

### Additional primary-agent corrections

- **History length:** a complete execution needs 449 known daily observations: a 365-day first training window plus three 28-day past validation windows. The initial minimum of 365 was insufficient. The current profile and each individual fit still use 365. The final plan separates these requirements explicitly.
- **Boundary balance verified without future values:** at the first selection origin, the below-half group contains 371 eligible IDs and the at/above-half group 392. The selected below-half IDs have 182 and 168 zeros out of 365; the selected at/above-half IDs have 215 and 219. These are not holdout routing labels, and they may change at final issuance.
- **Calendar and bundle size:** independently checked `d_1437` = 4 January 2015, `d_1801` = 3 January 2016, `d_1885` = 27 March 2016, and `d_1913` = 24 April 2016. The normalized target remains 477 × 12 = 5,724 rows; no bundle was created during planning.
- **Preserve the existing module:** do not change A01's current code, manually edited notebook explanations, section alignment, post, or assets while building this plan. The module-2 inputs and contracts live in separate files.
- **Scope stays small:** retain the same two numerical paths, one source store, twelve first-attempt C calls, one injection, and no LLM reporter. The extra complexity introduced by fallback accounting is required to interpret results rather than an invitation to build a general recovery system.

### Verification and remaining evidence requirements

Planning inspected the repository's M5 archive headers and calendar, reproduced counts/IDs using only the first-origin selection window, hashed the existing source files, and inspected the reusable A01 interfaces. Primary web references for routing terminology, TSB's interface, and M5 source context are linked in the final plan.

The final artifact was checked for valid local document links, consistent dates/selection counts, coverage of C1–C8, and preservation of the initial draft and independent critique. No numerical model fits, local-model calls, module-2 notebook execution, new dataset extraction, or article implementation were performed. Suitability of each candidate, actual routing outcomes, failure rates, runtimes, and human editorial assessment are build-time evidence requirements, not conclusions of this review.

### Subsequent user amendment: publication assets

After the critique-and-integration cycle, the user explicitly confirmed the A02 publication requirements: export all notebook graphs/tables to `graphs/` as independently numbered `atsf02-graph<number>.png` and `atsf02-table<number>.png`; use the existing `seaborn-v0_8`/CFG plotting setup; remove all table-image titles and bottom text; display decimal table results to two places; and make every post placeholder identify the correct graph/table path. The revised plan now spells these out in section 9 with example Markdown, an export-registry/asset-index requirement, and corresponding acceptance checks in section 10. This is a user-directed clarification of the build specification, not a second independent critic review. The initial draft and original critique remain unchanged.
