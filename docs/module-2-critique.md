# Module 2 critique and revision decisions

Reviewed and revised: 26 October 2026. Module: **One forecasting request, several possible problems**; pattern: **routing**.

A separate critic agent reviewed the initial plan against the full user request and inspected the existing routing/numerical interfaces. Its findings are preserved below. The primary author then evaluated each recommendation and revised [module-2-plan.md](module-2-plan.md). This is a review of a plan; no Module 2 notebook, full post, capture, or forecast experiment was implemented.

## Author decisions on substantive findings

| Finding | Decision | Reason and change integrated into the final plan |
|---|---|---|
| C1: donor-mutation invariant and provenance boundary | Accept | The critic is correct: a median is insensitive to some donor changes. Section 10 now separates request-hash invalidation, a known median-changing numerical fixture, a nonmedian no-change fixture, in-memory leakage tests after verified loading, and on-disk hash rejection. Whole-file provenance does not enter the request digest. |
| C2: macro MASE denominator | Accept | Delivery alone does not make scaled error defined. Section 5 defines the common delivered set and its positive-finite-scale subset, applies that same subset to every arm, reports both counts plus exclusions, and retains MAE/RMSE for undefined-scale cases. Table 4 explicitly includes the counts. |
| C3: timing cohorts and mixed-session components | Accept | Comparing different cohorts or summing unrelated runs would mislead. Section 5 and Table 6 fix the primary work cohort to 16 cases, report the two ambiguity attempts separately, exclude gate/fixture work from primary totals, record session IDs and observation counts, and prohibit calling cross-session sums measured end-to-end latency. The capture step records A/B dispatch durations alongside C. |
| C4: baseline A's lower history boundary | Accept | A shared capability envelope makes coverage comparable without adding an experiment. A now explicitly shares the 56-day floor and metadata consistency checks; the text distinguishes this teaching contract from seasonal naive's mathematical minimum. |
| C5: exact fixture derivations | Accept | This removes routine implementation ambiguity without expanding scope. Section 2 names all main case IDs and X01–X06, the source request for each, the removed date 2016-03-17, missing fields, malformed payload, and ineligible regular-on-60-days proposal. Expected behavior stays outside router evidence; injected failures make no real model call. |

No substantive finding was rejected. These changes preserve the user's requested four situations, matching section titles, exact image names, table restrictions, and planning-only scope.

## Decisions on simplification and narrative recommendations

| Recommendation | Decision and reason |
|---|---|
| Retain four numerical paths plus clarification | Accept/retain. Returning to the old two-route code would omit explicit module requirements. |
| Retain the full-history all-candidates diagnostic | Accept/retain. It isolates candidate-pruning consequences without another data source or final-outcome oracle. |
| Retain the shared fixed donor prior and avoid donor search/neural models | Accept/retain. It gives the baseline equal access to the constructed new-product capability and keeps the lesson focused. It does not validate real launch performance. |
| Keep claims narrow and separate richer-language motivation from observed evidence | Accept/retain. The suite measures a structured, policy-directed dispatcher; it cannot establish an advantage on untested free-text requests. |
| Keep nine publication assets and use text/JSON for detailed traces | Accept/retain. Every figure/table has a teaching purpose, while complete case-level evidence remains inspectable. Additional rendered outputs would still require export and inventory. |

## Additional author refinements

The final plan names the existing A02 context/output limits explicitly (`num_ctx=32768`, `num_predict=4096`, 240-second timeout), eliminating ambiguity between older and newer module adapters. It also clarifies that Module 1's historical start-from-scratch instruction applies to that earlier task; the current user request authorizes inspection and reuse of this project's conventions and methods.

## Review outcome and checks

The revised plan is self-contained and ready for implementation. The independent review found no fundamental technical or teaching blocker. Model availability, genuine inference, and numerical execution remain future implementation prerequisites, not completed validation.

The primary author's planning checks confirm eight canonical notebook/post section pairs, nine exact post image placeholders, three separately numbered graphs and six separately numbered tables, and no alternate Module 2 PNG names. Both requested Markdown files exist. The notebook clean-kernel, numerical, image, link and claim audits are explicitly planned for implementation; they have not been falsely reported as executed here.

## Preserved independent critic report

The report below is reproduced from the separate critic. Its line references refer to the initial draft, before the accepted revisions above.

---

# Independent critique: Module 2 implementation plan

Reviewed: `docs/module-2-plan.md`, draft dated by review on 2026-10-26. This is a review of a plan, not a verification of an executed notebook or model benchmark. I read the draft and checked the existing routing/numerical interfaces against the proposed extensions. I did not edit the plan or implement any module artifact.

## Overall assessment

The plan is technically credible and satisfies the substantive brief. It covers all four requested situations, provides both a simple numerical baseline and a deterministic routing baseline, and correctly distinguishes route agreement from forecasting performance. The controlled short/zero-history cases, the fixed donor prior, and the all-candidates diagnostic are appropriately labeled. The plan does not need another model, dataset, or experiment.

The main changes needed are precise validation invariants, explicit aggregation denominators, and an unambiguous cost cohort. These are bounded clarifications; none requires changing the teaching narrative or nine-asset budget.

## Substantive findings

### C1 — Medium: correct the donor-mutation assertion and separate mutation tests from provenance loading

**Location:** temporal boundaries in section 2; validation bullet in section 10 (draft lines 58–59 and 259).

The plan says an authorized pre-origin donor-value change must change the view hash/prior. The hash must change, but the forecast need not: changing a nonmedian donor mean can leave the median of three donor means exactly unchanged. Requiring every such change to alter the prior would reject correct numerical behavior. Conversely, the source CSV loader verifies a whole-file hash, so editing a held-out source row and rerunning that public loader should fail provenance verification even though the permitted request view is unchanged.

**Recommendation:** distinguish three tests. (1) Any authorized pre-origin input change invalidates exact request replay by changing its view hash. (2) A deliberately constructed mutation that changes the median donor mean changes the prior by a known amount; a mutation of a nonmedian donor may leave predictions unchanged. (3) Leakage tests operate on an in-memory copy after verified loading, comparing permitted-view/prompt/decision/frozen-forecast semantics; a separate test confirms that on-disk source corruption fails the manifest hash. Whole-file provenance must not enter the pre-evaluation request digest.

### C2 — Medium: give macro MASE its own explicit defined-scale denominator

**Location:** numerical metric rules and Table 4 (draft lines 73, 133–137 and 190).

The plan correctly preserves undefined scaled errors as `N/A`, but defines the common-subset table only by forecast delivery. Delivery does not imply a positive finite MASE denominator. If an all-constant series is delivered by all arms, implementations could silently skip its `N/A` in a mean or invalidate the whole macro metric; both are compatible with the current prose and yield different displayed denominators.

**Recommendation:** define the common delivered set first and a second set containing those requests with positive finite training scale. Compute all arms' macro MASE over the same second set; show both counts and the excluded-scale count. If no scale is defined, report `N/A`. Retain MAE/RMSE for every delivered case. Keep Table 4 compact by putting the exact count explanation in surrounding prose and adding only the essential count column(s). No epsilon substitution is needed.

### C3 — Medium: freeze timing cohorts and distinguish recorded components from an end-to-end measurement

**Location:** E4 and cost paragraph; S7 and Table 6 (draft lines 131, 141, 154 and 192).

There are 16 main requests, two live ambiguity requests, two gate cases, and two injected response fixtures. The plan asks for totals/medians over “declared cases” but does not prescribe whether Table 6's C total includes 16 or 18 model calls, while A/B are mainly described on the 16-case benchmark. It also deliberately combines archived inference evidence with recomputed numerical work, which must not be presented as a measured single-run end-to-end latency.

**Recommendation:** fix Table 6's primary A/B/C work cohort to the same 16 main cases. Report the two live ambiguity attempts separately, with gate/fixture work separately labeled and excluded from primary totals. Record exactly which session supplies each numerical timing; show archived dispatch/inference and current numerical duration as separate components. Do not sum components from different sessions and label that sum measured end-to-end time. Maintain D's separate logical/actual fit accounting. Give every median its observation count and preserve missing timing/token fields as `N/A`.

### C4 — Low: make baseline A's capability boundary explicit

**Location:** baseline A and route policy (draft lines 73, 81 and 98).

A forecasts “supported requests with observed history,” whereas the policy deliberately treats 1–55 days as unsupported. Seasonal naive itself only needs seven observations, so a future implementer could reasonably use it for 7–55 rows in A while B/C clarify, changing coverage at a boundary the validation suite explicitly tests.

**Recommendation:** state whether A shares the module's 56-day minimum capability envelope. The simplest choice for this focused benchmark is yes: A uses seasonal naive for at least 56 valid observations, the same prior for eligible zero-history requests, and otherwise clarifies. Explain that the floor is a module contract, not a mathematical minimum for seasonal naive. This does not alter the planned 16-case experiment.

### C5 — Low: make the fixed incomplete-input and injected cases fully reproducible

**Location:** request-suite additions and E3 (draft lines 50 and 130).

The four extra cases are conceptually clear but do not name which base request is cloned or which daily observation is removed. The ineligible-response fixture also does not identify the mismatched route/history pair. These choices do not require user clarification, but specifying them now would make this implementation plan self-contained down to its fixed benchmark manifest.

**Recommendation:** choose explicit derivations, for example: both zero-history ambiguity cases clone the dense anchor's synthetic new-product request; missing horizon and missing date clone its full-history request; remove one declared pre-origin interior date; inject malformed JSON on that valid full-history request and an otherwise valid `regular` route on its 60-day request. Assign stable case IDs, retain supporting evidence IDs in the ineligible fixture, and state that no fixture makes a model call. Preserve the existing six E3 rows and separate provenance labels.

## Simplification and narrative recommendations

- Retain all four numerical paths and the `clarify` outcome. Reducing back to the old two-route implementation would miss the user's explicit short-history and new-product scope.
- Retain the all-candidates diagnostic for full histories only. It answers whether candidate pruning matters without an oracle or a new experiment family.
- Retain the shared donor prior in A/B/C. It prevents a coverage advantage being falsely attributed to an LLM and makes zero-history forecasts a clear control. Do not add donor-search, neural models, target scaling, or real-launch performance claims.
- Keep the post's central claim narrow: this is an inspectable, policy-directed workflow demonstration. The richer-language motivation is context, not evidence of an advantage measured by these structured cases.
- Keep the nine-asset scope. Long traces, per-item metrics and implementation diagnostics can remain JSON/text; every actual displayed graph/table still needs its required export.

## Requirement coverage

| Requirement | Assessment |
|---|---|
| Applicable instructions and repository conventions inspected | Documented with concrete files, styling choices and recipe overrides; existing code reuse is appropriate |
| Module 2 resolved from the list | Correct title and routing pattern |
| Central question, objectives, practical data and methods | Clear and focused; all four requested cases represented |
| Simplest baseline, rule control and evaluation | Strong; C2/C4 sharpen denominators and boundaries |
| Missing/ambiguous information | Explicit router-level and gate-level cases; C5 removes remaining fixture ambiguity |
| Limitations and no invented superiority | Strong; masked launches, shared outcomes, arbitrary threshold and donor assumptions are disclosed |
| Matching substantive sections | Eight exact canonical headings and section-level notebook/post evidence map supplied |
| Opening context: origin, problem, popularity, justified complexity | Dedicated S1 and credible primary-source plan; correctly distinguishes documented visibility from measured adoption |
| Exact graph/table naming, folder and separate numbering | Three `atsf02-graphN.png` and six `atsf02-tableN.png` assets, sequential by first notebook appearance |
| Exact relative post placeholders and inventory | Explicit `graphs/...` paths relative to root-level `a02-post.md`; references are unambiguous |
| Configuration-driven plot style and reproducibility | Actual Module 1 style, dimensions, DPI, seed, palette and font resolution reused |
| Table restrictions and two-decimal results | Explicit for every image and block; typed sidecars, formatting checks and visual QA planned |
| Ordered dependencies, file plan and completion criteria | Actionable; no extra implementation is needed in the current planning task |
| Clean-kernel, leakage, links, headings and result agreement checks | Strong; C1 corrects a false-positive mutation assertion and provenance-test boundary |
| Cost and failure attribution | Strong foundation; C3 fixes cohort and timing-provenance ambiguity |
| Separate critic and accept/adapt/reject decisions | This report supplies the independent review; parent must record decisions and integrate accepted changes |
| Revised self-contained plan and requested output files | Parent must update draft status and save `docs/module-2-critique.md` with its decisions |
| Planning only | Preserved by this review; no notebook, post or experimental result was produced |

There is no fundamental technical or teaching blocker. The configured model's availability and a successful genuine dispatch remain future implementation prerequisites, as the plan already states.
