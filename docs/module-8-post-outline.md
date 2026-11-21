# Episode 8 post outline

Executed source: primary-v3; earlier v1/v2 cohorts disclosed. Same H2 hierarchy as notebook.

## The forecast is already out there

Purpose/evidence: The [companion notebook](a08-event-driven-stateful.ipynb) joins three earlier ideas. [m07](m07-conformal-prediction.ipynb) separates model fitting from interval calibration. [m16](m16-agents.ipynb) puts numerical tools around a local language model. [m17](m17-anomaly.ipynb) distinguishes unusual readings, persistent disagreement and notification burden.

> [Insert Graph 01 here — A delivered forecast meets a changed stream]

![Graph 01: Historical and shifted hourly taxi counts against a frozen weekday-hour forecast.](graphs/atsf08_graph_01.png)

## Give an alert somewhere to live

Purpose/evidence: The statistical detector opens the investigation. At every wake, Python gathers three read-only summaries: **recent** covers the last six arrived readings; **history** covers the opening reading onward; **candidate** uses up to six post-alert residuals to estimate a median offset. The opening reading counts as the first post-alert observation.

> [Insert Table 02 here — Who owns each decision]

![Table 02: Statistical code, model, persisted controller and external review have separate responsibilities.](graphs/atsf08_table_02.png)

## Investigate as observations arrive

Purpose/evidence: Two earlier full cohorts showed this model avoiding evidence tools: version 1 made ten calls, ending on wake limits for A/B and insufficient evidence for C; version 2 rejected all three one-call cases for skipping a mandatory tool. We preserved both cohorts, then moved evidence collection into Python and froze version 3 before rerunning all three cases once. The final study is therefore exploratory and informed by earlier failures. It demonstrates fixed evidence collection with model recommendations, without establishing autonomous tool selection or an improvement across versions. The full history remains in the [evidence guide](docs/module-8-implementation.md).

> [Insert Table 03 here — The final live recommendations]

![Table 03: Every final-cohort wake and recommendation, including the rejected proposal.](graphs/atsf08_table_03.png)

> [Insert Graph 04 here — Three shapes, three different questions]

![Graph 04: A single spike, sustained disagreement and two late elevated readings, normalized by the calibration radius.](graphs/atsf08_graph_04.png)

## Pause before changing the baseline

Purpose/evidence: A valid proposal stores a case ID, baseline version, literal offset, evidence references and creation index under a proposal hash. The workflow persists `awaiting_review`. An external event must identify that proposal and its reviewer before Python can accept a new baseline. The reviewer should see those actual parameters alongside the explanation.

> [Insert Table 05 here — Review changes what can happen next]

![Table 05: Scripted approve, edit, reject and timeout events act on an eligible persisted proposal.](graphs/atsf08_table_05.png)

## Count delays, interruptions and claims

Purpose/evidence: A detector can be prompt while its investigator remains unhelpful. We therefore separate opening delay, valid recommendation delay, controller outcome, delivered forecasts and review requests. Delay starts at the known injected onset; a missing valid recommendation has a stated reason rather than an invented zero.

> [Insert Table 06 here — Detection is only the first delay]

![Table 06: Per-case opening delay, valid recommendation delay, model calls and review requests.](graphs/atsf08_table_06.png)

> [Insert Table 07 here — Check the explanation as well as the action]

![Table 07: Atomic factual audit counts, with suggestions shown separately.](graphs/atsf08_table_07.png)

> [Insert Graph 08 here — Count the work an alert creates]

![Graph 08: Raw flags, six-hour cooldown notices and deterministic case openings in each replay.](graphs/atsf08_graph_08.png)

## Closing time

Purpose/evidence: The forecast can keep shipping while the investigation waits, fails or asks for more evidence. In this replay, the short stream earns a sensible insufficient-evidence stop. The isolated spike exposes an agent that will not close, and persistent disagreement exposes a premature proposal that deterministic validation catches.
