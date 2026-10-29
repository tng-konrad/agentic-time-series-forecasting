"""Author Module 2's teaching cells; execute_module02 runs the notebook offline."""
from pathlib import Path
import nbformat as nbf
from agentic_ts.module02_workflow import HEADINGS

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def cell(code, prose, ident):
    cells.append(nbf.v4.new_code_cell(code.strip(), id=ident)); md(prose)
def section(i): md("## " + HEADINGS[i-1])

md('''# Module 2 · One forecasting request, several possible problems

**Pattern: routing.** Which workflow can handle the information we actually have?
We compare a minimal numerical baseline, explicit routing rules, and a one-call LLM
router. The target is daily recorded item sales, not unconstrained demand.

The [companion post](a02-post.md) follows the same substantive headings and reuses
the PNGs exported here. Reusable methods live in
[routing_forecasting.py](agentic_ts/routing_forecasting.py); request contracts and
dispatch live in [module02_workflow.py](agentic_ts/module02_workflow.py).

This notebook recomputes forecasts in an offline clean kernel. It verifies and replays
genuine local-model responses; it does not contact Ollama. The original primary capture
is preserved, including failed contracts and unnecessary clarifications. A separately
labeled development call demonstrates a valid LLM-controlled dispatch and is excluded
from primary comparison metrics. Short and zero-history requests are constructed views
of real historical series, not observations of genuine product launches.''')
md('# Setup')
cell('''# Standard library
import json
from pathlib import Path
from unittest.mock import patch

# Data and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reusable project tools
from agentic_ts import module02_workflow as workflow
from agentic_ts import routing_forecasting as numerical
from agentic_ts import module02_publication as publication
from agentic_ts.module01_assets import PublicationAssets
''','''`Path` builds local paths and `json` reads the saved evidence. NumPy (`np`) supplies
arrays and pandas (`pd`) supplies dated observations. Matplotlib draws figures; it does
not calculate forecast scores. `patch` will make accidental live inference fail loudly.

The project imports divide responsibility: numerical routines fit and score models,
the workflow validates decisions and controls dispatch, and publication helpers only
present existing values. The table exporter is shared with Module 1 using explicit
Module 2 filenames. Importing these modules starts no model calls. This separation
keeps the teaching notebook connected to the implementation rather than duplicating it.''','imports')
cell('''class CFG:
    root = Path.cwd()
    data_folder = root / "data" / "module02"
    example_folder = root / "examples" / "module02"
    graph_folder = root / "graphs"
    run_mode = "replay"
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
    colors = {"history": "#4C72B0", "actual": "#222222", "A": "#8172B3", "B": "#55A868", "C": "#DD8452"}

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (CFG.img_dim1, CFG.img_dim2)
np.random.seed(CFG.SEED)
assert (CFG.root / "pyproject.toml").is_file(), "Run from the repository root"
assert CFG.run_mode == "replay"
''','''`CFG` gathers paths, dimensions, colors and reproducibility settings in one place.
The 20-by-10 default comes from the series; a shared scale of 0.65 produces a 13-inch
publication canvas. Export resolution is 180 DPI. Table heights depend on their rows,
not a fixed height that would crush long explanations.

The NumPy seed fixes any NumPy randomness. It does not promise identical language-model
inference across machines. Exact saved-input matching provides reproducible language
replay instead. Float formatting happens only in presentation; no metric is rounded
before model selection. The root assertion prevents paths silently referring to the
wrong checkout.''','config')
md('# Utils')
cell('''def read_record(name):
    return json.loads((CFG.example_folder / name).read_text())

protocol = read_record("protocol.json")
capture = read_record("recorded_run.json")
workflow.verify_capture(capture)
assets = PublicationAssets(CFG, workflow.digest(protocol), prefix="atsf02",
                           post="a02-post.md", sidecar_dir="examples/module02")
print("Replay model:", capture["model_identity"]["name"])
print("Versions:", numerical.package_versions())
''','''`read_record` reads a named evidence file. `verify_capture` checks its integrity and
model configuration before we trust any response. Later, each decision is checked
against its exact request view, prompt and schema too. A file existing on disk is not
enough to make it a valid replay.

`PublicationAssets` numbers graphs and tables separately and saves typed table
sidecars. Its source hash initially identifies the frozen protocol; later we point it
at the executed result used by each image. The notebook displays exported PNGs, so the
post uses the same image bytes.''','load-records')
cell('''print("Numerical routes:", numerical.FAMILIES)
print("Short-history candidates:", numerical.SHORT_CATALOG)
print("Forecast horizon:", workflow.NUMERICAL.horizon)
print("Replay cannot call a model: every replay check supplies recorded=...")
''','''These dictionaries are the available candidate menus. A route selects a workflow;
chronological numerical validation chooses a model inside that workflow. The LLM
cannot rewrite either menu or choose new smoothing parameters.

We reuse the established backtesting and failure-handling routines. The short-history
extension supplies a different validation window and a trailing mean; the new-product
extension accepts donor histories only. Neither can obtain the hidden target history
through a global panel argument.''','tool-contracts')
section(1)
md('''Routing classifies an input and sends it to a predefined specialist path. Module 1
used a fixed sequence; here one bounded decision chooses which numerical workflow runs.
It remains a workflow, with explicit limits, rather than an open-ended diagnostic agent.

The modular architecture has antecedents such as
[MRKL Systems (2022)](https://arxiv.org/abs/2205.00445), which combined language models
with specialist knowledge and reasoning modules. [Anthropic's December 2024 guide](https://www.anthropic.com/research/building-effective-agents)
describes routing as a practical workflow pattern; [LangGraph's official guide](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
shows structured routing and conditional branches. These sources demonstrate visibility
in research and tooling, not a measured adoption rate.

People care because one request can conceal several different tasks. Specialization
can make responsibilities clearer and avoid inappropriate work. It also introduces a
classifier, another contract and another failure point. That complexity earns its place
only if the resulting behavior is useful enough to justify its costs. Reliable structured
metadata may make ordinary rules sufficient; this experiment allows that conclusion.''')
cell('''assets.save(publication.architecture(CFG), "graph", workflow.HEADINGS[0], "routing-diagram",
            "A shared gate, a bounded decision, and four numerical workflows.")
''','''Graph 1 shows the authority boundary. The shared gate checks the request and visible
data. B applies rules; C proposes a route. Python validates C's proposal before dispatch.
An invalid response gets one rule fallback. A valid clarification produces no forecast.

A is a minimal numerical reference with the same basic capability checks. D is an
isolated all-candidates diagnostic, evaluated after main decisions are fixed. Its
results cannot influence the router. The arrows show control flow, not an expectation
that every route improves accuracy.''','routing-diagram')
section(2)
cell('''panel, source_manifest, cases = workflow.load_inputs()
views = {c["case_id"]: workflow.request_view(c, panel, source_manifest) for c in cases}
case_by_id = {c["case_id"]: c for c in cases}
print("Source shape:", panel.shape)
print("Origin:", workflow.ORIGIN, "| Horizon: 28 daily observations")
''','''`load_inputs` checks the M5 subset against its hash and validates the derived case
manifest. The source has 12 items and 477 daily observations per item. It is a small
teaching panel from one store, not a representative sample of all retail.

`request_view` is the information boundary. It slices target history at the forecast
origin, applies the 60-day or zero-history restriction, and supplies only authorized
pre-origin donor data. Downstream tools receive this restricted object, not the full
panel. Dates after 2016-03-27 remain unavailable until final scoring. The data provenance
is recorded in the source manifest and linked to the [M5 organizers](https://github.com/Mcompetitions/M5-methods).''','load-data')
cell('''visible = workflow.as_series(views["M02-F01"]["history"])
print(visible.head(3).to_string())
print("Short view rows:", len(views["M02-S01"]["history"]["values"]))
print("New-product view rows:", len(views["M02-N01"]["history"]["values"]))
''','''The text preview checks parsing without creating an unexported table. The short view
contains 60 real observations. The new-product view contains no target observations.
Neither condition is an all-zero history: zeros are observed sales; missing history is
an absence of observations.

The short and new views reuse two established products' future outcomes. They are
controlled information restrictions, not independent products or real product launches.
Their results must stay separate from the 12 full-history requests.''','preview')
cell('''assets.save(publication.histories(CFG, views), "graph", workflow.HEADINGS[1], "visible-histories",
            "Only observations permitted at the forecast origin are shown.")
''','''Graph 2 plots the raw permitted observations before model fitting. The first two
panels show an established series and a sparse series. The third deliberately removes
older observations. The final panel shows donor histories and explicitly states that
there is no target history.

No final outcome appears here. The donor choice is a fixed same-department/store
teaching assumption, with no target-based scaling. Matching department and store does
not establish a genuinely comparable launch: that limitation will matter when we
interpret the prior's forecast.''','visible-histories')
cell('''assets.table(publication.table_cases(), workflow.HEADINGS[1], "case-table",
             "Case counts and validation structure; final dates are 2016-03-28–2016-04-24.")
''','''Table 1 separates the sample units. Full-history requests use three rolling 28-day
validation blocks, each preceded by 365 training days. Their training ends are
2016-01-03, 2016-01-31 and 2016-02-28; the last validation block ends at the issue date.
The short view uses its first 32 days to predict its last 28 days, then refits on 60.

All main requests predict 2016-03-28 through 2016-04-24. The extra ambiguity/gate cases
test request handling and do not increase the accuracy sample. Selection strata were
chosen using an earlier training window; current routing evidence is recomputed at the
issue date, not copied from those labels.''','case-table')
section(3)
cell('''assets.table(publication.table_workflows(), workflow.HEADINGS[2], "workflow-table",
             "Workflow capabilities are declared before any final score is read.")
''','''Table 2 defines the menu. Regular and intermittent are candidate workflows, not
diagnoses. Both retain seasonal naive, so routing to TSB's family does not force TSB to
win. Full-history selection minimizes mean RMSE over all three validation blocks.

TSB updates an occurrence probability and a positive-sales size estimate; here both
smoothing parameters are fixed at 0.10. Holt–Winters uses additive seasonality and a
damped additive trend. The [pinned StatsForecast implementation](https://nixtlaverse.nixtla.io/statsforecast/docs/models/tsb.html)
and existing statsmodels wrapper do the arithmetic. Failed validation folds make a
candidate ineligible. Nonconverged fits are failures even when they return numbers.

A uses seasonal naive for 56 or more observed days and the same donor prior for an
eligible zero-history case. That 56-day floor is the module's capability contract,
not the mathematical minimum for seasonal naive.''','workflow-table')
cell('''baseline_example = numerical.fit_forecast(visible.tail(365), "seasonal_naive")
print("Last observed week:", visible.tail(7).tolist())
print("First forecast week:", baseline_example["forecast"][:7])
''','''Seasonal naive repeats the last observed week across the 28-day horizon. The two
printed lists should agree. This is a transparent baseline: no optimizer, LLM or
future observation is needed. The small teaching call is outside the timed benchmark.

For optimized candidates, code checks convergence, finite output and exact dates.
Negative forecasts are clipped to zero with a recorded count. A final-fit failure
falls back to seasonal naive; failure of that baseline leaves numerical failure.
Selection and delivery therefore have separate fields.''','baseline-example')
cell('''short = workflow.as_series(views["M02-S01"]["history"])
print("Short training:", short.index[0].date(), "to", short.index[-29].date(), "| n =", len(short)-28)
print("Short validation:", short.index[-28].date(), "to", short.index[-1].date())
''','''The short-history split is chronological. Thirty-two training observations are
enough for the declared seasonal-naive and trailing-28-day-mean candidates, but not
for the full workflow's 365-day training plus three validation blocks. After this
single comparison, the selected method is refitted using all 60 available days.

One validation block is weaker evidence than three. We do not claim that choosing
the short workflow solves limited-data forecasting; it simply avoids pretending that
unavailable training history exists.''','short-split')
cell('''new_evidence = workflow.profile(views["M02-N01"])
print(json.dumps(new_evidence["donor.prior"], indent=2))
''','''The prior is the median of three donor means, each computed from the 28 days ending
at the origin. It is a constant daily forecast with no target-history scaling or target
validation. Both the minimal baseline and routed approaches receive this same ability.

That is a deliberate control: supplying donors only to C would confuse access to
information with a routing advantage. A donor value change always changes the request
hash, but need not change the median. The tests distinguish those two properties.''','donor-example')
section(4)
md('''B and C share an ordered policy: resolve inconsistent metadata; handle zero history;
handle short history; then classify full histories using the last 365 days. The last
step chooses intermittent when `2 * zero_count >= observed_count`. This 50% threshold
is an explicit teaching policy, not an optimality claim or the ADI/CV² classification.

The LLM sees the same measured evidence and policy as B. It returns a route, evidence
IDs, a short reason and a question field. A forecasting route requires that question
to be JSON `null`. Valid disagreement between eligible full-history families really
dispatches the proposed family; code does not secretly force agreement with B.''')
cell('''primary_example = capture["requests"]["M02-F01"]["C"]
print(primary_example["response"]["text"])
print("Validation:", primary_example["status"], "|", primary_example.get("failure"))
''','''The first primary response proposes the regular route using the correct zero count.
It also asks for a launch date and confirmation of the supplied lifecycle. Those two
outputs do not satisfy the contract: a forecasting decision must have a null question.
The validator rejects it and records a rule fallback. The proposed route remains
visible so that a rejected answer is not rewritten into a successful decision.

This illustrates why a plausible route, valid evidence IDs, a supported reason, and
an acceptable response contract are different checks. The full record preserves
messages, schema, raw text, response hash, usage and timing.''','inspect-response')
cell('''route_rows = []
with patch.object(workflow.OllamaClient, "complete", side_effect=AssertionError("Replay made a live call")):
    for case in cases:
        arms = {}
        for arm in ("B", "C"):
            archived = capture["requests"][case["case_id"]]["C"]
            arms[arm] = {"decision": workflow.choose(arm, views[case["case_id"]], recorded=archived)}
        route_rows.append({"case": case, "arms": arms})
route_result = {"rows": route_rows}
print("Replayed decisions:", len(route_rows), "| Live calls: 0")
''','''Each request is reconstructed before replay. Its permitted-history hash, prompt,
schema and configuration must match the captured input exactly. A replay-integrity
error stops the run. It is not treated as an ordinary model error and hidden behind
a rule fallback.

`patch.object` replaces the HTTP client's completion method with a deliberate error.
If this cell accidentally tried live inference, execution would fail. Gate-stopped
requests require no recorded response. Valid clarifications retain no executed route;
invalid captured responses reproduce their explicit fallback behavior.''','replay-routes')
cell('''assets.source_hash = workflow.digest(workflow.semantic(route_result))
assets.table(publication.table_routes(route_result), workflow.HEADINGS[3], "route-table",
             "Primary proposals and execution sources; short and new are abbreviated route names.")
''','''Table 3 includes every main request. `Days` is the target history length; `Zeros /
window` uses at most the most recent 365 observed days. Zero-history ratios are N/A.
The two rightmost columns deliberately distinguish the LLM's proposal from what ran.

The primary run has 11 rejected forecasting proposals and five accepted clarifications.
None of those clarifications was necessary on the 16 supported main cases. Every
delivered primary C forecast therefore comes from rules. Policy agreement among
accepted LLM forecasting decisions has a zero denominator and must be reported N/A,
not 0% or 100%.''','route-table')
cell('''development = read_record("development_run.json")
assert workflow.digest({k: v for k, v in development.items() if k != "bundle_hash"}) == development["bundle_hash"]
with patch.object(workflow.OllamaClient, "complete", side_effect=AssertionError("Unexpected live call")):
    dev_decision = workflow.choose("C", views["M02-F01"], recorded=development["decision"], development=True)
assert dev_decision["source"] == "llm" and dev_decision["executed_route"] is not None
dev_numerical = workflow.execute(dev_decision["executed_route"], views["M02-F01"])
print(development["decision"]["response"]["text"])
print("Genuine development dispatch:", dev_decision["executed_route"], "→", dev_numerical["delivered"])
''','''This is the separately labeled development demonstration allowed by the plan. One
new call clarified that established products need no launch date and used an equivalent
nullable-field schema encoding. The same configured model returned a null question
and an accepted regular route. That real decision now controls a real numerical call.

It does not replace any primary response, enter Table 3, or improve C's benchmark
score. Two presentation changes were made together, so this single call cannot identify
which change helped or establish a reliable repair. Its separate archive exposes the
exact changed messages, schema, timing and response.''','development-dispatch')
section(5)
cell('''with patch.object(workflow.OllamaClient, "complete", side_effect=AssertionError("Unexpected live call")):
    frozen = workflow.freeze_run(panel, source_manifest, cases, capture)
workflow.write_json(CFG.example_folder / "reader_frozen_forecasts.json", frozen.data)
author_frozen = read_record("frozen_forecasts.json")
assert workflow.digest(workflow.semantic(frozen.data)) == workflow.digest(workflow.semantic(author_frozen))
print("Predictions frozen before scoring:", workflow.digest(workflow.semantic(frozen.data)))
''','''This cell recomputes the numerical workflows; it does not copy stored predictions.
Each arm runs independently without a numerical cache. D searches all three full-history
candidates on the same validation blocks, after A/B/C decisions have been made. It is
an isolated diagnostic, not an oracle selected using the final outcomes.

The frozen record contains dated predictions, selected and delivered methods, fit
statuses and dispatch sources. We save it before calling the scorer. The semantic
hash excludes measured durations and session timestamps, which naturally change on
re-execution, but preserves the predictions and decisions. It matches the author run.''','freeze-forecasts')
cell('''result = workflow.score_run(frozen, panel)
assert result["scored_after_freeze"]
workflow.write_json(CFG.example_folder / "reader_results.json", result)
stats = workflow.summary(result)
print(json.dumps(stats["full_history"], indent=2))
''','''Only now does the scorer open the 28 final observations. MAE averages absolute
errors; RMSE gives greater weight to large errors. For full histories, MASE divides
MAE by the mean absolute lag-7 difference in the final 365-day training window. Its
scale is computed without the final observations.

The headline comparison first finds requests delivered by every arm, then keeps the
subset with a positive finite MASE scale. Here both counts are nine; none is excluded
for an undefined scale. MASE below one compares against historical scaling error,
not necessarily the actual seasonal-naive forecast on the final horizon. We compare
A's measured final error directly. A missing forecast never receives zero error.''','score-outcomes')
cell('''assets.source_hash = workflow.digest(workflow.semantic(result))
assets.save(publication.forecasts(CFG, result, views), "graph", workflow.HEADINGS[4], "final-forecasts",
            "Predetermined representative cases, with identical curves explicitly grouped.")
''','''Graph 3 uses four cases chosen before evaluation: full dense, full sparse, short
dense, and new dense. It displays the final actuals beside the frozen forecasts.
When A, B and C have identical predictions, their labels share one curve rather than
overplotting it. C's rule-derived forecasts are labeled as such.

The synthetic new-product forecast cannot use a hidden target history; its line comes
from donor means. The plot demonstrates what that assumption produces for an existing
item treated as unseen. It does not validate forecasting actual launches or imply
calibrated uncertainty; no prediction bands have been estimated.''','final-forecasts')
cell('''assets.table(publication.table_accuracy(result), workflow.HEADINGS[4], "accuracy-table",
             "Coverage and scale-defined common-subset scores; constructed views stay separate.")
''','''On the nine full-history requests delivered by all arms, macro MASE is 1.04 for A,
0.89 for B and C, and 0.84 for D. A, B and D deliver all 12 full histories; C delivers
nine. Those C values are the same rule-fallback forecasts as B, not a gain attributable
to the LLM. The common subset was determined by observed delivery, so its result does
not describe the three omitted C abstentions.

The lower block reports the four constructed views individually in item units. Their
shared targets and metadata assumptions rule out pooling them into the 12-item macro
score. Full-precision per-request MAE/RMSE and paired differences are retained in the
result records. No threshold or model was retuned to improve the final comparison.''','accuracy-table')
section(6)
cell('''fixtures = workflow.run_fixtures(panel, source_manifest, cases)
for fixture in fixtures:
    print(fixture["case_id"], fixture["provenance"], "|", fixture["decision"]["failure"])
''','''X05 returns malformed JSON on a valid full-history request. X06 supplies a
schema-valid regular route for a 60-day history, which cannot meet that workflow's
449-day requirement. Both fixture clients are deterministic and make zero real model
calls. They demonstrate separate contract and capability failures, followed by one
rule fallback.

These injected failures must not be counted as observed LLM mistakes. The primary
capture already contains observed mistakes; the fixtures test recovery paths in a
controlled way without relying on a model to fail on demand.''','failure-fixtures')
cell('''assets.table(publication.table_failures(result, fixtures), workflow.HEADINGS[5], "failure-table",
             "Two genuine ambiguity responses, two shared-gate cases and two injected response failures.")
''','''Table 5 separates three sources of behavior. X01 lacks lifecycle/launch information;
X02 lacks an authorized donor set. Both genuine responses ask about genuinely missing
information, although X02 also reconfirms supplied facts and X01's reason falsely says
no end date was supplied. A useful question does not make its explanation fully correct.

X03 omits the horizon. X04 removes 2016-03-17 from a declared daily history. The shared
gate stops both without inference; that is ordinary validation, not agent judgment.
Missing rows are not zero-filled. The last two rows are injected response fixtures.
They are outside the forecasting accuracy and primary model-cost cohorts.''','failure-table')
section(7)
cell('''audit = read_record("routing_audit.json")
assert audit["capture_hash"] == capture["bundle_hash"]
print(audit["method"])
print("Audited genuine responses:", len(audit["reviews"]))
''','''The audit reviews every complete reason and question against its actual evidence
and the declared policy. It is a manual semantic judgment, bound to response hashes;
the code checks the binding but cannot prove that prose is supported merely by finding
valid evidence IDs.

Among the 16 main reasons, three are supported, eleven contradicted, one unsupported
and one ambiguous. Several responses correctly compute the zero threshold, then
incorrectly apply new-product requirements to established items. The
[full review](examples/module02/routing_audit.md) preserves the judgment and explanation
for each response. Reason support is distinct from schema validity and forecast delivery.''','audit-reasons')
cell('''work = publication.work_summary(result, capture, audit)
workflow.write_json(CFG.example_folder / "work.json", work)
routing_metrics = publication.routing_metrics(result, capture, audit)
workflow.write_json(CFG.example_folder / "routing_metrics.json", routing_metrics)
print("Primary routing metrics:", json.dumps(routing_metrics["cohorts"]["main"], indent=2))
for row in work["dispatch"]:
    print(row["cohort"], row["arm"], "n=", row["n"], "calls=", row["calls"],
          "archived seconds=", round(row["seconds"], 2), "median seconds=", round(row["median_seconds"], 2))
''','''The primary work cohort is exactly the same 16 requests for A/B/C. The two ambiguity
attempts are separate; gate cases and response fixtures do not inflate primary model
cost. Counts include unsuccessful attempts. Missing usage would remain N/A, rather
than being treated as zero.

The routing metrics distinguish the supplied JSON schema's shape from the additional
Python decision contract. All 16 main responses match the shape; only five pass the
contract, all clarifications. Accepted forecasting policy agreement has denominator
zero and is N/A. The same record counts eligibility, relevant/unnecessary questions,
input stops and numerical fallbacks without confusing these outcomes.

Archived dispatch time comes from the author capture. Current numerical time comes
from this fresh notebook run, whose session ID is stored separately. We do not add
those components and call the sum measured end-to-end latency. The full record includes
cohort sizes and medians. These are local observations with loading effects, not a
hardware-independent speed benchmark or a dollar-price estimate.''','work-record')
cell('''assets.table(publication.table_work(work), workflow.HEADINGS[6], "work-table",
             "Primary and ambiguity cohorts, archived dispatch, and current numerical work are separate.")
''','''Table 6 starts with audit and delivery counts, then separates archived language
dispatch from recomputed numerical work. Zero accepted primary LLM forecasting decisions
is compatible with eleven delivered C forecasts because those came from fallback rules.
The numerical fit count includes all attempted folds/refits, including failed fits;
donor arithmetic is timed but is not called a fitted model.

The first primary call included model loading, recorded in its metadata. The separate
development call has its own time and is excluded from these primary totals. Very
small measured A/B durations may display as 0.00 because all table floats use exactly
two decimals; the stored measurements are not hardcoded zeros. Captions and notes stay
outside every table image.''','work-table')
section(8)
md('''The primary router added work and unnecessary clarification in this experiment.
Its delivered forecasts came from the deterministic safety path. The separate
development call shows that a real accepted proposal can control dispatch, but does
not turn the primary comparison into a success or establish a general repair.

Routing itself can still be useful: it makes capability requirements and the response
to missing information explicit. When structured evidence makes the policy simple,
rules are the stronger default in this example. Richer natural-language requests may
motivate an LLM router, but this suite does not evaluate that advantage.

Limitations remain substantial: one store, one final horizon, an illustrative threshold,
fixed smoothing parameters, artificial short/new views, assumed donors, repeated target
outcomes, and one ordered local-model capture. Observed sales do not reveal latent
demand. The next module can let a diagnostic agent choose its next tool, but added
autonomy will still need the same evidence boundaries and honest baseline comparisons.''')
cell('''verification = {
    "capture_hash": capture["bundle_hash"],
    "frozen_semantic_hash": workflow.digest(workflow.semantic(frozen.data)),
    "result_semantic_hash": workflow.digest(workflow.semantic(result)),
    "live_calls_in_replay": 0,
    "development_genuine_dispatch": dev_decision["source"] == "llm" and dev_decision["executed_route"] is not None,
    "asset_count": len(assets.assets),
    "graph_count": sum(a["kind"] == "graph" for a in assets.assets),
    "table_count": sum(a["kind"] == "table" for a in assets.assets),
}
assert (verification["graph_count"], verification["table_count"]) == (3, 6)
workflow.write_json(CFG.example_folder / "replay_verification.json", verification)
print(json.dumps(verification, indent=2))
''','''The final record connects the executed predictions, captured responses and nine
exported assets. The publication verifier additionally checks the notebook's clean
execution counts, matching post headings, image bytes, relative paths, typed table
formats and the article's evidence references. Semantic review and visual inspection
remain necessary alongside automated checks.

To rerun from a clean kernel, use `.venv/bin/python -m scripts.execute_module02` from
the repository root after `uv sync --locked`. Model capture is a separate author action
and is never an incidental side effect of opening this notebook.''','completion')

notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (project)", "language": "python", "name": "python3"},
                                                         "language_info": {"name": "python", "version": "3.12"}})
nbf.write(notebook, ROOT / "a02-routing.ipynb")
print("Wrote", sum(c.cell_type == "code" for c in cells), "code cells")
