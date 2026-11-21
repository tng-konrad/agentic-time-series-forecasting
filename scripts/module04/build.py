from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
core = (ROOT / "scripts/module04/core.py").read_text()
publication = (ROOT / "scripts/module04/publication.py").read_text()
nb = nbf.v4.new_notebook()
cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text.strip()))
code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))

md("""# Plan the forecasting experiment before executing it

Episode 4 · Plan-and-execute · Daily retail demand

A planner writes a typed experiment plan, Python executes its dependencies, and a failed diagnostic can trigger one revision. We use every store in a 913,000-row retail table, aggregated to ten daily store series. The complete plan and each step's status remain inspectable.

## Setup

Use Python 3.12 and a Jupyter kernel with `numpy==2.4.6 pandas==3.0.3 scipy==1.15.3 statsmodels==0.14.6 matplotlib==3.11.0 ipykernel nbformat nbclient`. The repository's `.venv` already supplies these packages. Select that kernel, start at the repository root and Run All.

Default **analysis-only replay** uses the bundled data and a named genuine capture, without contacting Ollama or refitting models. It verifies identities and regenerates analysis/figures; it does not reproduce numerical tool execution. A new live run requires Ollama at the configured endpoint and the explicitly installed `qwen3.5:9b-q8_0` model (about10.4GB weights). Set `A04_MODE=live` before starting the kernel; a new run retains every first attempt and pauses scoring until its author audit exists. See [the launch guide](docs/module-4-implementation.md). Never substitute cached responses for a failed live call.
""")
code("""# Standard library
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import textwrap
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
""")
md("""NumPy (`np`) handles arrays; pandas (`pd`) handles calendars and tables; Matplotlib (`plt`) renders publication assets. Statsmodels supplies fixed statistical models and diagnostics. `norm.ppf` returns Gaussian quantiles; `Image` displays the exact exported PNG bytes. Model requests use Python's standard HTTP client, so no agent framework is needed.""")
code('''class CFG:
    episode_number = 4
    asset_prefix = f"atsf{episode_number:02d}_"
    root = Path.cwd()
    data_dir = root / "data/module04"
    evidence_dir = root / "examples/module04"
    graph_dir = root / "graphs"
    timezone = "Europe/Amsterdam"
    execution_started_at = datetime.now(ZoneInfo(timezone)).isoformat()
    mode = os.environ.get("A04_MODE", "replay")
    run_id = os.environ.get("A04_RUN_ID", datetime.now(ZoneInfo(timezone)).strftime("live-%Y%m%dT%H%M%S%f"))
    export_id = datetime.now(ZoneInfo(timezone)).strftime("export-%Y%m%dT%H%M%S%f")
    export_dir = evidence_dir / "exports" / export_id
    endpoint = "http://127.0.0.1:11434"
    model = "qwen3.5:9b-q8_0"
    temperature, seed, num_ctx, num_predict = 0, 42, 16384, 3000
    keep_alive = "20m"
    max_requests, max_tools, request_seconds, workflow_seconds = 2, 16, 90, 900
    max_input_characters, reason_words = 24000, 60
    stores = list(range(1, 11))
    prefix_n, horizon, origins = 1736, 90, [1466, 1556, 1646]
    diagnostic_window, diagnostic_lag, diagnostic_alpha = 364, 28, .05
    annual_period, harmonics, maxiter, alpha = 365.25, 3, 500, .2
    method_order = ["seasonal_naive", "trend", "bsm", "sarimax"]
    figure_width, figure_height, dpi, table_font = 9.5, 6.5, 200, 12
    core_sha256 = "CORE_HASH"
    assets = []

np.random.seed(CFG.seed)
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"figure.figsize": (CFG.figure_width, CFG.figure_height), "font.size": 12})
'''.replace("CORE_HASH", hashlib.sha256(core.encode()).hexdigest()))
md("""`CFG` centralizes the episode identity, settings, paths, budgets and export dimensions. Seeds and temperature reduce variation; a fresh local sample can still differ across hardware and model/server versions. The 900-second limit stops admission before each tool **and each fit**; an already running fit can overrun. Model requests have a90-second timeout clipped to remaining admission time. Two shared requests allow an initial plan and at most one revision, with zero retries. Setup and later fixed comparators have separately measured timing.

## Utils
### Data boundaries and numerical tools

The data gateway returns only each store's first1,736 observations. Pointwise log transformation has no learned parameters. Every statistical model is fitted independently inside each training fold; annual Fourier regressors use absolute days since1January2013, including future phases. Warnings and convergence failures stay in the evidence.

The trend-only model contains a stochastic level and a deterministic slope. We constrain slope noise to zero after a synthetic development pilot exposed explosive long-horizon variance in an unrestricted stochastic slope. This is a declared numerical adaptation of m10, not a repair selected from final outcomes. `bsm` adds a seven-day stochastic seasonal state and three annual sine/cosine pairs. `sarimax` combines `(1,1,1)(1,1,1,7)` dependence with three annual Fourier pairs as regressors. Maximum-likelihood fitting uses L-BFGS with at most500 iterations; nonconverged or nonfinite candidates are ineligible.
""")
first = core.index('INITIAL_TOOLS =')
second = core.index('def run_live(')
third = core.index('def verify_archive(')
code(core[:first])
md("""`candidate_record` evaluates the same three90-day folds and fits the current-origin model using the permitted prefix. Predictions and fitted innovations are different evidence: validation measures forecasting performance, while the monitor examines what the fitted trend model leaves behind. A small Ljung–Box p-value triggers a review; this fixed engineering rule is not a calibrated certificate of model adequacy or a diagnosis of the cause.

### Plans, model requests and contracts

The planner chooses a transformation and, on revision, a seasonal candidate subset. The small executable graph is fixed so each prerequisite can be checked. This deliberately constrained example demonstrates model-family decisions and revision, rather than arbitrary task-graph synthesis. The schema constrains output shape; semantic validation also checks the immutable boundary, exact dependencies, unique candidates and failed-evidence identity. A valid contract does not establish a true explanation.""")
code(core[first:second])
md("""Each local request records its exact input, schema, prompt, provider result, token counts and terminal status. Stop/truncation, malformed output and transport failures consume the request and remain failures. The bounded replan summary includes per-store fold scores, monitor values and artifact hashes; full numerical arrays remain in the archive. There are no JSON repairs or hidden provider changes.

### Execute, monitor and revise

The executor commits every step's status and output hash before continuing. A failed trend check blocks the remaining suffix. The replan must retain successful artifacts and the original failed event; it supersedes blocked steps and supplies four new dependent steps. A second failed diagnostic permits explicitly labeled unresolved delivery, with no second revision. Shared controller status and individual-store statuses answer different questions.""")
code(core[second:third])
md("""The initial no-repair arm is a counterfactual comparator, so it may deliver despite a failed candidate check. It is never counted as a successful agent workflow. Invalid initial planning falls back to seasonal naive; an invalid revision retains the valid initial selection with a labeled fallback. Expiry admits no new candidate fitting. Fixed comparators run afterward and their results are hidden from both planning requests.

### Freeze, audit and score

The capture and forecast hashes are frozen before scoring. The author audit checks public factual propositions against evidence available at each request, before final outcomes are read. Supported, contradicted, unsupported and ambiguous claims are retained; proposals and unassessable responses are counted separately. This is an author audit, distinct from independent plan/editorial reviews.

Replay validates archived analysis identities, including data, protocol, helper code, prompts and semantic inputs. It never silently calls a model or completes missing work. It does **not** refit or numerically reproduce the archived forecasts.""")
code(core[third:])
md("""RMSE squares errors before averaging and emphasizes large misses. MAE averages absolute errors. Seasonal MASE divides MAE by the permitted prefix's mean absolute seven-day change; it uses `y[7:] - y[:-7]`, not a seventh-order difference. An undefined scale keeps valid unscaled metrics. Interval score penalizes wide intervals and adds a penalty when an outcome falls outside them; lower is better. Nominal80% intervals are marginal plug-in intervals, not simultaneous90-day guarantees.

### Plotting and publication

One graph or table is exported per image without an internal title. Saved PNG bytes are displayed here and linked in the post. Sidecars retain typed source values and display strings; the manifest records inference/export identities, source hashes, dimensions and review status.""")
code(publication)
md("""These helpers centralize figure size, table wrapping, numbering and provenance. Previous images are archived before stable filenames are replaced. Visual inspection is a separate final check bound to each PNG hash.

The next call runs the complete frozen workflow in live mode, or loads its genuine archive in analysis-only replay. We then unpack that record in execution order. Final outcomes remain outside planning, fitting and monitor inputs; scoring is a separate call later in the notebook.""")
code('''panel, data_manifest = load_panel()
protocol = load_json(CFG.evidence_dir / "protocol.json")
if protocol["core_sha256"] != CFG.core_sha256 or protocol["contract"] != contract_settings():
    raise ValueError("Helper code changed: create and validate a new protocol before recapture.")
capture, run_dir = obtain_study(panel, protocol)
CFG.run_dir = run_dir
tests = load_json(CFG.evidence_dir / "development/checks.json")
result = None
print("Capture:", capture["run_id"], "Controller:", capture["controller_status"])
''')
md("""The controller status summarizes the shared run, not every store's diagnostic. The raw archive retains all attempts, selected/delivered methods, per-store results and the forecast freeze. Replaying it supplies no new model-quality or inference-speed observation.""")

md("""## Why write the experiment before running it?

Daily demand brings decisions that depend on earlier decisions: define validation before selecting a model, choose a transformation before fitting, inspect a failed model before scheduling a repair, and freeze predictions before scoring. An explicit plan makes those dependencies executable.

[Post1's notebook](a01-forecasting-workflow.ipynb) follows a fixed sequence. [Episode3](a03-diagnostic-loop.ipynb) lets a model choose the next diagnostic after each observation. Here a local planner specifies a complete experiment, Python executes it, and the planner is called again only when a monitor raises a problem. The supplied [source document](docs/a04-Plan-and-Execute%20Forecasting%20Agents.docx) describes this separation; [LangChain's architecture article](https://www.langchain.com/blog/planning-agents) documents the same broad pattern. Its speed/quality motivations are not measured benefits of this experiment.

[m10](m10-state-space.ipynb) supplies the structural-model ladder, transformations and daily weekly/annual seasonality. We adapt its chain-wide aggregation to ten separate store totals. The source's toy candidate code uses test actuals to create predictions; our tools fit real models on allowed history instead. [m09](m09-causal-methods.ipynb) is an optional multivariate extension, not evidence implemented here.""")
md("""## Put the future behind a boundary

The [competition data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) contain913,000 rows: ten stores,50 items and1,826 dates. We sum each store's50 items, producing18,260 store-days and **ten forecasting tasks**. This is not an item-level500-series benchmark, and the competition source does not establish real-retailer authenticity.

All stores use1January2013–2October2017 as permitted history. Validation forecasts6January–5April,6April–4July and5July–2October2017 from expanding prefixes of1,466/1,556/1,646 days. Each predicts90 days. The final90 days,3October–31December2017, are opened only after the forecast freeze and explanation audit. This familiar historical dataset was already explored in m10; it is a retrospective example, not a prospective test. Store1 is the predeclared plot example.""")
code("make_visual(1, panel, capture, result, tests)")
md("""*Graph01. Store1's permitted daily history and three validation blocks.*

Weekly variation and a slower annual envelope are visible. These motivate candidate seasonal structure; the chart cannot choose a winning model. Every store, including less convenient cases, remains in the cohort.""")
md("""## Give the planner an executable contract

We deliberately assign a trend-only first experiment, following m10's simplest structural rung. The planner chooses identity or log scale and writes the complete settings/dependency contract before fitting. Seasonal naive is mandatory; horizons, folds, diagnostic rules and uncertainty definitions are immutable. After a failed check, the model can add BSM, SARIMAX or both. Code owns the forecasts and selection.

This restriction makes a failure-and-revision lesson readable. It is not an independently discovered planner mistake and is not the sole accuracy comparator. A fixed log-scale weekly/annual BSM provides a sensible seasonal reference.""")
code("make_visual(2, panel, capture, result, tests)")
md("""*Table02. Genuine initial plan and its prerequisites.*

The complete JSON also specifies transformation, validation, candidate families, monitor and uncertainty. The graph makes premature delivery impossible under the normal path: intervals and delivery wait for an explicit monitor disposition. The public reason is audited separately from syntax.""")
md("""## Execute until a check fails

Each candidate is ranked by mean RMSE across the three90-day validation forecasts. Exact ties follow seasonal naive, trend, BSM, SARIMAX. A failed/nonconverged candidate is retained and ineligible. The monitor then tests the **trend candidate**, even if seasonal naive already won validation; failure does not mean the selected baseline forecast failed.

For each store, the monitor checks the last364 standardized one-step innovations with Ljung–Box through lag28 at the fixed.05 trigger and zero degrees-of-freedom adjustment. These fitted residuals are not90-day forecast errors. The p-value is an engineering trigger; model fitting, dependence and repeated checking limit inferential interpretation.""")
code("make_visual(3, panel, capture, result, tests)")
md("""*Graph03. Store1's trend-model innovation autocorrelations.*

Read recurring lag patterns alongside the recorded monitor value. They can motivate a seasonal repair, but they do not establish a causal explanation or guarantee that a proposed replacement will pass.""")
code("make_visual(4, panel, capture, result, tests)")
md("""*Table04. Initial execution status before revision.*

Completed work remains available, the failed check remains failed, and pending uncertainty/delivery are blocked. These statuses explain why a new plan is needed without silently rerunning the completed prefix.""")
md("""## Revise the remaining work

The replanner sees every store's fold scores and monitor values, plus hashes of the initial plan and completed artifacts. It receives no final actuals and no fixed-arm results. It must preserve the transformation and experimental boundary while choosing seasonal repair candidates.

The accepted suffix evaluates those candidates, repeats the monitor on each validation-selected structural method, records a disposition, and delivers. Baseline selection has no fitted-innovation monitor. A failed repeated check produces `diagnostic_unresolved`; delivery can still be valid, while the workflow remains unresolved. One shared unresolved status does not mean that every store failed.""")
code("make_visual(5, panel, capture, result, tests)")
md("""*Table05. Genuine revised suffix and terminal status.*

The original failed event is retained; the two blocked initial steps are superseded. Baseline and trend artifacts are reused. A plan revision records what changed and why; the residual check and final errors separately assess what that change achieved.""")
md("""## Freeze forecasts, then open the outcome

For a Gaussian forecast on log scale, `exp(mu + variance/2)` gives the plug-in conditional mean used for RMSE; `exp(mu)` is its median. Exponentiating Gaussian quantiles gives asymmetric original-scale intervals. Identity-scale models use Gaussian means/intervals directly. Both ignore parameter uncertainty. Seasonal naive uses a symmetric width from the80th percentile of its absolute validation errors, without a nominal coverage guarantee.

The initial no-repair arm reuses the initial validation selection. The agent arm uses the expanded eligible menu. The fixed arm independently evaluates log BSM against seasonal naive. The fourth arm repeats the last seven days. All use identical information boundaries and scoring. The next call verifies the author audit and frozen-output hashes before reading final actuals.""")
code("result = score_study(panel, capture, run_dir)\nprint(pd.DataFrame(result['summaries']).drop(columns='eligible_cases').to_string(index=False))")
md("""The printed rows retain coverage out of all ten stores. Pooled RMSE/MAE use equally long store-day forecasts; MASE is averaged equally across stores. Paired conclusions require the same delivered stores in both arms. Undefined metrics or missing predictions retain explicit reasons.""")
code("make_visual(6, panel, capture, result, tests)")
md("""*Graph06. Frozen store1 forecasts, nominal80% agent interval and final daily outcomes.*

The vertical line marks the first final-evaluation day. Coincident forecasts are retained with different markers/line styles. A single plotted store does not summarize the cohort; interval behavior over90 dependent days is descriptive.""")
code("make_visual(7, panel, capture, result, tests)")
md("""*Table07. Forecast error and interval diagnostics across all ten stores.*

Lower RMSE, MASE and interval score are better. Coverage needs width for context: a very wide interval can cover everything. The full sidecar/score archive contains all40 case/arm rows, including MAE, selected/delivered method and failure status. These dependent competition series do not supply a general reliability or superiority estimate.""")
code("make_visual(8, panel, capture, result, tests)")
md("""*Table08. Actual original-run work and durations.*

Agent time includes the initial plan, repair and finalization, plus local inference; loading within a request consumes its timeout. Fixed BSM runs afterward with a separate numerical cache. The no-repair comparator shares the initial prefix, so it has no invented independent runtime. Synthetic pilot work and this export's replay speed are excluded. A saved fit count is not an end-to-end speed improvement.""")
md("""## Check the controller when things go wrong

Fixtures check invalid plans, stale evidence, temporal boundaries, phase alignment, failure/fallback and archive integrity. They use deterministic clients or synthetic numerical inputs, never a replacement for the genuine run. Their pass counts do not estimate the LLM's reliability.""")
code("make_visual(9, panel, capture, result, tests)")
md("""*Table09. Development checks outside genuine model/forecast denominators.*

Withheld-value mutation is separate from file-checksum verification: changing final values in memory leaves permitted inputs and forecasts unchanged. Changing a permitted input or contract changes identity or rejects replay. Annual Fourier phases continue across the origin. The revision ledger preserves the failed event and marks old blocked steps as superseded.

An m09 extension could plan a VAR/VECM experiment on multiple available series. It would need an explicit statement about which future regressors are known, lag order and chronological alignment; this notebook implements univariate store-total forecasts only.""")
md("""## Closing time

An explicit plan makes the forecasting experiment reviewable before the expensive work starts. Here the useful boundary is concrete: model-authored transformation and repair choices become typed instructions, Python executes them, and a failed check interrupts the pending suffix without erasing prior evidence.

The final interpretation must keep workflow completion, diagnostic resolution, forecast accuracy and uncertainty calibration separate. This is one shared first-attempt plan/replan on ten dependent historical competition series. The prescribed trend-only challenge is pedagogical; the fixed seasonal model remains a meaningful reference. Use this pattern when several modelling decisions depend on earlier checks and you need a durable record of what changed.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]""")
code("asset_manifest = finish_exports(capture, result)\nprint('Export:', CFG.export_id, 'Visuals:', len(asset_manifest['assets']), 'Replay scope: analysis-only')")
md("""The final manifest links this export to genuine inference and full-precision sidecars. Each visual receives a separate checksum-bound inspection. The [launch and validation guide](docs/module-4-implementation.md) records what ran and the limits of replay.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
nbf.write(nb, ROOT / "a04-plan-and-execute.ipynb")
print("Built", len(cells), "cells")
