This repo is about agentic workflows for time series problems:
1. A01: From a forecasting notebook to an agentic workflow


   Pattern:  Sequential workflow / prompt chaining.  
   Build a fixed sequence: inspect data → propose candidates → backtest → forecast → explain. Introduce reusable numerical tools and structured outputs between stages. Compare a conventional Python pipeline with an LLM-assisted workflow, measuring forecast consistency, report correctness, and execution overhead.  

   [Executed notebook](a01-forecasting-workflow.ipynb) · [Post](a01-post.md) · [Evidence audit](examples/module01/claim_audit.md) · [Implementation record](docs/module-1-implementation.md) · [SEO and LinkedIn copy](docs/module-1-promo.md)

2. A02: One forecasting request, several possible problems
   
   
   Pattern: Routing.  
   Introduce established demand, intermittent demand, short histories, and new products. Route each case to an appropriate candidate workflow using measured properties and available metadata. Compare agent routing with explicit rules, including cases where information is missing or ambiguous.  

   [Executed notebook](a02-routing.ipynb) · [Post](a02-post.md) · [Routing audit](examples/module02/routing_audit.md) · [Implementation record](docs/module-2-implementation.md) · [SEO and LinkedIn copy](docs/module-2-promo.md)

   Companion evaluation: [Does routing travel?](a02-routing-panel.ipynb) · [Panel-study post](a02-post-new.md) · [Panel implementation](docs/module-2-panel-implementation.md). It tests the fixed numerical routing policy on 100 products, three stores and six origins, with a development-selected fixed comparator and no new LLM calls.


3. A03: Let the agent choose its next diagnostic

   Pattern: Single-agent tool loop / ReAct-style agent.  
   Choose a diagnostic or benchmark, observe its actual result, and choose again within a five-decision budget. Compare eight genuine local-model investigations with a fixed checklist, retaining rejected actions, incomplete searches, and forecast consequences.

   [Executed notebook](a03-diagnostic-loop.ipynb) · [Post](a03-post.md) · [Decision audit](examples/a03/diagnostic_audit.md) · [Implementation record](docs/module-03-implementation.md)

If [Time Series with Konrad](https://github.com/tng-konrad/time-series-with-Konrad) was a TV show, this would be a spinoff focused on a new character that turned out to be quite popular.

## Run Module 1

Module 1 was rebuilt from scratch with a fresh FRED snapshot and a new local-model capture. From the repository root, use Python 3.12 and the locked environment:

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_module01
.venv/bin/python -m unittest discover -s tests -p 'test_module01.py' -v
.venv/bin/python -m scripts.verify_module01_publication
```

The runner executes all 26 code cells in a clean kernel, recomputes the forecasts, verifies the saved genuine responses against their inputs, and exports three graphs and six tables to `graphs/`. After environment installation, this run needs neither a network connection nor an Ollama server. The post embeds the exact exported PNGs. The [implementation record](docs/module-1-implementation.md) documents data provenance, model settings, limitations and validation.

The separate authoring scripts `scripts.prepare_module01_data` and `scripts.capture_module01` refuse to overwrite the original snapshot or capture. They are not needed for reader replay. The primary capture used the installed local `qwen3-coder-next:q4_K_M`; one completed chain is a mechanism demonstration, not a benchmark of general model quality.

## Run Module 2

Module 2 reuses the bundled M5 sales panel and adds short-history and synthetic new-product requests. From the repository root, use the same locked environment:

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_module02
.venv/bin/python -m unittest tests.test_module02_routing tests.test_a02_routing tests.test_module01 -v
.venv/bin/python -m scripts.verify_module02_publication
```

The runner executes all 27 code cells in a fresh kernel, recomputes the numerical forecasts, verifies exact replay of the genuine local-model responses, and exports three graphs and six tables to `graphs/`. After environment installation, it needs neither network access nor Ollama. The post embeds those exported PNGs.

The primary capture preserves eleven rule fallbacks and five unnecessary clarifications across sixteen supported requests. A separately archived development call demonstrates accepted LLM-controlled dispatch and is excluded from the primary benchmark. The [implementation record](docs/module-2-implementation.md) explains the numerical comparison, provenance, limitations and validation. Author capture scripts refuse to overwrite their archives and are unnecessary for reader replay.

## Run the Module 2 panel study

The companion notebook recomputes 600 development and 1,800 main forecasts using the same numerical methods. Exact training-window fits are shared across arms/origins; a separate 36-case uncached benchmark measures B/D runtime. The extract and author records are bundled, so the raw M5 archive and companion repository are unnecessary for reader execution.

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_module02_panel
.venv/bin/python -m unittest tests.test_module02_panel tests.test_module02_routing tests.test_a02_routing tests.test_module01 -v
.venv/bin/python -m scripts.verify_module02_panel_publication
```

The clean-kernel run executes 31 code cells and exports five graphs/four tables under `graphs/module02-panel/`. It takes a few minutes on the author machine; durations vary. The [implementation record](docs/module-2-panel-implementation.md) distinguishes shared computation, logical per-arm work, measured runtime and conditional uncertainty.

## Earlier module runners

The commands below describe earlier builds. They are independent of the fresh Module 1 implementation above; deleted older artifacts are not restored by its runner.

Use Python 3.12 and run these commands from this repository's root. The numerical dependencies are pinned to the versions used for the saved examples.

```sh
uv sync --locked
.venv/bin/python -m scripts.execute_a01
.venv/bin/python -m scripts.execute_a03
.venv/bin/python -m unittest discover -s tests -p 'test_a0*.py' -v
```

The earlier runners execute [A01](atsf-01.ipynb) and [A03](a03-diagnostic-loop.ipynb) in fresh kernels using this project's Python environment, save their executed outputs, and regenerate the numbered images in `graphs/`. Use the current Module 2 runner above for A02. All default to offline replay: numerical forecasts are recomputed and the bundled genuine model responses are reused only when their inputs match. The bundled datasets and captures make these runs independent of the original repository and of Ollama. Live capture requires Ollama and the model configured in `CFG.llm`.

On this Mac, the installed SciPy wheel required a local loader repair. If a fresh installation reports `__thread_bss` with a nonzero offset, run the included repair once with this environment's Python, then rerun the notebooks:

```sh
.venv/bin/python -m scripts.repair_scipy_macos
```

The repair preserves backups of the original libraries. The [environment verification record](docs/environment-verification.md) documents the successful runs after migration.




A03 defaults to exact offline replay and needs no Ollama server. It regenerates four `atsf03-graph` and nine `atsf03-table` PNGs, then verifies the original eight terminal outcomes. [Capture details](docs/module-03-implementation.md) distinguish live inference from replay, numerical delivery from agent success, and synthetic fixtures from observed model behavior. The live capture script refuses to overwrite its original archive.
