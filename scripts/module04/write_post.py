from pathlib import Path
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[2]
E = ROOT/"examples/module04"
manifest = json.loads((E/"asset-manifest.json").read_text())
run = E/"runs"/manifest["inference_run_id"]
capture = json.loads((run/"capture.json").read_text())
result = json.loads((run/"results.json").read_text())

def visual(number):
    a = manifest["assets"][number-1]
    label = f"{a['type'].capitalize()} {number:02d}"
    return f"> [Insert {label} here — {a['title']}]\n\n![{a['alt_text']}]({a['filename']})\n\n*{label}. {a['caption']}*"

post = '''# Plan the forecasting experiment before executing it

Forecasting work tends to arrive as a queue of decisions. Should we transform demand? Which validation periods represent the task? Does a model need weekly seasonality, annual seasonality, or both? Which failed check should stop the next expensive fit? A notebook can answer these questions and still leave its dependencies scattered across twenty cells.

This episode puts those decisions into an executable plan. A local model writes the experiment, Python follows its prerequisites, and a failed residual check sends the remaining work back for one revision. On ten daily store series, the revision improves most forecasts while leaving five diagnostic checks unresolved. That combination is useful: a forecast can improve while an investigation remains unfinished.

## Why write the experiment before running it?

Plan-and-execute separates two jobs. The planner specifies the work and its dependencies; the executor runs numerical tools and records what happened. A monitor can interrupt the plan when evidence requires another decision. The [supplied source document](docs/a04-Plan-and-Execute%20Forecasting%20Agents.docx) describes this architecture. [LangChain's planning-agents article](https://www.langchain.com/blog/planning-agents) documents a similar separation of planning, execution and replanning. Its architectural motivations do not establish forecasting gains here.

In [post 1's notebook](a01-forecasting-workflow.ipynb), the sequence is fixed. In [Episode 3](a03-diagnostic-loop.ipynb), a model chooses its next diagnostic after each observation. Here the model writes a complete plan first and returns only when a check fails. Python owns fitting, ranking and scoring throughout.

The forecasting ingredients come from [m10's state-space notebook](m10-state-space.ipynb): transformations, trend models, weekly and annual seasonality, and predictive intervals. The [companion notebook](a04-plan-and-execute.ipynb) keeps the implementation inline, with a structured plan, an execution ledger and the actual revision. Its default mode analyses a saved genuine run; the launch guide explains a new local run.

## Put the future behind a boundary

We use the [Store Item Demand Forecasting competition data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) already available in the series material. The source has 913,000 rows: ten stores, 50 items and 1,826 daily dates. Summing each store's items gives ten forecasting tasks and 18,260 store-days. All ten stores stay in the experiment. The competition source does not establish authentic retailer provenance or population representativeness.

The task is to forecast the next 90 days of each store's total demand. Permitted history runs from 1 January 2013 through 2 October 2017. Three expanding validation folds predict 6 January–5 April, 6 April–4 July and 5 July–2 October 2017. Every candidate faces those same 90-day tasks.

The final period, 3 October–31 December 2017, is excluded from planning, fitting and model selection. We freeze forecasts and audit the model's explanations before scoring it. This is familiar historical competition data, already explored in m10, so the boundary does not turn it into a prospective or unseen test. Store 1 was chosen in advance for the reader plots.

VISUAL_01

Store 1 has fast weekly variation inside a slower annual envelope. A model that carries only a changing level and slope will have plenty to leave unexplained. The validation blocks include different parts of that envelope, which matters when a 90-day forecast crosses into a different season.

## Give the planner an executable contract

We deliberately start with a trend-only challenge, following m10's structural ladder. The planner chooses the original sales scale or a log transformation and specifies the complete contract. Seasonal naive—repeat the last seven days—is mandatory. Horizon, folds, diagnostic rule and uncertainty definitions are fixed. After failure, the planner can add either seasonal candidate or both.

The small dependency graph is also prescribed. This keeps the lesson focused on transformation and model-family decisions, their execution, and a bounded revision. It does not test whether a model can invent a good arbitrary task graph. The first rung is an assigned simplification, not an independently discovered planning mistake.

The genuine planner is `qwen3.5:9b-q8_0` through local Ollama, with temperature 0 and zero retries. These are selected fields from its actual initial plan; the dependency graph follows separately.

PLAN_EXCERPT

VISUAL_02

The live planner chose **identity**, so the agent's models use sales units directly. It selected the required trend candidate but gave no specific justification for its transformation choice. The full plan also names the three validation origins, residual check and nominal 80% intervals, including a separate empirical interval rule for seasonal naive.

The initial trend model has a stochastic level and deterministic slope. The level can receive new random shocks; the slope does not. We made that constraint before freezing the protocol: an unrestricted stochastic-slope model produced explosive long-horizon variance in a synthetic development pilot. The seasonal structural candidate retains a stochastic slope. This is a documented numerical adaptation of m10.

Schema validation checks the JSON shape. Semantic validation also checks cutoffs, candidate names, prerequisites and evidence identities. Neither establishes that the planner's explanation is true. A nicely formatted plan can still make a poor choice.

## Execute until a check fails

Python evaluates candidates using mean RMSE across the three validation folds. RMSE squares errors before averaging, so large misses matter more. Nonconverged or nonfinite fits are ineligible; exact ties follow the declared catalog order. Seasonal naive wins the initial comparison in every store.

The next check examines the **trend candidate**, even though it was not selected for delivery. For each store, we take the last 364 standardized one-step innovations—the model's scaled surprises after seeing earlier observations—and run a Ljung–Box check through lag 28. A p-value below the fixed 0.05 threshold triggers review. These are fitted one-step residuals, not 90-day forecast errors.

VISUAL_03

Store 1's remaining autocorrelation peaks at weekly multiples; its lag-seven value is about 0.90. All ten trend checks fail. Some computed p-values underflow to floating-point zero. The residual pattern supports investigating seasonal structure, while the joint autocorrelation test does not identify a unique cause or certify a replacement.

VISUAL_04

The executor has completed data checks, transformation, baseline evaluation and trend evaluation. The failed monitor blocks uncertainty and delivery in the original graph. The record makes the pause concrete: completed work stays completed, the failure stays failed, and the remaining steps cannot quietly proceed as though nothing happened.

## Revise the remaining work

The replanner receives all ten stores' fold scores and monitor values, plus hashes of completed artifacts. Full arrays stay in the archive; final outcomes and fixed-arm results are absent. The transformation and experimental boundary must remain unchanged.

It chooses both repairs. **BSM**, the basic structural model, adds a weekly seasonal state and three annual sine/cosine pairs to a local linear trend. **SARIMAX** uses weekly ARIMA dependence with three annual Fourier pairs as regressors. Those calendar regressors continue from an absolute daily time origin, so forecasting does not restart the annual phase. Both are direct adaptations of m10's daily-demand example.

VISUAL_05

The new suffix evaluates the repairs, checks each validation-selected structural model, records the disposition and delivers. It reuses 20 baseline/trend artifacts across the ten stores. The old blocked steps become superseded; the original failed event remains in the ledger.

Validation selects SARIMAX in eight stores and BSM in two. The repeated residual check passes in stores 1, 2, 3, 4 and 7, and fails in the other five. The shared controller therefore ends as **diagnostic unresolved**. All ten forecasts are delivered, with those individual diagnostic outcomes retained. The one-revision budget stops the investigation here.

The explanation audit also preserves a limitation. Of six factual propositions across the two public reasons, four are supported and two are ambiguous; one purpose statement is excluded as a proposal. The revision's singular “p-value 0.0” does not identify which store it means, and its seasonality diagnosis goes beyond what the joint test establishes. This author audit uses only the evidence available at each request.

## Freeze forecasts, then open the outcome

The agent's identity-scale structural forecasts use Gaussian predictive means and nominal 80% marginal intervals. A log-scale model needs care: `exp(mu)` is a median, while `exp(mu + variance/2)` is the plug-in conditional mean used here for RMSE. Here `mu` and `variance` are the predictive mean and variance on the log scale. Exponentiated Gaussian quantiles give asymmetric intervals. These intervals ignore parameter uncertainty; they do not promise simultaneous coverage of the whole 90-day path.

Seasonal naive uses a symmetric width from the 80th percentile of its absolute validation errors, without a coverage guarantee. We also retain the initial selection without repair and a fixed plan that evaluates log-scale BSM against the same baseline. All four arms use the same boundary and final dates.

VISUAL_06

For store 1, the repaired forecast has RMSE 235.9 versus 461.8 for seasonal naive. The initial and fixed plans select that same naive forecast, so their lines coincide. The repaired trajectory follows the changing seasonal level more closely, but its interval covers only 58.9% of this store's final days. A better point forecast has not settled uncertainty calibration.

VISUAL_07

Every arm delivers for all ten stores. The repaired plan's pooled RMSE is **342.6**, against **518.7** for the other arms; mean MASE is **2.38** versus **3.48**. MASE scales each store's MAE (mean absolute error) by its permitted history's mean absolute seven-day change. It is not the ratio to that store's final naive error.

The agent has lower RMSE in nine stores. Store 3 goes the other way: 709.3 versus 557.7 for seasonal naive. Its error also contributes strongly to pooled RMSE. All 40 store/arm records, including store 3, remain available.

The fixed log-BSM plan selects seasonal naive everywhere: eight BSM candidates lose validation and two fail convergence. This result belongs to that configuration. The repaired arm changes transformation and candidate menu as well as control; we did not compare it with a deterministic plan using the same expanded menu. The error difference cannot isolate a benefit from LLM planning.

Average agent coverage is 76.7%, with mean interval width 954.0 units, versus 69.3% and 1,235.0 for the naive forecasts. Its interval score is also lower, 1,228.8 versus 1,978.7; that score penalizes both width and missed outcomes. Dependence across stores and days, and the variation visible in store 1, limit a calibration claim from this one final period.

VISUAL_08

The shared agent workload takes 187.92 seconds, makes two model requests and attempts 120 numerical fits. The fixed BSM workload takes 44.15 seconds and attempts 38 fits, including its two failed attempts. Baseline computation takes about 0.57 milliseconds. The initial counterfactual shares the agent's completed prefix; it has no separately measured independent runtime.

Those are original live timings, excluding development and publication export. The two responses contain 674 output tokens. Reuse avoided redoing completed artifacts, but the agent evaluated a larger menu and did more work overall. This experiment does not measure a speed advantage over a reactive agent or a fixed search of the same menu.

## Check the controller when things go wrong

The genuine run is one observation of model behavior. Separate development checks exercise contract validation, stale evidence, convergence failure, fallback delivery, request limits, timeout accounting and archive rejection at the component level. They also mutate withheld values in memory and check that earlier decision inputs and real trend/baseline forecasts remain unchanged.

VISUAL_09

All nine checks pass. They establish selected software invariants, not the model's real-world failure probability. In particular, annual Fourier features continue across training and forecast boundaries, and a revision preserves failed history while superseding blocked work. Missing or changed archives fail closed.

The notebook's default Run All passed in a fresh kernel with model/network access blocked. This is **analysis-only replay**: it verifies saved identities and regenerates scores and visuals from frozen numerical artifacts. It does not refit models or independently reproduce their forecasts.

[m09's multivariate methods](m09-causal-methods.ipynb) could extend the idea to VAR or VECM planning. That would add questions about alignment, lags and future regressor availability. The present experiment stays with ten univariate store totals, so it makes no multivariate or causal forecasting claim.

## Closing time

The explicit plan made the revision inspectable. We can see the original transformation and candidate, the completed dependencies, the failed check, the retained artifacts and the replacement suffix. That record is the concrete benefit demonstrated here.

The revision improved point forecasts in nine stores and reduced pooled error, but five diagnostic checks remained unresolved and interval coverage varied substantially. The fixed reference had convergence failures and selected its baseline everywhere. One shared plan on ten dependent historical competition series cannot establish general superiority or attribute the gains to the planning architecture.

Use this pattern when a forecasting task has several dependent decisions and a failed check should change the remaining work. Keep the plan small, the numerical boundary enforceable, and the distinction between “delivered” and “resolved” visible.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
'''
for i in range(1,10):
    post = post.replace(f"VISUAL_{i:02d}", visual(i))
selected_fields = {k:capture["initial_plan"][k] for k in ["transform","horizon","origins","candidates","diagnostic","uncertainty"]}
excerpt = "```json\n{\n" + ",\n".join("  " + json.dumps(k) + ": " + json.dumps(v) for k,v in selected_fields.items()) + "\n}\n```"
post = post.replace("PLAN_EXCERPT", excerpt)
(ROOT/"a04-post.md").write_text(post)

# The outline is evidence-specific and preserves exactly the shared hierarchy.
outline = ["# Episode 4 executed post outline", "", "Source: the frozen genuine retail cohort, author audit and validated analysis-only replay."]
sections = re.split(r"(?m)^## ", post)[1:]
for section in sections:
    heading, body = section.split("\n",1)
    outline.extend(["", "## "+heading, "", "Purpose: "+body.strip().split("\n\n")[0]])
    for asset in manifest["assets"]:
        if asset["section"] == heading:
            label = f"{asset['type'].capitalize()} {asset['visual_number']}"
            caption = f"*{label}. {asset['caption']}*"
            reading = post.split(caption,1)[1].strip().split("\n\n")[0]
            outline.extend(["", visual(int(asset["visual_number"])), "", "Immediate reading: "+reading])
(ROOT/"docs/module-4-post-outline.md").write_text("\n".join(outline)+"\n")

# Prose-only notebook completion: retain all executed computational sources/outputs.
path = ROOT/"a04-plan-and-execute.ipynb"
nb = nbformat.read(path,as_version=4)
interpretations = {
1: "Store 1 has weekly oscillation within an annual envelope; the three shaded validation blocks cover distinct parts of that envelope. All ten stores remain in the cohort.",
2: "The genuine planner chose identity scale and the assigned trend candidate. It did not explain the transformation choice specifically. Its graph is prescribed; meaningful free decisions here are transformation and repair candidate subset.",
3: "Store 1's trend residual ACF7 is0.899. All ten trend checks fail; some p-values underflow to floating-point zero. Weekly-lag dependence motivates a seasonal hypothesis, while Ljung–Box does not identify its cause.",
4: "All ten initial validation selections are seasonal naive. The failed checks concern the trend candidates, not those selected baselines. Completed work is retained and two pending steps are blocked.",
5: "The genuine revision retains identity scale and adds both BSM and SARIMAX. Validation selects SARIMAX in eight stores and BSM in two. Stores1/2/3/4/7 pass the repeated check; the other five fail, so the shared controller is diagnostic_unresolved despite ten deliveries. Twenty existing baseline/trend artifacts are reused; the original failed event remains and blocked steps become superseded. The author audit finds4/6 factual propositions supported and2 ambiguous, with one proposal excluded; the singular p-value and inferred cause are ambiguous.",
6: "Store1 agent RMSE is235.9 versus461.8 for seasonal naive; the initial and fixed selections coincide with that naive line. Agent interval coverage is only58.9% here, despite lower point error.",
7: "All four arms deliver10/10 cases. Agent pooled RMSE is342.6 versus518.7, and mean MASE is2.38 versus3.48. Agent RMSE is lower in nine stores; store3 loses709.3 versus557.7. Fixed log BSM loses validation in eight stores and fails convergence in two, selecting naive everywhere. The comparison changes transformation and menu, so it cannot isolate LLM-planning benefit. Agent mean coverage76.7%, width954.0 and interval score1228.8 improve on naive69.3%,1235.0 and1978.7, but do not establish calibrated uncertainty across dependent stores/days.",
8: "The original agent workload takes187.92s, with2 model requests,120 fits and674 output tokens. Fixed BSM takes44.15s and38 fit attempts, including two failures. Baseline time is0.57ms, not zero. No-repair shares the initial prefix and has no independent runtime. The workload differs, so this is not an efficiency comparison against an equivalent search.",
9: "All nine development checks pass, outside genuine-agent denominators. Fresh-kernel analysis-only replay also passed with model/network sockets blocked; it verifies frozen analysis and does not recompute numerical tools or decisions."}
for c in nb.cells:
    if c.cell_type != "markdown":
        continue
    c.source = re.sub(r"\b(Graph|Table)(\d{2})\b",r"\1 \2",c.source)
    if c.source.startswith("## Give the planner an executable contract") and "Selected fields from the genuine initial plan" not in c.source:
        c.source += "\n\nSelected fields from the genuine initial plan are shown below; Table 02 gives its dependency graph separately. The live planner is `qwen3.5:9b-q8_0` through local Ollama, temperature 0, zero retries.\n\n"+excerpt
    if "The level can receive new random shocks; the slope does not." not in c.source:
        c.source = c.source.replace("The trend-only model contains a stochastic level and a deterministic slope.", "The trend-only model contains a stochastic level and a deterministic slope. The level can receive new random shocks; the slope does not.")
    match = re.match(r"\*(Graph|Table) (\d{2})\.", c.source)
    if match:
        number = int(match.group(2)); a=manifest["assets"][number-1]
        c.source = f"*{a['type'].capitalize()} {number:02d}. {a['caption']}*\n\n"+interpretations[number]
    if c.source.startswith("## Closing time"):
        c.source = """## Closing time

The explicit plan makes the revision inspectable: identity transformation, a prescribed trend-only first rung, completed dependencies, a failed check, retained artifacts and a replacement suffix. One genuine shared revision adds weekly/annual BSM and SARIMAX. It improves RMSE in nine of ten stores and lowers pooled error from518.7 to342.6, while five repeated diagnostics remain unresolved. Delivery succeeds for all ten stores; uncertainty varies substantially, including only58.9% coverage for predeclared store1.

The fixed log-BSM reference selects seasonal naive everywhere, with two convergence failures and eight validation losses. This comparison changes transformations and candidate menu and cannot isolate the effect of LLM planning. One first-attempt shared plan/replan on ten dependent historical competition series does not establish general superiority, causal diagnosis or human-time savings.

Use plan-and-execute when several modelling decisions depend on earlier checks and you need a durable record of what changed. Keep workflow completion, diagnostic resolution, point accuracy and uncertainty calibration distinct. Default analysis-only replay verifies frozen artifacts, not numerical model reproduction.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]"""
    # Repair compact authoring whitespace around numbers in prose only.
    for before,after in {"913,000 rows":"913,000 rows", "all50":"all 50", "each50":"each 50", "ten90":"ten 90", "last364":"last 364", "last1":"last 1", "first1,736":"first 1,736", "since1January2013":"since 1 January 2013", "1January2013":"1 January 2013", "maximum500":"maximum 500", "most500":"most 500", "nominal80":"nominal 80", "nominal 80%":"nominal 80%", "at90":"at 90", "three90":"three 90", "each90":"each 90", "those90":"those 90", "the80th":"the 80th", "all40":"all 40", "through28":"through 28", "fixed.05":"fixed .05", "a90":"a 90", "about10":"about 10", "Episode3":"Episode 3", "Post1":"Post 1", "store1":"store 1", "Store1":"Store 1", "store3":"store 3", "Stores1":"Stores 1"}.items():
        c.source=c.source.replace(before,after)
    words = "contain|contains|has|from|to|is|in|with|for|of|at|than|about|all|each|first|last|only|the|The|and|over|three|maximum|through|takes|versus|stores|store|Stores|Store|NumPy|Python|SciPy|pandas|statsmodels|Matplotlib|before|after|a|permitted|counts|finds|supported|ambiguous|Request|Requests"
    lines=[]; fenced=False
    for line in c.source.splitlines():
        if line.startswith("```"):
            fenced=not fenced; lines.append(line); continue
        if not fenced:
            parts=re.split(r"(`[^`]*`)",line)
            for j in range(0,len(parts),2):
                t=parts[j].replace("m 10","m10").replace("m 09","m09").replace("all 50","all 50")
                t=re.sub(r"\b("+words+r")(?=\d)",r"\1 ",t)
                t=re.sub(r"(?<=\d)(?=January|February|March|April|May|June|July|August|September|October|November|December)"," ",t)
                t=re.sub(r",(?=[A-Za-z])",", ",t)
                parts[j]=t
            line="".join(parts)
        lines.append(line)
    c.source="\n".join(lines)
nbformat.write(nb,path)

claims=[]
for i,s in enumerate(result["summaries"]):
    for field in ["delivered","denominator","rmse","mae","mase","coverage","width","interval_score"]:
        claims.append({"kind":"result_field","label":s["arm"]+" "+field,"path":["summaries",i,field],"value":s[field]})
for key in ["controller_status","agent_seconds","fixed_seconds","baseline_seconds"]:
    claims.append({"kind":"capture_field","label":key,"path":[key],"value":capture[key]})
for store in [1,3]:
    for arm in ["agent_plan","seasonal_naive"]:
        i=next(i for i,r in enumerate(result["records"]) if r["store"]==store and r["arm"]==arm)
        for field in ["rmse","coverage"]:
            claims.append({"kind":"result_field","label":f"store{store} {arm} {field}","path":["records",i,field],"value":result["records"][i][field]})
record={"run_id":capture["run_id"],"protocol_sha256":capture["protocol_sha256"],"claims":claims,
        "reviewed_derived_claims":{"original_rows":"data/module04/manifest.json:raw_shape[0]=913000","store_days":"manifest.store_days=18260",
                                  "cohort":"protocol.case_order stores1–10;40 scores","initial_failures":"10/10 initial_monitors failed_check",
                                  "remaining_failures":"stores5/6/8/9/10 failed_check","selected_models":"eight sarimax,two bsm in arms.agent_plan",
                                  "wins":"nine of ten paired store RMSEs;store3 loses","fits":"len(work.fits)=120;len(fixed_work.fits)=38,two failed",
                                  "tokens":"sum requests raw_response.eval_count=674","audit":"audit.json counts4 supported/2 ambiguous/1 proposal",
                                  "checks":"development/checks.json contains9 passed checks"},
        "background_sources":["supplied source DOCX, operational lifecycle/reflection/pathologies","m10-state-space.ipynb daily-demand sections","https://www.langchain.com/blog/planning-agents","https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data"],
        "scope":"retrospective teaching comparison; changed transform/menu; no causal architecture, multivariate, adoption, prospective, or speed-superiority claim"}
(E/"claims.json").write_text(json.dumps(record,indent=2))
print("Post words:",len(post.split()),"Claims:",len(claims))
