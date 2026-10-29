"""Publication views only: calculations/dispatch remain in the shared tools."""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

from . import module02_workflow as w

LABELS = {"regular": "regular", "intermittent": "intermittent", "short_history": "short", "new_product": "new",
          "seasonal_naive": "naive", "mean28": "mean (28)", "donor_median28": "donor prior",
          "hw_add_damped": "Holt–Winters", "tsb": "TSB", "clarify": "clarify", None: "none"}


def primary_rows(result):
    return [r for r in result["rows"] if r["case"]["cohort"] == "main"]


def figure(cfg, scale=1):
    return plt.subplots(figsize=(cfg.img_dim1 * cfg.publication_scale, cfg.img_dim2 * cfg.publication_scale * scale))


def architecture(cfg):
    fig, ax = figure(cfg, .8)
    ax.set(xlim=(0, 10), ylim=(0, 7)); ax.axis("off")
    boxes = {"gate": (0.2, 3.0, 1.8, 1.0, "Check request\nand visible data"),
             "router": (2.65, 3.0, 2.1, 1., "B: rules\nC: one LLM call"),
             "regular": (6.0, 5.65, 2.7, .7, "Regular → validate candidates"),
             "intermittent": (6.0, 4.6, 2.7, .7, "Intermittent → validate candidates"),
             "short": (6.0, 3.55, 2.7, .7, "Short history → simple methods"),
             "new": (6.0, 2.5, 2.7, .7, "New product → fixed donor prior"),
             "clarify": (6.0, 1.45, 2.7, .7, "Clarify → no forecast"),
             "a": (.2, .15, 4.2, .65, "A: seasonal naive / the same donor prior"),
             "d": (5.1, .15, 4.2, .65, "D: all candidates; isolated diagnostic")}
    for key, (x, y, width, height, text) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                                   facecolor="#eaf0f4" if key != "router" else "#dcebdd", edgecolor="#25455d"))
        ax.text(x + width/2, y + height/2, text, ha="center", va="center", fontsize=11)
    ax.annotate("", (2.55, 3.5), (2.1, 3.5), arrowprops={"arrowstyle": "->", "color": "#25455d"})
    for key in ("regular", "intermittent", "short", "new", "clarify"):
        _, y, _, height, _ = boxes[key]
        ax.annotate("", (5.9, y + height/2), (4.85, 3.5), arrowprops={"arrowstyle": "->", "color": "#25455d"})
    ax.text(3.7, 2.2, "Validate proposal\nReject → one rule fallback", ha="center", fontsize=10)
    ax.set_title("One bounded decision; Python owns every forecast", pad=16)
    fig.tight_layout()
    return fig


def histories(cfg, views):
    fig, axes = plt.subplots(2, 2, figsize=(cfg.img_dim1 * cfg.publication_scale, cfg.img_dim2 * cfg.publication_scale), layout="constrained")
    for ax, cid, title in zip(axes.flat, ["M02-F01", "M02-F05", "M02-S01", "M02-N01"],
                              ["Established: visible history", "Intermittent: visible history", "Short history: only 60 days", "New-product scenario: no target history"]):
        view = views[cid]
        h = w.as_series(view["history"])
        if len(h):
            ax.plot(h.index, h, color=cfg.colors["history"], linewidth=2, label="Visible target sales")
        else:
            for sid, donor in view["donors"].items():
                series = w.as_series(donor["history"])
                ax.plot(series.index, series, alpha=.65, linewidth=1.3, label="Donor " + sid.split("_CA")[0])
            ax.text(.02, .95, "Target observations: 0", transform=ax.transAxes, va="top")
        ax.set_title(title, fontsize=12); ax.set_ylabel("Recorded item units / day")
        ax.legend(fontsize=9, loc="best", frameon=True, facecolor="white", framealpha=.95)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return fig


def forecasts(cfg, result, views):
    by_id = {r["case"]["case_id"]: r for r in result["rows"]}
    fig, axes = plt.subplots(2, 2, figsize=(cfg.img_dim1 * cfg.publication_scale, cfg.img_dim2 * cfg.publication_scale), layout="constrained")
    for ax, cid in zip(axes.flat, ("M02-F01", "M02-F05", "M02-S01", "M02-N01")):
        row = by_id[cid]; h = w.as_series(views[cid]["history"]).tail(60)
        if len(h): ax.plot(h.index, h, color=cfg.colors["history"], linewidth=2, label="Visible history")
        dates = w.pd.DatetimeIndex(row["actual_dates"])
        ax.plot(dates, row["actual"], color=cfg.colors["actual"], linewidth=2, label="Final actuals")
        groups = []
        for a in "ABC":
            out = row["arms"][a]["numerical"]
            if not out or out.get("forecast") is None:
                ax.text(.02, .93, f"{a}: no forecast", transform=ax.transAxes, va="top")
                continue
            existing = next((g for g in groups if np.array_equal(g["values"], out["forecast"])), None)
            if existing: existing["arms"].append(a)
            else: groups.append({"arms": [a], "values": out["forecast"]})
        for g in groups:
            a = g["arms"][0]
            label = " = ".join(g["arms"])
            if "C" in g["arms"] and row["arms"]["C"]["decision"]["source"] == "rule_fallback": label += " (C: rules)"
            ax.plot(dates, g["values"], label=label, color=cfg.colors[a], linestyle={"A": ":", "B": "--", "C": "-."}[a],
                    linewidth=2, marker="o" if a != "A" else None, markevery=7)
        ax.axvline(w.pd.Timestamp(w.ORIGIN), color="gray", linestyle="--", linewidth=1)
        label = "zero-history scenario" if row["case"]["scenario"] == "new" else row["case"]["scenario"] + " history"
        ax.set_title(f"{cid.removeprefix('M02-')} · {label}", fontsize=12)
        ax.set_ylabel("Recorded item units / day")
        ax.legend(fontsize=9, loc="upper left", frameon=True, facecolor="white", framealpha=.95)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return fig


def table_cases():
    return [(["Case class", "Cases", "Target days", "Validation before origin"],
             [["Full history", 12, 449, "3 × 28 days; 365-day training"],
              ["Short view", 2, 60, "28 days; 32-day training"],
              ["New-product view", 2, 0, "No target validation; fixed donor prior"],
              ["Additional input cases", 4, "0 or 449", "Separate ambiguity / gate checks"]], [.23, .1, .16, .51])]


def table_workflows():
    return [(["Route", "Eligibility", "Candidates / action"],
             [["regular", "449+ observed days", "Seasonal naive; damped additive Holt–Winters"],
              ["intermittent", "449+ observed days", "Seasonal naive; TSB (0.10, 0.10)"],
              ["short_history", "56–448 observed days", "Seasonal naive; trailing 28-day mean"],
              ["new_product", "0 days + declared launch + authorized donors", "Median of three donor 28-day means"],
              ["clarify", "Missing / inconsistent requirements", "Ask for information; no forecast"]], [.19, .35, .46])]


def table_routes(result):
    rows = []
    for row in primary_rows(result):
        b, c = (row["arms"][a]["decision"] for a in "BC")
        e = c["evidence"]; window = e["history.current_window"]
        rows.append([row["case"]["case_id"].removeprefix("M02-"), e["history.available_span"]["observations"],
                     f'{window["zero_count"]} / {window["observed_count"]}' if window["observed_count"] else "N/A",
                     LABELS[b["accepted_route"]], LABELS[c["proposed_route"]],
                     (LABELS[c["executed_route"]] + " / rules") if c["source"] == "rule_fallback" else "none / clarify"])
    return [(["Case", "Days", "Zeros / window", "B route", "C proposal", "C execution"], rows, [.1, .08, .18, .18, .18, .28])]


def table_accuracy(result):
    stats = w.summary(result)["full_history"]
    full = [[a, f'{s["delivered"]} / {s["total"]}', s["common_n"], s["scaled_n"], s["macro_mase"]] for a, s in stats.items()]
    constructed = [[r["case"]["case_id"].removeprefix("M02-"),
                    *[r["arms"][a]["metrics"]["mae"] if r["arms"][a]["metrics"] else None for a in "ABC"]]
                   for r in primary_rows(result) if r["case"]["scenario"] != "full"]
    return [(["Full-history arm", "Delivered", "Common n", "Scaled n", "Macro MASE"], full, [.24, .19, .18, .17, .22]),
            (["Constructed case", "A MAE", "B MAE", "C MAE"], constructed, [.4, .2, .2, .2])]


def table_failures(result, fixtures):
    rows = []
    for row in result["rows"]:
        if row["case"]["cohort"] == "main": continue
        c, b = row["arms"]["C"]["decision"], row["arms"]["B"]["decision"]
        rows.append([row["case"]["case_id"].removeprefix("M02-"), row["case"]["cohort"] + " request",
                     b["status"], c["status"], c.get("archived_attempts", 0), "none"])
    for row in fixtures:
        c = row["decision"]
        rows.append([row["case_id"].removeprefix("M02-"), "injected response", "not a B trial", "rejected", 0, LABELS[c["executed_route"]] + " / rules"])
    return [(["Case", "Provenance", "B status", "C / fixture", "Live calls", "Recovery"], rows, [.09, .21, .17, .18, .1, .25])]


def work_summary(result, capture, audit):
    output = {"capture_session": capture["session_id"], "numerical_session": result["session_id"], "dispatch": [], "numerical": [], "audit": []}
    for cohort, arms in [("main", "ABC"), ("ambiguity", "C")]:
        cases = [r["case"]["case_id"] for r in result["rows"] if r["case"]["cohort"] == cohort]
        for a in arms:
            records = [capture["requests"][cid][a] for cid in cases]
            def tokens(key):
                values = [r.get("response", {}).get("usage", {}).get(key) for r in records] if a == "C" else [0] * len(records)
                return sum(values) if all(v is not None for v in values) else None
            output["dispatch"].append({"cohort": cohort, "arm": a, "n": len(records), "calls": sum(r["llm_attempts"] for r in records),
                                       "seconds": sum(r["seconds"] for r in records), "median_seconds": float(np.median([r["seconds"] for r in records])),
                                       "input_tokens": tokens("prompt_eval_count"), "output_tokens": tokens("eval_count")})
        c = [capture["requests"][cid]["C"] for cid in cases]
        judgments = [r for r in audit["reviews"] if r["case_id"] in cases]
        output["audit"].append({"cohort": cohort, "n": len(c), "schema_valid": sum(bool(r.get("schema_valid")) for r in c),
                                "accepted_forecasts": sum(r["accepted_route"] in w.ROUTES[:-1] for r in c),
                                "fallbacks": sum(r["source"] == "rule_fallback" for r in c),
                                "abstentions": sum(r["status"] == "abstained" for r in c),
                                "reasons": dict(Counter(r["reason_rating"] for r in judgments))})
    for a in "ABCD":
        rows = [r for r in primary_rows(result) if a in r["arms"]]
        numerical = [r["arms"][a]["numerical"] for r in rows if r["arms"][a]["numerical"]]
        output["numerical"].append({"arm": a, "requests": len(rows), "delivered": sum(o.get("forecast") is not None for o in numerical),
                                    "fits": sum(len(o["fit_trace"]) for o in numerical),
                                    "seconds": sum(o["seconds"] for o in numerical), "cache_reuse": 0})
    return output


def shape_valid(value, evidence):
    """Check the exact small JSON schema, separately from Python's cross-field contract."""
    if not isinstance(value, dict) or set(value) != {"route", "evidence_ids", "reason", "clarification_question"}:
        return False
    ids, reason, question = value["evidence_ids"], value["reason"], value["clarification_question"]
    return (isinstance(value["route"], str) and value["route"] in w.ROUTES
            and isinstance(ids, list) and bool(ids) and all(isinstance(i, str) and i in evidence for i in ids)
            and len(set(ids)) == len(ids) and isinstance(reason, str) and len(reason) >= 1
            and (question is None or isinstance(question, str) and len(question) >= 1))


def routing_metrics(result, capture, audit):
    """Integer denominators distinguish JSON shape, contract, capability and policy."""
    output = {"capture_hash": capture["bundle_hash"], "cohorts": {}}
    for cohort in ("main", "ambiguity", "gate"):
        rows = [r for r in result["rows"] if r["case"]["cohort"] == cohort]
        archived = [capture["requests"][r["case"]["case_id"]]["C"] for r in rows]
        attempts = sum(d["llm_attempts"] for d in archived)
        calls = [d for d in archived if d["llm_attempts"]]
        def count(n, denominator):
            return {"numerator": n, "denominator": denominator, "display": f"{n}/{denominator}" if denominator else "N/A"}
        shape_count = sum(shape_valid(w.strict_json(d["response"]["text"]), d["evidence"])
                          for d in calls if d.get("response"))
        accepted = [d for d in calls if d.get("accepted_route") is not None]
        forecasting = [d for d in accepted if d["executed_route"] is not None]
        reviews = [r for r in audit["reviews"] if r["case_id"] in {x["case"]["case_id"] for x in rows}]
        output["cohorts"][cohort] = {
            "requests": len(rows), "attempts": attempts,
            "json_schema_shape_valid": count(shape_count, attempts),
            "decision_contract_valid": count(sum(bool(d.get("schema_valid")) for d in calls), attempts),
            "accepted_eligible_decisions": count(len(accepted), attempts),
            "accepted_forecasting_decisions": count(len(forecasting), attempts),
            "policy_agreement_accepted_forecasts": count(sum(d["accepted_route"] == d["policy_route"] for d in forecasting), len(forecasting)),
            "clarifications": count(sum(d["status"] == "abstained" for d in calls), len(rows)),
            "questions_request_missing_requirements": count(sum(r["question_rating"].startswith("relevant") for r in reviews), len(reviews)),
            "unnecessary_questions": count(sum(r["question_rating"] == "unnecessary" for r in reviews), len(reviews)),
            "unnecessary_clarifications": count(sum(d["status"] == "abstained" and any(r["case_id"] == row["case"]["case_id"] and r["question_rating"] == "unnecessary" for r in reviews)
                                                       for d, row in zip(archived, rows)), len(rows)),
            "rule_fallbacks": count(sum(d["source"] == "rule_fallback" for d in calls), attempts),
            "input_gate_stops": count(sum(d["source"] == "input_gate" for d in archived), len(rows)),
            "reason_review": dict(Counter(r["reason_rating"] for r in reviews)),
        }
    main = primary_rows(result)
    output["numerical_fallbacks"] = {a: {"numerator": sum(i["status"] == "model_fallback" for i in [r["arms"][a]["numerical"] for r in main if a in r["arms"]] if i),
                                         "denominator": sum(bool(r["arms"][a]["numerical"]) for r in main if a in r["arms"])} for a in "ABCD"}
    output["notes"] = ["Shape checks implement the exact supplied small schema; cross-field/null and word-limit rules are additional contract checks.",
                       "Archived schema_valid means the complete Python decision contract, retained for capture integrity.",
                       "Zero accepted forecasting decisions give N/A policy agreement, not zero or perfect compliance.",
                       "A relevant question can contain redundancy or accompany an inaccurate reason; see the manual audit."]
    return output


def table_work(work):
    audit_rows = [[r["cohort"], f'{r["schema_valid"]}/{r["n"]}', r["accepted_forecasts"], r["fallbacks"], r["abstentions"], r["reasons"].get("supported", 0)] for r in work["audit"]]
    dispatch = [[r["cohort"] + " " + r["arm"], r["n"], r["calls"], r["seconds"], r["input_tokens"], r["output_tokens"]] for r in work["dispatch"]]
    nums = [[r["arm"], r["requests"], r["delivered"], r["fits"], r["seconds"]] for r in work["numerical"]]
    return [(["C cohort", "Contract valid", "Accepted forecasts", "Fallbacks", "Abstained", "Supported reasons"], audit_rows, [.17, .16, .19, .16, .16, .16]),
            (["Archived dispatch", "Cases", "Calls", "Seconds", "Input tokens", "Output tokens"], dispatch, [.25, .1, .1, .17, .19, .19]),
            (["Current numeric arm", "Requests", "Delivered", "Fit attempts", "Seconds"], nums, [.28, .17, .17, .2, .18])]
