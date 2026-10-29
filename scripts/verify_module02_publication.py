"""Validate the executed Module 2 publication without live inference."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from agentic_ts import module02_workflow as w
from agentic_ts import module02_publication as p
from scripts.audit_module02 import REVIEWED_CAPTURE


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads((w.ROOT / path).read_text())


def record(name):
    return read(f"examples/module02/{name}.json")


def prose(text):
    """Exclude fenced code when reading headings, links and empirical prose."""
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```\s*$", "", text)


def local_links(text, parent):
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose(text)):
        if "://" not in target and not target.startswith("#"):
            require((parent / target.split("#")[0]).is_file(), f"Broken relative link: {target}")


def reference_value(reference):
    path, pointer = reference.split("#", 1)
    value = read(path)
    for key in pointer.strip("/").split("/") if pointer.strip("/") else []:
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def main():
    folder = w.ROOT / "examples/module02"
    notebook_path, post_path = w.ROOT / "a02-routing.ipynb", w.ROOT / "a02-post.md"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    post = post_path.read_text()
    codes = [c for c in notebook.cells if c.cell_type == "code"]
    require(len(codes) == 27 and [c.execution_count for c in codes] == list(range(1, 28)),
            "Require a complete sequential clean-kernel execution")
    headings, sections, section = [], {}, None
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            for title in re.findall(r"^## (.+)$", prose(cell.source), re.M):
                if title not in {"Setup", "Utils"}:
                    headings.append(title); section = title
            local_links(cell.source, notebook_path.parent)
        else:
            sections[cell.id] = section
            require(i + 1 < len(notebook.cells) and notebook.cells[i+1].cell_type == "markdown",
                    f"Code cell lacks immediate explanation: {cell.id}")
            require(all(o.output_type != "error" for o in cell.outputs), f"Execution error: {cell.id}")
            require(all("text/html" not in o.get("data", {}) for o in cell.outputs), "Uninventoried HTML table")
    require(headings == w.HEADINGS == re.findall(r"^## (.+)$", prose(post), re.M), "Substantive headings differ")
    local_links(post, post_path.parent)

    panel, source, cases = w.load_inputs()
    capture, protocol = record("recorded_run"), record("protocol")
    w.verify_capture(capture)
    require(capture["bundle_hash"] == REVIEWED_CAPTURE and capture["complete"], "Reviewed primary capture changed")
    require(capture["protocol_hash"] == w.digest(protocol), "Protocol binding changed")
    frozen, author = record("reader_frozen_forecasts"), record("frozen_forecasts")
    result, original = record("reader_results"), record("numerical_results")
    require(w.semantic(frozen) == w.semantic(author), "Reader forecasts differ from author forecasts")
    require(w.semantic(result) == w.semantic(original), "Reader scores differ from author scores")
    require(result["frozen_semantic_hash"] == w.digest(w.semantic(frozen)) and result["scored_after_freeze"],
            "Scores are not bound to frozen forecasts")
    require(all("actual" not in row and "actual_dates" not in row for row in frozen["rows"]), "Outcomes entered freeze")
    by_id = {row["case"]["case_id"]: row for row in result["rows"]}
    route_rows = []
    checked_metrics = 0
    with patch.object(w.OllamaClient, "complete", side_effect=AssertionError("Network forbidden in verification")):
        for case in cases:
            cid = case["case_id"]; row = by_id[cid]
            view = w.request_view(case, panel, source)
            require(row["view_hash"] == w.digest(view), "Incorrect restricted view")
            decisions = {}
            for arm in "ABC":
                decision = w.choose(arm, view, recorded=capture["requests"][cid]["C"])
                require(decision["llm_attempts"] == 0, "Replay made a model call")
                require(w.semantic(decision) == w.semantic(row["arms"][arm]["decision"]), "Decision drift")
                if arm in "BC": decisions[arm] = {"decision": decision}
            route_rows.append({"case": case, "arms": decisions})
            series = panel.loc[panel.series_id == case["series_id"]].set_index("date").sales_units.sort_index()
            dates = pd.date_range(pd.Timestamp(case["as_of"]) + pd.Timedelta(days=1), periods=28)
            actual = series.reindex(dates).to_numpy(float)
            require(row["actual_dates"] == dates.strftime("%Y-%m-%d").tolist() and row["actual"] == actual.tolist(),
                    "Actual values/dates do not match bundled sales")
            for arm, item in row["arms"].items():
                numerical, metrics = item["numerical"], item["metrics"]
                if numerical is None or numerical.get("forecast") is None:
                    require(metrics is None, "Missing forecast received a score"); continue
                require(numerical["dates"] == row["actual_dates"], "Forecast dates differ")
                predicted = np.array(numerical["forecast"])
                require(len(predicted) == 28 and np.isfinite(predicted).all() and (predicted >= 0).all(), "Invalid prediction")
                errors = actual - predicted
                require(np.isclose(metrics["mae"], np.abs(errors).mean()) and
                        np.isclose(metrics["rmse"], np.sqrt(np.mean(errors**2))), "Final metric arithmetic failed")
                if case["scenario"] == "full":
                    train = series.loc[:case["as_of"]].tail(365).to_numpy(float)
                    scale = np.abs(train[7:] - train[:-7]).mean()
                    require(np.isclose(metrics["scale"], scale) and np.isclose(metrics["mase"], metrics["mae"]/scale),
                            "Training-only seasonal MASE failed")
                else:
                    require(metrics["mase"] is None, "Constructed case received a headline scaled score")
                for fit in numerical["fit_trace"]:
                    require(fit.get("train_end", w.ORIGIN) <= w.ORIGIN, "Fit includes post-origin data")
                for candidate in numerical.get("candidates", {}).values():
                    for fold in candidate["folds"]:
                        require(fold["train_end"] < fold["validation_start"] <= fold["validation_end"] <= w.ORIGIN,
                                "Nonchronological validation")
                checked_metrics += 1
        dev = record("development_run")
        require(dev["excluded_from_primary"] and dev["primary_capture_hash"] == capture["bundle_hash"], "Development cohort mixed")
        require(w.digest({k:v for k,v in dev.items() if k != "bundle_hash"}) == dev["bundle_hash"], "Development integrity failed")
        view = w.request_view(next(c for c in cases if c["case_id"] == dev["case_id"]), panel, source)
        decision = w.choose("C", view, recorded=dev["decision"], development=True)
        require(decision["source"] == "llm" and decision["executed_route"] == "regular", "No accepted genuine dispatch")
        require(w.semantic(w.execute(decision["executed_route"], view)) == w.semantic(dev["numerical"]), "Development numerical drift")

    stats = w.summary(result)
    require(stats == record("summary"), "Summary differs from executed results")
    require(all(s["common_n"] == s["scaled_n"] == 9 and s["excluded_scale_n"] == 0 for s in stats["full_history"].values()),
            "Common/scale denominator changed")
    audit = record("routing_audit")
    require(audit["capture_hash"] == capture["bundle_hash"] and len(audit["reviews"]) == 18, "Audit binding incomplete")
    for review in audit["reviews"]:
        saved = capture["requests"][review["case_id"]]["C"]
        require(review["response_hash"] == saved["response_hash"] and review["proposal"] == json.loads(saved["response"]["text"]),
                "Audited response changed")
    work = p.work_summary(result, capture, audit)
    require(work == record("work"), "Work accounting differs")
    routing_metrics = p.routing_metrics(result, capture, audit)
    require(routing_metrics == record("routing_metrics"), "Routing metric denominators differ")
    main_metrics = routing_metrics["cohorts"]["main"]
    require(main_metrics["json_schema_shape_valid"]["display"] == "16/16" and
            main_metrics["decision_contract_valid"]["display"] == "5/16" and
            main_metrics["policy_agreement_accepted_forecasts"]["display"] == "N/A" and
            main_metrics["unnecessary_clarifications"]["display"] == "5/16", "Routing interpretation changed")
    main_audit = next(x for x in work["audit"] if x["cohort"] == "main")
    require((main_audit["fallbacks"], main_audit["abstentions"], main_audit["accepted_forecasts"]) == (11, 5, 0), "Primary conclusion changed")
    require(all(row["arms"]["C"]["decision"]["source"] == "rule_fallback" for row in p.primary_rows(result)
                if row["arms"]["C"]["metrics"] is not None), "C forecast attribution changed")
    fixtures = record("failure_fixtures")
    require(len(fixtures) == 2 and all(f["provenance"] == "injected response" and
            f["decision"]["llm_attempts"] == 0 and f["decision"]["source"] == "rule_fallback" for f in fixtures), "Fixture provenance changed")

    assets = record("assets")
    require(len(assets) == 9, "Incorrect export count")
    images, current = [], None
    for line in prose(post).splitlines():
        if line.startswith("## "): current = line[3:]
        images += [(name, current) for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", line)]
    require(Counter(name for name, _ in images) == Counter(a["path"] for a in assets), "Post image inventory differs")
    code_order = {c.id:i for i,c in enumerate(codes)}
    require([code_order[a["cell_id"]] for a in assets] == sorted(code_order[a["cell_id"]] for a in assets), "Asset order differs")
    pngs = {c.id:[base64.b64decode(o["data"]["image/png"]) for o in c.outputs if "image/png" in o.get("data", {})] for c in codes}
    require(sum(map(len, pngs.values())) == 9, "Uninventoried notebook image")
    route_hash = w.digest(w.semantic({"rows": route_rows}))
    checked_table_cells = 0
    for kind, count in [("graph", 3), ("table", 6)]:
        require([a["number"] for a in assets if a["kind"] == kind] == list(range(1, count+1)), "Nonsequential numbering")
    for asset in assets:
        name = f"atsf02-{asset['kind']}{asset['number']}.png"
        require(asset["filename"] == name and asset["path"] == f"graphs/{name}", "Incorrect filename")
        path = w.ROOT / asset["path"]
        require(w.file_hash(path) == asset["sha256"], "PNG checksum failed")
        require(asset["section"] == sections[asset["cell_id"]] and (asset["path"], asset["section"]) in images,
                "Asset section correspondence failed")
        expected_source = w.digest(protocol) if asset["cell_id"] in {"routing-diagram", "visible-histories", "case-table", "workflow-table"} else route_hash if asset["cell_id"] == "route-table" else w.digest(w.semantic(result))
        require(asset["source_result_hash"] == expected_source, "Stale asset source")
        require(len(pngs[asset["cell_id"]]) == 1 and hashlib.sha256(pngs[asset["cell_id"]][0]).hexdigest() == asset["sha256"],
                "Notebook does not display exported image bytes")
        with Image.open(path) as image:
            require(image.format == "PNG" and image.width > 1000 and image.height > 200, "Invalid image dimensions")
            require("dpi" in image.info and all(abs(d-180) < .1 for d in image.info["dpi"]), "Incorrect resolution")
            image.verify()
        require(asset["dpi"] == 180 and asset["figsize"][0] == 13.0 and asset["font"], "Configured style metadata failed")
        if asset["kind"] == "table":
            require(asset["free_text_artists"] == 0, "Table has title, footer or free text")
            require(w.file_hash(w.ROOT / asset["table_sidecar"]) == asset["table_sidecar_sha256"], "Table sidecar changed")
            for block in read(asset["table_sidecar"]):
                for raw, rendered, types in zip(block["raw"], block["display"], block["types"], strict=True):
                    require(len(raw) == len(rendered) == len(types) == len(block["headers"]), "Malformed table row")
                    for value, display, typename in zip(raw, rendered, types, strict=True):
                        require(type(value).__name__ == typename, "Typed source mismatch")
                        expected = f"{value:.2f}" if isinstance(value, float) else "N/A" if value is None else str(value)
                        require(display == expected, "Float precision / integer formatting mismatch")
                        if isinstance(value, float): require(re.fullmatch(r"-?\d+\.\d{2}", display), "Float lacks two decimals")
                        checked_table_cells += 1

    claims = record("post_claims")
    require(claims["post_sha256"] == w.file_hash(post_path) and claims["capture_hash"] == capture["bundle_hash"], "Editorial review stale")
    for paragraph in claims["paragraphs"]:
        require(paragraph["text"] in post and paragraph["evidence"], "Reviewed prose changed")
        for reference in paragraph["evidence"]:
            reference_value(reference)
    dispatch = next(d for d in work["dispatch"] if d["cohort"] == "main" and d["arm"] == "C")
    numerical = {d["arm"]:d for d in work["numerical"]}
    require(sum(len(row["arms"]["B"]["numerical"]["fit_trace"]) for row in result["rows"]
                if row["case"]["cohort"] == "main" and row["case"]["scenario"] == "full") == 96,
            "Matched full-history fit count changed")
    expected_fragments = [
        f"**{dispatch['seconds']:.2f} seconds**", f"**{dispatch['input_tokens']:,} input tokens and {dispatch['output_tokens']:,} output tokens**",
        f"**{dispatch['median_seconds']:.2f} seconds**", f"**{numerical['B']['fits']} fit attempts for B**", f"**{numerical['D']['fits']} for D**",
        "**" + ", ".join(f"{stats['full_history'][a]['macro_mase']:.2f} for {a}" for a in "ABC") + f", and {stats['full_history']['D']['macro_mase']:.2f} for D**",
        "eleven forecasting proposals were rejected and fell back to rules; five responses asked unnecessary clarification questions",
        "Two changes were made together", "every delivered primary C forecast came from the rule-selected workflow",
        "three as supported, eleven as contradicted, one as unsupported and one as ambiguous",
    ]
    for fragment in expected_fragments: require(fragment in post, f"Numerical claim or qualification changed: {fragment}")
    promo = (w.ROOT / "docs/module-2-promo.md").read_text()
    seo = promo.split("## SEO description\n\n")[1].split("\n\n")[0]
    require(50 <= len(seo) <= 160 and f"Character count: {len(seo)}" in promo and "## LinkedIn promo post" in promo, "Promo incomplete")
    local_links(promo, w.ROOT / "docs")
    readme = (w.ROOT / "README.md").read_text()
    local_links(readme.split("2. A02:")[1].split("3. A03:")[0], w.ROOT)
    local_links(readme.split("## Run Module 2")[1].split("## Earlier module runners")[0], w.ROOT)
    visual = record("visual_qa")
    require(visual["status"] == "passed" and visual["widths"] == ["native", 760], "Visual review missing")
    require(visual["asset_hashes"] == {a["filename"]:a["sha256"] for a in assets}, "Visual review stale")
    replay = record("replay_verification")
    require(replay["live_calls_in_replay"] == 0 and replay["development_genuine_dispatch"] and replay["asset_count"] == 9, "Notebook completion failed")
    verification = {"status":"passed", "capture_hash":capture["bundle_hash"], "new_model_calls":0,
                    "executed_code_cells":len(codes), "matching_substantive_sections":len(headings),
                    "checked_forecast_metrics":checked_metrics, "graphs":3, "tables":6,
                    "checked_table_cells":checked_table_cells, "reviewed_empirical_paragraphs":len(claims["paragraphs"]),
                    "seo_characters":len(seo), "notebook_sha256":w.file_hash(notebook_path), "post_sha256":w.file_hash(post_path),
                    "scope":"Exact replay, independent final-score arithmetic, asset integrity/formatting, evidence mappings and links; semantic and visual judgments are explicitly recorded author reviews."}
    w.write_json(folder / "publication_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
