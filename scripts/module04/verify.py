from pathlib import Path
import base64
import hashlib
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[2]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
digest = lambda v: hashlib.sha256(canonical(v).encode()).hexdigest()
evidence = ROOT/"examples/module04"
run_id = read(evidence/"latest.json")["run_id"]
run = evidence/"runs"/run_id
capture, freeze, results, audit = [read(run/f"{f}.json") for f in ["capture", "freeze", "results", "audit"]]
manifest = read(evidence/"asset-manifest.json")
protocol = read(evidence/"protocol.json")
nb = nbformat.read(ROOT/"a04-plan-and-execute.ipynb", as_version=4)
post = (ROOT/"a04-post.md").read_text()
outline = (ROOT/"docs/module-4-post-outline.md").read_text()
checks = []
def check(label, condition):
    if not condition:
        raise AssertionError(label)
    checks.append(label)

check("Capture/forecast freeze and author audit binding", sha(run/"capture.json") == freeze["capture_sha256"] == audit["capture_sha256"]
      and digest(capture["arms"]) == freeze["forecasts_sha256"] == audit["forecast_sha256"])
check("Protocol unchanged", digest(protocol) == freeze["protocol_sha256"] == read(evidence/"protocol-freeze.json")["sha256"])
check("Complete ten-store four-arm coverage accounting", len(results["records"]) == 40 and len(results["summaries"]) == 4
      and all(r["denominator"] == 10 for r in results["summaries"]))
paired = {arm: {r["store"]: r for r in results["records"] if r["arm"] == arm} for arm in capture["arms"]}
check("All forty outputs delivered", all(r["delivery"] == "delivered" for r in results["records"]))
check("Nine paired wins and store3 loss", sum(paired["agent_plan"][s]["rmse"] < paired["seasonal_naive"][s]["rmse"] for s in range(1,11)) == 9
      and paired["agent_plan"][3]["rmse"] > paired["seasonal_naive"][3]["rmse"])
check("Ten initial and five repeated failures", sum(v["status"] == "failed_check" for v in capture["initial_monitors"].values()) == 10
      and sum(v["status"] == "failed_check" for v in capture["revised_monitors"].values()) == 5)
check("Eight SARIMAX and two BSM selections", sum(v["delivered"] == "sarimax" for v in capture["arms"]["agent_plan"].values()) == 8
      and sum(v["delivered"] == "bsm" for v in capture["arms"]["agent_plan"].values()) == 2)
check("Fixed eligibility and losses retained", sum(v["bsm"]["status"] == "failed" for v in capture["fixed_catalogs"].values()) == 2
      and all(v["delivered"] == "seasonal_naive" for v in capture["arms"]["fixed_bsm"].values()))
check("Work and token claims", len(capture["work"]["fits"]) == 120 and len(capture["fixed_work"]["fits"]) == 38
      and sum(v["status"] == "failed" for v in capture["fixed_work"]["fits"]) == 2
      and sum(r["raw_response"].get("eval_count",0) for r in capture["requests"]) == 674)
check("Audit counts and exclusions", audit["counts"] == {"supported":4,"contradicted":0,"unsupported":0,"ambiguous":2,
      "factual_denominator":6,"proposals_excluded":1} and audit["unassessable_responses"] == 0)
check("All shared requests retained within budget", len(capture["requests"]) <= 2 and all(r["status"] in ["accepted", "failed"] for r in capture["requests"]))
check("Each actual fit has terminal status", all(r["status"] in ["completed", "failed"] for r in capture["work"]["fits"] + capture["fixed_work"]["fits"]))
check("No final targets in planning requests", all("2017-12-31" not in canonical(r["semantic_input"]) for r in capture["requests"]))
check("Frozen code identity", sha(ROOT/"scripts/module04/core.py") == protocol["core_sha256"])
heads = lambda text: re.findall(r"^## (.+)$", text, re.M)
notebook_heads = heads("\n".join(c.source for c in nb.cells if c.cell_type == "markdown"))
check("Setup and Utils first", notebook_heads[:2] == ["Setup", "Utils"])
check("Matching substantive section hierarchy", notebook_heads[2:] == heads(post) == heads(outline)
      and notebook_heads[-1] == "Closing time")
code_cells = [c for c in nb.cells if c.cell_type == "code"]
check("Fresh-kernel Run All has complete error-free outputs", all(c.execution_count is not None for c in code_cells)
      and not any(o.output_type == "error" for c in code_cells for o in c.outputs))
import_cells = [c for c in code_cells if re.search(r"^(?:import |from .+ import )", c.source, re.M)]
check("One imports cell and inline project implementation", len(import_cells) == 1
      and not any(re.search(r"^from (agentic_ts|scripts)\b", c.source, re.M) for c in code_cells))
check("Nine assets in shared display order", [a["visual_number"] for a in manifest["assets"]] == [f"{i:02d}" for i in range(1,10)])
inline_png = [base64.b64decode(o.data["image/png"]) for c in code_cells for o in c.outputs
              if o.output_type in ["display_data", "execute_result"] and "image/png" in o.get("data", {})]
for asset in manifest["assets"]:
    path, sidecar = ROOT/asset["filename"], ROOT/asset["sidecar"]
    check(f"Asset {asset['visual_number']} naming, hash and sidecar", bool(re.fullmatch(r"graphs/atsf04_(graph|table)_\d{2}\.png", asset["filename"]))
          and sha(path) == asset["png_sha256"] and sha(sidecar) == asset["sidecar_sha256"])
    check(f"Asset {asset['visual_number']} exact inline bytes", path.read_bytes() in inline_png)
    check(f"Asset {asset['visual_number']} post and outline placement", asset["filename"] in post and asset["filename"] in outline
          and asset["section"] in post and asset["caption"] in post)
    review = asset.get("visual_review")
    check(f"Asset {asset['visual_number']} current visual signoff", isinstance(review, dict)
          and review["status"] == "passed" and review["reviewed_png_sha256"] == sha(path))
    raw = read(sidecar)
    if asset["type"] == "table":
        check(f"Asset {asset['visual_number']} display strings match typed rows", len(raw["rows"]) == len(raw["displayed_strings"])
              and all([re.sub(r"\s+", "", str(a)) == re.sub(r"\s+", "", b) for a,b in zip(row,shown)] == [True]*len(row)
                      for row,shown in zip(raw["rows"],raw["displayed_strings"])))
for match in re.findall(r"\]\(([^)]+)\)", post):
    if not match.startswith(("http:", "https:", "#")):
        check(f"Local post link resolves: {match}", (ROOT/match.replace("%20", " ")).exists())
check("Standard publication footer", "United States of Banan is a reader-supported publication." in post and "[ ✓ Subscribed ]" in post)
check("Outline readings are concrete", "use the verified interpretation" not in outline)
claims = read(evidence/"claims.json")
for claim in claims["claims"]:
    if claim["kind"] == "result_field":
        value = results
        for part in claim["path"]:
            value = value[part]
        check("Claim: " + claim["label"], value == claim["value"])
    elif claim["kind"] == "capture_field":
        value = capture
        for part in claim["path"]:
            value = value[part]
        check("Claim: " + claim["label"], value == claim["value"])
record = {"status": "passed", "run_id": run_id, "checks": checks,
          "notebook_code_sha256": digest([c.source for c in code_cells]),
          "notebook_output_sha256": digest([c.outputs for c in code_cells]),
          "post_sha256": sha(ROOT/"a04-post.md"), "asset_manifest_sha256": sha(evidence/"asset-manifest.json"),
          "replay_scope": "fresh-kernel analysis-only replay with sockets blocked; no numerical tool or decision recomputation"}
(evidence/"validation.json").write_text(json.dumps(record, indent=2))
print("PASS", len(checks), "checks")
