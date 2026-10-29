"""One separately labeled development capture; never replaces primary outcomes."""
import json
from agentic_ts import module02_workflow as w


def main():
    path = w.ROOT / "examples/module02/development_run.json"
    if path.exists(): raise FileExistsError("Preserving development capture")
    panel, source, cases = w.load_inputs()
    case = next(c for c in cases if c["case_id"] == "M02-F01")
    view = w.request_view(case, panel, source)
    primary = json.loads((path.parent / "recorded_run.json").read_text())
    result = {"created_at": w.timestamp(), "case_id": case["case_id"], "excluded_from_primary": True,
              "model_identity": primary["model_identity"], "primary_capture_hash": primary["bundle_hash"],
              "change": "Clarify established-product launch-date semantics and use equivalent type-union nullable schema; one first attempt, not a controlled ablation.",
              "decision": w.choose("C", view, mode="live", development=True)}
    decision = result["decision"]
    result["numerical"] = w.execute(decision["executed_route"], view) if decision["executed_route"] else None
    result["bundle_hash"] = w.digest(result)
    w.write_json(path, result, exclusive=True)
    print(decision["status"], decision["source"], decision["proposed_route"], decision.get("failure"))


if __name__ == "__main__": main()
