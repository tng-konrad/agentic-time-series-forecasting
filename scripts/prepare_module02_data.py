"""Create the fixed request manifest without duplicating or modifying M5 sales."""
from dataclasses import asdict
from agentic_ts import module02_workflow as w


def main():
    panel, source = w.numerical.load_routing_panel(w.ROOT / "data/a02")
    cases = w.make_cases(source)
    folder = w.ROOT / "data/module02"
    manifest = {"created_at": w.timestamp(), "contract": w.CONTRACT, "source_path": "data/a02/m5_routing_panel.csv",
                "source_csv_sha256": source["csv_sha256"], "cases_sha256": w.digest(cases),
                "main_n": 16, "ambiguity_n": 2, "gate_n": 2,
                "source_manifest": source, "note": "Short and zero-history cases are controlled information restrictions, not real launches."}
    # Validate every view now, while the final values are never supplied to a model.
    for case in cases:
        check = w.gate(w.request_view(case, panel, source))
        if case["cohort"] != "gate" and check["status"] != "supported":
            raise ValueError((case["case_id"], check))
    w.write_json(folder / "cases.json", cases, exclusive=True)
    w.write_json(folder / "manifest.json", manifest, exclusive=True)
    w.write_json(w.ROOT / "examples/module02/protocol.json", {
        "created_at": w.timestamp(), "contract": w.CONTRACT, "policy": w.POLICY, "system": w.SYSTEM,
        "model_config": asdict(w.CONFIG), "numerical_config": asdict(w.NUMERICAL),
        "backend_options": "Ollama adapter defaults; keep_alive=10m; no thinking option supplied",
        "case_manifest_hash": w.digest(cases), "case_order": [c["case_id"] for c in cases],
        "primary_cohort": [c["case_id"] for c in cases if c["cohort"] == "main"],
        "representative_cases": ["M02-F01", "M02-F05", "M02-S01", "M02-N01"],
        "origin": w.ORIGIN, "target_start": "2016-03-28", "target_end": "2016-04-24",
        "selection": "Full: mean RMSE on three rolling 28-day folds; short: one 28-day fold; new: fixed donor prior",
        "first_attempts_only": True, "source_csv_sha256": source["csv_sha256"],
    }, exclusive=True)
    print("Prepared 16 main cases, 2 ambiguity cases and 2 gate cases; protocol frozen.")


if __name__ == "__main__": main()
