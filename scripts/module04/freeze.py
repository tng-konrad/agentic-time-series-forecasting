from pathlib import Path
import json
from development import namespace

ROOT = Path(__file__).resolve().parents[2]
n = namespace()
cfg = n["CFG"]
checks = n["load_json"](cfg.evidence_dir/"development/checks.json")
assert all(c["status"] == "passed" for c in checks["checks"])
pilot = n["load_json"](cfg.evidence_dir/"development/numerical-pilot.json")
identity = n["server_identity"]()
protocol = {"version": "a04-retail-v1", "episode_number": cfg.episode_number,
            "core_sha256": cfg.core_sha256, "contract": n["contract_settings"](),
            "model_identity": identity, "environment": n["environment_record"](),
            "data_manifest": n["load_json"](cfg.data_dir/"manifest.json"),
            "source_sha256": n["file_hash"](ROOT/"docs/a04-Plan-and-Execute Forecasting Agents.docx"),
            "source_adaptation": "planner/executor/monitor/replanner; replace source toy test-derived predictions and high-order MASE difference",
            "case_order": list(range(1,11)), "forecast_unit": "store daily total sales over50 items",
            "final_dates": ["2017-10-03", "2017-12-31"], "prior_exploration": "historical competition data and m10 already explored",
            "validation": "expanding prefixes1466/1556/1646,90-day nonoverlapping targets",
            "selection": "mean fold RMSE, exact tie catalog order; only complete converged finite candidates eligible",
            "comparators": ["weekly seasonal naive", "counterfactual initial selection without repair", "shared repaired agent plan", "fixed log BSM vs baseline"],
            "aggregation": "pooled equal-length daily RMSE/MAE; equal-store MASE/coverage/width/interval score; coverage denominator10",
            "undefined": "zero seasonal MASE scale -> null with reason, unscaled errors retained; missing forecasts remain in coverage",
            "failure_policy": "zero retries; invalid initial -> baseline fallback; invalid revision -> initial selection fallback; unresolved diagnostic may deliver labeled outputs",
            "admission": "900s workflow, check every tool and fit; running fit may overrun; no-fit safety delivery exception; fixed comparators afterward with separate timing",
            "audit_rubric": {"unit": "atomic factual proposition in all public reasons", "categories": ["supported","contradicted","unsupported","ambiguous"],
                             "exclusions": "pure proposals counted separately", "malformed": "unassessable response counted", "reviewer": "author Codex", "timing": "before opening final outcomes"},
            "replay_scope": "analysis-only: verifies identities and regenerates frozen analysis/visuals; does not rerun numerical tools or model decisions",
            "replay_equality": "exact checksum/typed-value equality; no numeric recomputation tolerance",
            "development_sha256": n["digest"](checks), "synthetic_pilot_sha256": n["digest"](pilot),
            "frozen_at": n["datetime"].now(n["ZoneInfo"](cfg.timezone)).isoformat()}
path = cfg.evidence_dir/"protocol.json"
if path.exists():
    raise FileExistsError("Protocol is frozen; use a new version/path for semantic changes")
n["save_json"](path, protocol)
n["save_json"](cfg.evidence_dir/"protocol-freeze.json", {"version": protocol["version"], "sha256": n["digest"](protocol), "file_sha256": n["file_hash"](path)})
print("FROZEN", n["digest"](protocol))
