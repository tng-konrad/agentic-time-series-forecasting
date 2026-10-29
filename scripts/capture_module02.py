"""Capture all first-attempt routing outputs; preserve failures and original archive."""
from dataclasses import asdict
import json
import platform
from urllib.request import urlopen
from agentic_ts import module02_workflow as w


def save(path, record):
    record.pop("bundle_hash", None)
    record["bundle_hash"] = w.digest(record)
    temporary = path.with_suffix(".tmp")
    w.write_json(temporary, record)
    temporary.replace(path)


def main():
    path = w.ROOT / "examples/module02/recorded_run.json"
    if path.exists(): raise FileExistsError("Preserving original capture")
    panel, source, cases = w.load_inputs()
    protocol = json.loads((path.parent / "protocol.json").read_text())
    if protocol["policy"] != w.POLICY or protocol["system"] != w.SYSTEM or protocol["model_config"] != asdict(w.CONFIG):
        raise ValueError("Protocol changed before capture")
    def get(endpoint):
        with urlopen(w.CONFIG.base_url + endpoint, timeout=20) as response:
            return json.load(response)
    identity = next(m for m in get("/api/tags")["models"] if m["name"] == w.CONFIG.model)
    record = {"created_at": w.timestamp(), "session_id": "module02-primary-20261026",
              "contract_version": w.CONTRACT, "model_config": asdict(w.CONFIG), "model_identity": identity,
              "server": get("/api/version"), "initial_loaded": get("/api/ps"),
              "machine": {"platform": platform.platform(), "python": platform.python_version()},
              "protocol_hash": w.digest(protocol), "source_csv_sha256": source["csv_sha256"],
              "requests": {}, "complete": False}
    w.write_json(path, record, exclusive=True)
    save(path, record)
    for case in cases:
        view = w.request_view(case, panel, source)
        arms = {}
        for arm in ("A", "B", "C"):
            arms[arm] = w.choose(arm, view, mode="live")
        record["requests"][case["case_id"]] = arms
        save(path, record)
        c = arms["C"]
        print(case["case_id"], c["status"], c["proposed_route"], "->", c["executed_route"], f'{c["seconds"]:.2f}s', flush=True)
    record["complete"] = True
    save(path, record)
    print("Preserved", sum(v["C"]["llm_attempts"] for v in record["requests"].values()), "genuine attempts")


if __name__ == "__main__": main()
