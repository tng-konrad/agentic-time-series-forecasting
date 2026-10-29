"""Recompute selected workflows, persist frozen predictions, then open outcomes."""
import json
from agentic_ts import module02_workflow as w


def main():
    folder = w.ROOT / "examples/module02"
    if (folder / "frozen_forecasts.json").exists():
        raise FileExistsError("Preserving author numerical run; reader replay is separate")
    panel, source, cases = w.load_inputs()
    capture = json.loads((folder / "recorded_run.json").read_text())
    frozen = w.freeze_run(panel, source, cases, capture)
    w.write_json(folder / "frozen_forecasts.json", frozen.data, exclusive=True)
    scored = w.score_run(frozen, panel)
    w.write_json(folder / "numerical_results.json", scored, exclusive=True)
    w.write_json(folder / "failure_fixtures.json", w.run_fixtures(panel, source, cases), exclusive=True)
    w.write_json(folder / "summary.json", w.summary(scored))
    print(json.dumps(w.summary(scored)["full_history"], indent=2))


if __name__ == "__main__": main()
