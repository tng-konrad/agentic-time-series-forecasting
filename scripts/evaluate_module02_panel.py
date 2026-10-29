"""Author run: freeze forecasts before scores; preserve original evidence."""
import argparse
import time
from agentic_ts import module02_panel as p
from agentic_ts import module02_workflow as w


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot", action="store_true"); args = parser.parse_args()
    data = p.load_data(); protocol = p.record("protocol.json")
    bank = p.FitBank(); start = time.perf_counter()
    if args.pilot:
        cases = [c for c in data.cases if c["role"] == "development"][:4]
        p.freeze_cases(data, cases, bank)
        record = {"scope":"Four development cases; no final outcomes scored", "cases":4,
                  "seconds":time.perf_counter()-start,"unique_fits":len(bank.fits),"bank_lookups":bank.lookups}
        p.write(p.EXAMPLES / "pilot.json",record,exclusive=True); print(record); return
    if (p.EXAMPLES / "frozen_forecasts.json.gz").exists(): raise FileExistsError("Preserving author forecasts")
    dev_cases = [c for c in data.cases if c["role"] == "development"]
    frozen_dev = p.freeze_cases(data,dev_cases,bank,progress=lambda s:print(s,flush=True))
    p.write(p.EXAMPLES / "development_frozen.json.gz",frozen_dev.data,exclusive=True)
    development = p.score(frozen_dev,data); fixed = p.select_fixed(development)
    p.write(p.EXAMPLES / "development_results.json.gz",development,exclusive=True)
    p.write(p.EXAMPLES / "fixed_workflow.json",fixed,exclusive=True)
    print("Fixed comparator:",fixed,flush=True)
    main_cases = [c for c in data.cases if c["role"] == "main"]
    frozen = p.freeze_cases(data,main_cases,bank,fixed["selected_arm"],progress=lambda s:print(s,flush=True))
    p.write(p.EXAMPLES / "frozen_forecasts.json.gz",frozen.data,exclusive=True)
    p.write(p.EXAMPLES / "fit_bank.json.gz",bank.snapshot(),exclusive=True)
    result = p.score(frozen,data)
    p.write(p.EXAMPLES / "results.json.gz",result,exclusive=True)
    summary = p.summarize(result); p.write(p.EXAMPLES / "summary.json",summary,exclusive=True)
    timings = p.measure_uncached(data,p.timing_cases(data.cases,protocol))
    p.write(p.EXAMPLES / "timings.json",timings,exclusive=True)
    p.write(p.EXAMPLES / "work.json",p.work_record(bank,result,timings),exclusive=True)
    p.write(p.EXAMPLES / "author_run.json",{"protocol_hash":w.digest(protocol),"frozen_semantic_hash":w.digest(w.semantic(frozen.data)),
             "result_semantic_hash":w.digest(w.semantic(result)),"bank_semantic_hash":w.digest(w.semantic(bank.snapshot())),
             "seconds":time.perf_counter()-start,"new_llm_calls":0,"created_at":w.timestamp()},exclusive=True)
    print({a:s["macro_mase"] for a,s in summary["arms"].items()},flush=True)


if __name__ == "__main__": main()
