"""Point-in-time panel evaluation of Module 2's existing numerical routing policy."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from . import routing_forecasting as n
from . import module02_workflow as w
from .forecasting import FrozenJSON, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/module02_panel"
EXAMPLES = ROOT / "examples/module02_panel"
VERSION = "module02-panel-v1"
CFG = n.ForecastConfig()
STORES = ("CA_1", "TX_1", "WI_1")
DEV_DAYS = (1661, 1689, 1717)
MAIN_DAYS = (1745, 1773, 1801, 1829, 1857, 1885)
ARMS = ("A", "B", "R", "I", "F", "D")
HEADINGS = ["Why test routing beyond one store and one month?",
            "Build a panel without looking at future outcomes",
            "Separate workflow selection from forecast evaluation",
            "Repeat routing across products, stores and origins",
            "Where does routing help, and where does it hurt?",
            "How much work does routing save?",
            "What does the larger experiment establish?"]
FAMILIES = {"R": "regular", "I": "intermittent"}


def write(path, value, exclusive=False):
    """Deterministic gzip headers; original author records use exclusive creation."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"Preserving original record: {path}")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw); temporary.replace(path)


def read(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def record(name, reader=False):
    return read(EXAMPLES / (("reader_" if reader else "") + name))


@dataclass
class StudyData:
    series: dict
    items: dict
    cases: list
    manifest: dict


def load_data(directory=DATA):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    for name, expected in manifest["files"].items():
        if w.file_hash(directory / name) != expected:
            raise ValueError(f"Panel source checksum mismatch: {name}")
    frame = pd.read_csv(directory / "sales.csv.gz", parse_dates=["date"])
    items = {m["series_id"]: m for m in read(directory / "items.json")}
    cases = read(directory / "cases.json")
    if set(frame.columns) != {"series_id", "date", "sales_units"} or len(frame) != 210300 or len(items) != 300:
        raise ValueError("Expected the fixed 300-series, 210300-row panel")
    if set(frame.series_id) != set(items) or frame.isna().any().any():
        raise ValueError("Panel membership or missing-value error")
    values = frame.sales_units.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Sales must be nonnegative finite integer counts")
    expected_dates = pd.date_range("2014-05-25", "2016-04-24")
    series = {}
    for sid, part in frame.groupby("series_id", sort=True):
        part = part.sort_values("date")
        if not pd.DatetimeIndex(part.date).equals(expected_dates):
            raise ValueError(f"Incomplete or duplicate panel calendar: {sid}")
        series[sid] = pd.Series(part.sales_units.to_numpy(float), index=pd.DatetimeIndex(part.date))
    if Counter(c["role"] for c in cases) != {"development": 600, "main": 1800}:
        raise ValueError("Incorrect experiment grid")
    if any(c["store"] == "WI_1" for c in cases if c["role"] == "development"):
        raise ValueError("WI_1 cannot select the fixed workflow")
    if len({c["case_id"] for c in cases}) != 2400:
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case["series_id"] not in items or any(case[k] != items[case["series_id"]][k] for k in ("item_id", "store", "department")):
            raise ValueError("Case metadata mismatch")
        h = history(StudyData(series, items, [], manifest), case)
        if len(h) != CFG.required_history or h.index[-1] != pd.Timestamp(case["as_of"]):
            raise ValueError("Incorrect history restriction")
    return StudyData(series, items, cases, manifest)


def history(data, case):
    """Only this boundary accesses the full source series before final scoring."""
    origin = pd.Timestamp(case["as_of"])
    start = origin - pd.Timedelta(days=CFG.required_history - 1)
    h = data.series[case["series_id"]].loc[start:origin].copy()
    if not h.index.equals(pd.date_range(start, origin)):
        raise ValueError("Incomplete permitted history")
    return h


def view(data, case):
    h = history(data, case)
    result = {"request": {"case_id": case["case_id"], "series_id": case["series_id"], "as_of": case["as_of"],
                           "visible_start": h.index[0].date().isoformat(), "target": "observed_sales", "horizon": 28,
                           "frequency": "D", "units": "item units", "lifecycle": "established", "launch_date": None,
                           "analogy_authorized": False, "donor_ids": []},
              "history": w.series_payload(h), "item": {k: case[k] for k in ("department", "store")}, "donors": {}}
    if w.gate(result)["status"] != "supported":
        raise ValueError("Panel view failed the shared request gate")
    return result


class FitBank:
    """Memoize a pure numerical fit, never an outcome-informed routing decision."""
    def __init__(self):
        self.fits = {}; self.lookups = 0; self.hits = 0
        self.versions = n.package_versions()

    def key(self, train, model, config=CFG):
        return digest({"version": VERSION, "training": w.series_payload(train), "model": model,
                       "config": asdict(config), "versions": self.versions})

    def forecast(self, train, model, config=CFG):
        key = self.key(train, model, config)
        self.lookups += 1
        if key not in self.fits:
            self.fits[key] = {"key": key, "model": model, "fit": n.fit_forecast(train, model, config)}
        else:
            self.hits += 1
        return deepcopy(self.fits[key]["fit"])

    def snapshot(self):
        return {"version": VERSION, "versions": self.versions, "fits": self.fits,
                "lookups": self.lookups, "cache_hits": self.hits, "actual_fit_attempts": len(self.fits)}


def reconstruct_path(h, route, candidates, finals, fit_keys):
    models = n.CATALOG if route == "all_candidates_diagnostic" else n.FAMILIES[route]
    subset = {m: deepcopy(candidates[m]) for m in models}
    path = n._finish_path(h, route, subset, [], CFG, n.fit_forecast, reusable_final=finals)
    logical = [fit_keys[m][i] for m in models for i in range(CFG.n_folds)]
    logical += [fit_keys[m][-1] for m in path["final_fits"]]
    return {k: deepcopy(path[k]) for k in ("route", "status", "selected", "delivered", "forecast", "dates")} | {
        "logical_fit_keys": logical, "logical_fit_attempts": len(logical)}


def freeze_case(data, case, bank, fixed=None):
    request = view(data, case)
    h = w.as_series(request["history"])
    evidence = w.profile(request)
    route = w.rule_route(evidence)
    # The router's decision is fixed before it can see candidate comparisons.
    if route not in ("regular", "intermittent"):
        raise ValueError("Unexpected full-history routing outcome")
    windows = list(n.validation_windows(h, CFG))
    fit_keys = {m: [bank.key(train, m) for train, _ in windows] + [bank.key(h.tail(365), m)] for m in n.CATALOG}
    candidates = n.backtest_candidates(h, n.CATALOG, CFG, bank.forecast)
    finals = {m: bank.forecast(h.tail(365), m) for m in n.CATALOG}
    base = finals["seasonal_naive"]
    arms = {"A": {"route": "seasonal_naive", "status": "ok" if base["status"] == "ok" else "numerical_failure",
                   "selected": "seasonal_naive", "delivered": "seasonal_naive" if base["status"] == "ok" else None,
                   "forecast": base.get("forecast"), "dates": base.get("dates"),
                   "logical_fit_keys": [fit_keys["seasonal_naive"][-1]], "logical_fit_attempts": 1}}
    for arm, family in [("B", route), ("R", "regular"), ("I", "intermittent"), ("D", "all_candidates_diagnostic")]:
        arms[arm] = reconstruct_path(h, family, candidates, finals, fit_keys)
    if fixed is not None:
        if fixed not in ("R", "I"):
            raise ValueError("Fixed comparator must be selected from R/I on development data")
        arms["F"] = deepcopy(arms[fixed])
    return {"case": deepcopy(case), "view_hash": digest(request),
            "evidence": evidence, "policy_route": route, "candidates": candidates,
            "fit_keys": fit_keys, "arms": arms}


def freeze_cases(data, cases, bank, fixed=None, progress=None):
    rows = []
    for i, case in enumerate(cases, 1):
        rows.append(freeze_case(data, case, bank, fixed))
        if progress and (i % 50 == 0 or i == len(cases)):
            progress(f"{case['role']} {case['as_of']}: {i}/{len(cases)} cases; {len(bank.fits)} unique fits")
    return FrozenJSON.of({"version": VERSION, "rows": rows})


def score(frozen, data):
    result = frozen.data
    for row in result["rows"]:
        case = row["case"]
        dates = pd.date_range(pd.Timestamp(case["as_of"]) + pd.Timedelta(days=1), periods=28)
        actual = data.series[case["series_id"]].reindex(dates).to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError("Missing outcome dates")
        train = history(data, case).tail(365)
        for arm in row["arms"].values():
            arm["metrics"] = None
            if arm["forecast"] is not None:
                if arm["dates"] != dates.strftime("%Y-%m-%d").tolist():
                    raise ValueError("Prediction/outcome date mismatch")
                arm["metrics"] = n.score_forecast(actual, arm["forecast"], train, CFG)
        row["actual_dates"] = dates.strftime("%Y-%m-%d").tolist()
        row["actual"] = actual.tolist()
    result["frozen_semantic_hash"] = digest(w.semantic(frozen.data))
    result["scored_after_freeze"] = True
    return result


def common_rows(rows, arms=ARMS):
    common = [r for r in rows if all(r["arms"][a]["metrics"] is not None for a in arms)]
    scaled = [r for r in common if all(r["arms"][a]["metrics"]["mase"] is not None for a in arms)]
    return common, scaled


def select_fixed(development):
    rows = development["rows"]
    if not rows or any(r["case"]["role"] != "development" or r["case"]["store"] not in ("CA_1", "TX_1") for r in rows):
        raise ValueError("Fixed selection accepts CA_1/TX_1 development cases only")
    _, scaled = common_rows(rows, ("R", "I"))
    if not scaled:
        raise ValueError("No common positive-scale development evidence")
    means = {a: float(np.mean([r["arms"][a]["metrics"]["mase"] for r in scaled])) for a in ("R", "I")}
    selected = min(("R", "I"), key=lambda a: (means[a], ("R", "I").index(a)))
    return {"version": VERSION, "selected_arm": selected, "family": FAMILIES[selected],
            "common_scaled_n": len(scaled), "eligible_n": len(rows), "macro_mase": means,
            "development_result_hash": digest(w.semantic(development)), "tie_order": ["R", "I"]}


def arm_summary(rows, arms=ARMS):
    common, scaled = common_rows(rows, arms)
    output = {}
    for arm in arms:
        delivered = [r for r in rows if r["arms"][arm]["metrics"] is not None]
        output[arm] = {"eligible_n": len(rows), "delivered_n": len(delivered), "common_n": len(common),
                       "scaled_n": len(scaled), "excluded_scale_n": len(common)-len(scaled),
                       "macro_mase": float(np.mean([r["arms"][arm]["metrics"]["mase"] for r in scaled])) if scaled else None,
                       "delivered_macro_mae": float(np.mean([r["arms"][arm]["metrics"]["mae"] for r in delivered])) if delivered else None,
                       "numerical_fallback_n": sum(r["arms"][arm]["status"] == "model_fallback" for r in rows),
                       "logical_fit_attempts": sum(r["arms"][arm]["logical_fit_attempts"] for r in rows)}
    return output


def contrast(rows, other):
    common, scaled = common_rows(rows)
    def delta(group, metric):
        return float(np.mean([r["arms"]["B"]["metrics"][metric]-r["arms"][other]["metrics"][metric] for r in group])) if group else None
    return {"eligible_n": len(rows), "common_n": len(common), "scaled_n": len(scaled),
            "delta_mase": delta(scaled, "mase"), "delta_mae": delta(common, "mae"),
            "routing_wins": sum(r["arms"]["B"]["metrics"]["mae"] < r["arms"][other]["metrics"]["mae"] for r in common),
            "routing_losses": sum(r["arms"]["B"]["metrics"]["mae"] > r["arms"][other]["metrics"]["mae"] for r in common),
            "ties": sum(r["arms"]["B"]["metrics"]["mae"] == r["arms"][other]["metrics"]["mae"] for r in common)}


def zero_band(row):
    e = row["evidence"]["history.current_window"]
    ratio = e["zero_count"] / e["observed_count"]
    return min(int(ratio * 4), 3)


def bootstrap_items(rows, other, repetitions=2000, seed=42):
    """Resample complete item-ID blocks with paired comparisons retained."""
    _, scaled = common_rows(rows)
    ids = sorted({r["case"]["item_id"] for r in rows})
    sums, counts = [], []
    for item in ids:
        values = [r["arms"]["B"]["metrics"]["mase"]-r["arms"][other]["metrics"]["mase"] for r in scaled if r["case"]["item_id"] == item]
        sums.append(sum(values)); counts.append(len(values))
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(repetitions, len(ids)))
    numer = np.asarray(sums)[indices].sum(axis=1)
    denom = np.asarray(counts)[indices].sum(axis=1)
    values = numer[denom > 0] / denom[denom > 0]
    return {"item_blocks": len(ids), "repetitions": repetitions, "valid_repetitions": len(values),
            "seed": seed, "low": float(np.quantile(values, .025)) if len(values) else None,
            "high": float(np.quantile(values, .975)) if len(values) else None,
            "scope": "Item-block resampling conditional on these three stores and six origins; not independent daily-error inference."}


def summarize(result):
    rows = result["rows"]
    common, scaled = common_rows(rows)
    origins = sorted({r["case"]["as_of"] for r in rows})
    return {"version": VERSION, "result_semantic_hash": digest(w.semantic(result)),
            "arms": arm_summary(rows), "common_case_ids": [r["case"]["case_id"] for r in common],
            "scaled_case_ids": [r["case"]["case_id"] for r in scaled],
            "contrasts": {a: contrast(rows, a) for a in ("F", "D")},
            "by_store": {s: {a: contrast([r for r in rows if r["case"]["store"] == s], a) for a in ("F", "D")} for s in STORES},
            "by_origin": {o: {a: contrast([r for r in rows if r["case"]["as_of"] == o], a) for a in ("F", "D")} for o in origins},
            "by_store_origin": [{"store": s, "origin": o, **{a: contrast([r for r in rows if r["case"]["store"] == s and r["case"]["as_of"] == o], a) for a in ("F", "D")}}
                                for s in STORES for o in origins],
            "by_zero_band": [{"band": b, **{a: contrast([r for r in rows if zero_band(r) == b], a) for a in ("F", "D")}} for b in range(4)],
            "bootstrap": {a: bootstrap_items(rows, a) for a in ("F", "D")},
            "leave_one_origin_out": {o: {a: contrast([r for r in rows if r["case"]["as_of"] != o], a) for a in ("F", "D")} for o in origins},
            "routes": dict(Counter(r["policy_route"] for r in rows)),
            "selected_methods": {a: dict(Counter(r["arms"][a]["selected"] for r in rows)) for a in ARMS}}


def timing_cases(cases, protocol):
    selected = set(protocol["timing_item_ids"])
    return [c for c in cases if c["role"] == "main" and c["item_id"] in selected]


def measure_uncached(data, cases):
    # Warm imports/fitter on development-only data, outside measured totals.
    warm = next(c for c in data.cases if c["role"] == "development")
    h = history(data, warm)
    n.run_forecast_path("regular", h); n.run_forecast_path("intermittent", h)
    rows = []
    for i, case in enumerate(cases):
        arms = {}
        for arm in (("B", "D") if i % 2 == 0 else ("D", "B")):
            start = time.perf_counter()
            h = history(data, case)
            if arm == "B":
                evidence = w.profile(view(data, case)); route = w.rule_route(evidence)
                out = n.run_forecast_path(route, h)
            else:
                out = n.run_unrestricted_reference(h, [])
            arms[arm] = {"seconds": time.perf_counter()-start, "fit_attempts": len(out["fit_trace"]),
                         "forecast_hash": digest(out.get("forecast")), "selected": out["selected"], "status": out["status"]}
        rows.append({"case_id": case["case_id"], "order": list(arms), "arms": arms})
    return {"session_id": w.timestamp(), "platform": platform.platform(), "versions": n.package_versions(),
            "scope": "36 paired uncached cases; B includes profile/dispatch; D searches all methods; import warm-up excluded.",
            "rows": rows}


def work_record(bank, result, timings):
    snapshot = bank.snapshot() if isinstance(bank, FitBank) else bank
    timed = {}
    for arm in ("B", "D"):
        values = [r["arms"][arm] for r in timings["rows"]]
        timed[arm] = {"n": len(values), "seconds": sum(v["seconds"] for v in values),
                      "median_seconds": float(np.median([v["seconds"] for v in values])),
                      "fit_attempts": sum(v["fit_attempts"] for v in values)}
    return {"bank": {"actual_fit_attempts": snapshot["actual_fit_attempts"], "lookups": snapshot["lookups"],
                     "cache_hits": snapshot["cache_hits"], "seconds": sum(r["fit"]["seconds"] for r in snapshot["fits"].values()),
                     "failed_fits": sum(r["fit"]["status"] != "ok" for r in snapshot["fits"].values()),
                     "scope": "Development plus main; exact training-window fits shared across arms and origins."},
            "logical_main": {a: sum(r["arms"][a]["logical_fit_attempts"] for r in result["rows"]) for a in ARMS},
            "uncached_timing": timed, "timing_session": timings["session_id"], "new_llm_calls": 0}
