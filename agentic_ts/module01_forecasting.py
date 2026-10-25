"""Fresh A01 numerical tools. Language stages cannot choose or alter forecasts."""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {
    "seasonal_naive": "Repeat the last twelve observed months.",
    "hw_add": "Damped additive trend and additive seasonality, period twelve.",
    "hw_mul": "Damped additive trend and multiplicative seasonality, period twelve.",
}
PROTOCOL = {
    "version": "module01-fresh-v1", "series_id": "IPG2211N", "units": "Index 2017=100",
    "frequency": "MS", "start": "2000-01-01", "development_end": "2024-12-01",
    "final_start": "2025-01-01", "final_end": "2025-12-01", "horizon": 12,
    "period": 12, "fold_train_n": [264, 276, 288], "development_n": 300,
    "catalog": CATALOG, "selection": "Minimum full-precision mean three-fold RMSE; catalog-order exact ties.",
    "fit": {"trend": "add", "damped_trend": True, "seasonal_periods": 12,
            "initialization_method": "estimated", "use_boxcox": False},
    "optimizer": {"optimized": True, "method": "L-BFGS-B", "use_brute": True,
                  "remove_bias": False, "minimize_kwargs": {"options": {"maxiter": 2000}}},
    "policy": "All folds must converge. Final refit failure delivers labeled seasonal naive. No retries.",
    "scope": "One revised-vintage series and one final year; point forecasts only.",
}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def semantic(value):
    """Exclude execution clocks, never numeric results, from reproducibility checks."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {"runtime_s", "created_at"}}
    if isinstance(value, list):
        return [semantic(x) for x in value]
    return value


def versions():
    return {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels")}


def validate_series(series, *, start=None, end=None):
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Expected a dated pandas Series")
    if not len(series) or series.index.has_duplicates:
        raise ValueError("Empty series or duplicate months")
    expected = pd.date_range(start or series.index[0], end or series.index[-1], freq="MS")
    if not series.index.equals(expected):
        raise ValueError("Expected an uninterrupted, ordered month-start grid")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Observations must be finite and positive; no silent imputation")


def load_snapshot(directory=ROOT / "data/module01"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for key, expected in {"series_id": "IPG2211N", "units": "Index 2017=100",
                          "frequency": "MS", "seasonal_adjustment": "Not seasonally adjusted"}.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Unexpected source metadata: {key}")
    if file_digest(directory / "electric_power_monthly.csv") != manifest["processed_sha256"]:
        raise ValueError("Processed snapshot checksum mismatch")
    if file_digest(directory / manifest["raw_path"]) != manifest["raw_sha256"]:
        raise ValueError("Raw snapshot checksum mismatch")
    frame = pd.read_csv(directory / "electric_power_monthly.csv", parse_dates=["date"])
    if list(frame.columns) != ["date", "production_index"]:
        raise ValueError("Unexpected data columns")
    series = frame.set_index("date").production_index.astype(float)
    validate_series(series, start="2000-01-01", end="2025-12-01")
    if len(series) != 312:
        raise ValueError("Expected 312 months")
    return series, manifest


def split_snapshot(series):
    validate_series(series, start="2000-01-01", end="2025-12-01")
    return series.loc[:"2024-12-01"].copy(), series.loc["2025-01-01":].copy()


def profile(development):
    validate_series(development, start="2000-01-01", end="2024-12-01")
    values = development.to_numpy(dtype=float)
    a, b = values[:-12], values[12:]
    corr = None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    return {"series_id": "IPG2211N", "units": "Index 2017=100", "n": len(values),
            "start": "2000-01-01", "end": "2024-12-01", "missing": 0,
            "minimum": float(values.min()), "maximum": float(values.max()),
            "lag12_correlation": corr, "prefix_hash": digest({"dates": development.index.strftime("%Y-%m-%d").tolist(),
                                                              "values": values.tolist()})}


def scores(actual, forecast):
    a, p = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if a.ndim != 1 or a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require same-length finite one-dimensional arrays")
    error = a - p
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


def fit_method(train, model, horizon=12):
    """Exactly one fit attempt. Optimizer warnings do not become silent successes."""
    start = time.perf_counter()
    result = {"model": model, "train_n": len(train), "train_start": str(train.index[0].date()),
              "train_end": str(train.index[-1].date()), "status": "failed", "warnings": []}
    caught = []
    try:
        validate_series(train)
        if model not in CATALOG or len(train) < 24 or horizon != 12:
            raise ValueError("Unknown model, insufficient history or unsupported horizon")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "seasonal_naive":
                prediction = train.to_numpy()[-12:].copy()
            else:
                fit = ExponentialSmoothing(train.to_numpy(), seasonal="add" if model == "hw_add" else "mul",
                                           **PROTOCOL["fit"]).fit(**PROTOCOL["optimizer"])
                success = fit.mle_retvals.get("success")
                result["optimizer_success"] = bool(success) if success is not None else None
                if success is None or not bool(success) or any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("Optimizer did not provide an unambiguous converged fit")
                prediction = np.asarray(fit.forecast(horizon), dtype=float)
        if prediction.shape != (12,) or not np.isfinite(prediction).all():
            raise ValueError("Forecast must contain twelve finite values")
        dates = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
        result.update(status="ok", forecast=prediction.tolist(), dates=dates.strftime("%Y-%m-%d").tolist(),
                      negative_forecast=bool((prediction < 0).any()))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    result["runtime_s"] = time.perf_counter() - start
    return result


def backtest(development, fitter=fit_method):
    profile(development)  # Enforces the permitted prefix; no final outcomes enter.
    comparisons = {}
    for model in CATALOG:
        folds = []
        for n in PROTOCOL["fold_train_n"]:
            train, valid = development.iloc[:n].copy(), development.iloc[n:n + 12].copy()
            fitted = fitter(train, model)
            fold = {"train_n": n, "validation_start": str(valid.index[0].date()),
                    "validation_end": str(valid.index[-1].date()), "fit": fitted}
            if fitted["status"] == "ok":
                fold["metrics"] = scores(valid, fitted["forecast"])
            folds.append(fold)
        eligible = all(f["fit"]["status"] == "ok" for f in folds)
        comparisons[model] = {"folds": folds, "eligible": eligible,
                              "mean_rmse": float(np.mean([f["metrics"]["rmse"] for f in folds])) if eligible else None}
    return comparisons


def select_model(comparisons):
    eligible = [m for m in CATALOG if comparisons[m]["eligible"]]
    if not eligible:
        raise ValueError("No eligible candidate")
    return min(eligible, key=lambda m: (comparisons[m]["mean_rmse"], list(CATALOG).index(m)))


def prepare_forecast(development, fitter=fit_method):
    """Freeze all numerical choices without accepting a final-outcome argument."""
    started = time.perf_counter()
    summary = profile(development)
    comparisons = backtest(development, fitter)
    selected = select_model(comparisons)
    baseline = fitter(development, "seasonal_naive")
    if baseline["status"] != "ok":
        raise ValueError("Baseline failed; cannot deliver the comparison")
    chosen = baseline if selected == "seasonal_naive" else fitter(development, selected)
    fallback = chosen["status"] != "ok"
    delivered = baseline if fallback else chosen
    result = {"protocol_hash": digest(PROTOCOL), "versions": versions(), "profile": summary,
              "backtest": comparisons, "selected_model": selected,
              "delivered_model": "seasonal_naive" if fallback else selected,
              "delivery_status": "fallback" if fallback else "ok", "selected_attempt": chosen,
              "baseline_attempt": baseline, "dates": delivered["dates"],
              "forecast": delivered["forecast"], "baseline_forecast": baseline["forecast"]}
    result["frozen_hash"] = digest(semantic(result))
    result["runtime_s"] = time.perf_counter() - started
    return result


def score_frozen(frozen, outcomes):
    if digest(semantic({k: v for k, v in frozen.items() if k != "frozen_hash"})) != frozen["frozen_hash"]:
        raise ValueError("Frozen forecast has changed")
    validate_series(outcomes, start="2025-01-01", end="2025-12-01")
    if outcomes.index.strftime("%Y-%m-%d").tolist() != frozen["dates"]:
        raise ValueError("Outcome dates do not match the frozen forecast")
    selected, baseline = scores(outcomes, frozen["forecast"]), scores(outcomes, frozen["baseline_forecast"])
    comparison = "lower" if selected["rmse"] < baseline["rmse"] else "higher" if selected["rmse"] > baseline["rmse"] else "equal"
    return {"frozen_hash": frozen["frozen_hash"], "dates": frozen["dates"], "actual": outcomes.tolist(),
            "delivered": selected, "baseline": baseline, "rmse_direction": comparison,
            "rmse_difference": selected["rmse"] - baseline["rmse"]}


def profile_evidence(summary):
    facts = {"data.profile": deepcopy(summary), "data.meaning": "Real output index for US electric power generation, transmission and distribution; not electricity demand or kWh.",
             "protocol.selection": PROTOCOL["selection"], "protocol.validation": [2022, 2023, 2024],
             "limitations.vintage": "A revised snapshot is not a real-time vintage backtest.",
             "limitations.scope": "One series, three validation years and one final year do not establish general performance.",
             "limitations.uncertainty": "Point forecasts only; no calibrated prediction intervals or causal explanation.",
             "limitations.workflow": "The fixed Python catalog and numerical selection are not controlled by the language model."}
    facts.update({"catalog." + m: note for m, note in CATALOG.items()})
    return facts


def result_evidence(frozen, evaluation):
    if evaluation["frozen_hash"] != frozen["frozen_hash"]:
        raise ValueError("Evaluation belongs to a different forecast")
    facts = profile_evidence(frozen["profile"])
    facts.update({"validation.results": {m: {"eligible": x["eligible"], "fold_rmse": [f.get("metrics", {}).get("rmse") for f in x["folds"]], "mean_rmse": x["mean_rmse"]} for m, x in frozen["backtest"].items()},
                  "forecast.delivery": {k: frozen[k] for k in ("selected_model", "delivered_model", "delivery_status")},
                  "evaluation.final": deepcopy(evaluation)})
    return facts
