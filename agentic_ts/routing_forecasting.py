"""A02's fixed daily candidates and dated, training-only numerical procedures."""
from dataclasses import asdict, dataclass, replace
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

from .forecasting import FrozenJSON, digest

FAMILIES = {"regular": ("seasonal_naive", "hw_add_damped"),
            "intermittent": ("seasonal_naive", "tsb")}
CATALOG = ("seasonal_naive", "hw_add_damped", "tsb")


@dataclass(frozen=True)
class ForecastConfig:
    train_window: int = 365
    horizon: int = 28
    season_length: int = 7
    n_folds: int = 3
    optimizer: str = "L-BFGS-B"
    maxiter: int = 2000
    tsb_alpha_d: float = .1
    tsb_alpha_p: float = .1
    nonnegative: bool = True

    @property
    def required_history(self):
        return self.train_window + self.horizon * self.n_folds


def load_routing_panel(directory="data/a02"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = (directory / "m5_routing_panel.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("Panel fingerprint differs from its manifest")
    panel = pd.read_csv(directory / "m5_routing_panel.csv", parse_dates=["date"])
    expected = {s["series_id"] for s in manifest["series"]}
    if set(panel.columns) != {"series_id", "date", "sales_units"} or len(expected) != 12:
        raise ValueError("Expected the declared 12-series daily schema")
    if len(panel) != 5724 or set(panel.series_id) != expected or panel.isna().any().any():
        raise ValueError("Incomplete or unexpected panel membership")
    if (manifest["target"], manifest["units"], manifest["frequency"]) != ("observed_sales", "item units", "D"):
        raise ValueError("Unexpected target, units, or frequency")
    values = panel.sales_units.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Source sales must be finite nonnegative counts")
    dates = pd.date_range("2015-01-04", "2016-04-24", freq="D")
    for sid, part in panel.groupby("series_id"):
        if not pd.DatetimeIndex(part.sort_values("date").date).equals(dates):
            raise ValueError(f"Incomplete or duplicate daily grid: {sid}")
    return panel, manifest


def profile_history(history, request, config=ForecastConfig()):
    """Build evidence from the available prefix, using the last 365 days for routing."""
    current = history.iloc[-config.train_window:]
    zeros = int((current == 0).sum())
    return FrozenJSON.of({
        "request.context": dict(request),
        "history.available_span": {"start": history.index[0].date().isoformat(),
                                   "end": history.index[-1].date().isoformat(), "observations": len(history)},
        "history.current_window": {"start": current.index[0].date().isoformat(), "end": current.index[-1].date().isoformat(),
                                   "observed_count": len(current), "zero_count": zeros,
                                   "positive_count": len(current) - zeros, "zero_fraction": zeros / len(current)},
        "data.target": {"target": "observed_sales", "units": "item units", "frequency": "D"},
        "data.availability": "No availability or price observations are supplied. Zero sales alone do not establish stockouts, obsolescence, or zero demand.",
    })


def validation_windows(history, config=ForecastConfig()):
    if len(history) < config.required_history:
        raise ValueError("Insufficient dated history for the fixed validation protocol")
    for i in range(config.n_folds):
        end = len(history) - (config.n_folds - i) * config.horizon
        train = history.iloc[end - config.train_window:end].copy()
        valid = history.iloc[end:end + config.horizon].copy()
        if len(train) != config.train_window or len(valid) != config.horizon or train.index[-1] >= valid.index[0]:
            raise ValueError("Invalid chronological boundary")
        yield train, valid


def score_forecast(actual, prediction, train, config=ForecastConfig()):
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Forecast and actual shapes must match")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite evaluation values")
    error = actual - prediction
    mae = float(np.mean(np.abs(error)))
    values = np.asarray(train, dtype=float)
    scale = float(np.mean(np.abs(values[config.season_length:] - values[:-config.season_length])))
    defined = np.isfinite(scale) and scale > 0
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": mae,
            "mase": mae / scale if defined else None, "scale": scale if np.isfinite(scale) else None,
            "scale_status": "defined" if defined else "undefined"}


def fit_forecast(train, model, config=ForecastConfig()):
    """One bounded fit, with explicit optimizer failure and output-clamp records."""
    started = time.perf_counter()
    result = {"model": model, "status": "failed", "warnings": [], "clip_count": 0,
              "train_start": train.index[0].date().isoformat(), "train_end": train.index[-1].date().isoformat(),
              "train_n": len(train)}
    caught = []
    try:
        values = np.asarray(train, dtype=float)
        if model not in CATALOG or len(values) < 2 * config.season_length or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Unsupported model or invalid training values")
        if model == "tsb":
            # Import/UI warnings are not numerical fitting diagnostics.
            from statsforecast.models import TSB
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "seasonal_naive":
                predictions = np.resize(values[-config.season_length:], config.horizon)
            elif model == "hw_add_damped":
                fitted = ExponentialSmoothing(values, trend="add", damped_trend=True, seasonal="add",
                                              seasonal_periods=config.season_length, initialization_method="estimated",
                                              use_boxcox=False).fit(method=config.optimizer, optimized=True, remove_bias=False,
                                              minimize_kwargs={"options": {"maxiter": config.maxiter}})
                if not bool(fitted.mle_retvals.get("success", False)) or any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("Holt-Winters optimizer did not converge")
                predictions = np.asarray(fitted.forecast(config.horizon), dtype=float)
            else:
                predictions = np.asarray(TSB(alpha_d=config.tsb_alpha_d, alpha_p=config.tsb_alpha_p)
                                         .forecast(values, h=config.horizon)["mean"], dtype=float)
        if predictions.shape != (config.horizon,) or not np.isfinite(predictions).all():
            raise ValueError("Invalid forecast shape or nonfinite output")
        if config.nonnegative:
            result["clip_count"] = int((predictions < 0).sum())
            predictions = np.maximum(0, predictions)
        dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=config.horizon, freq="D")
        result.update(status="ok", forecast=predictions.tolist(), dates=dates.strftime("%Y-%m-%d").tolist())
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["warnings"] = [str(w.message) for w in caught]
    result["seconds"] = time.perf_counter() - started
    return result


def _attempt(train, model, phase, trace, config, forecaster):
    fit = forecaster(train, model, config)
    trace.append({"phase": phase, **{k: v for k, v in fit.items() if k not in ("forecast", "dates")}})
    return fit


def backtest_candidates(history, models, config=ForecastConfig(), forecaster=fit_forecast, trace=None):
    trace = [] if trace is None else trace
    candidates = {}
    for model in models:
        folds = []
        for i, (train, valid) in enumerate(validation_windows(history, config), 1):
            fit = _attempt(train, model, f"validation_{i}", trace, config, forecaster)
            fold = {"status": fit["status"], "train_start": train.index[0].date().isoformat(),
                    "train_end": train.index[-1].date().isoformat(), "validation_start": valid.index[0].date().isoformat(),
                    "validation_end": valid.index[-1].date().isoformat(), "warnings": fit["warnings"]}
            if fit["status"] == "ok":
                fold["metrics"] = score_forecast(valid, fit["forecast"], train, config)
            else:
                fold["error"] = fit.get("error", "Numerical fit failed")
            folds.append(fold)
        eligible = all(f["status"] == "ok" for f in folds)
        candidates[model] = {"folds": folds, "eligible": eligible,
                             "mean_rmse": float(np.mean([f["metrics"]["rmse"] for f in folds])) if eligible else None}
    return candidates


def select_candidate(candidates, catalog=CATALOG):
    if not candidates.get("seasonal_naive", {}).get("eligible"):
        raise ValueError("The common baseline failed validation")
    eligible = [m for m in catalog if m in candidates and candidates[m]["eligible"]]
    return min(eligible, key=lambda m: (candidates[m]["mean_rmse"], catalog.index(m)))


def _finish_path(history, route, candidates, trace, config, forecaster, reusable_final=None, catalog=CATALOG):
    result = {"route": route, "status": "numerical_failure", "candidates": candidates, "fit_trace": trace,
              "selected": None, "delivered": None, "final_fits": {}, "forecast": None, "dates": None}
    try:
        selected = select_candidate(candidates, catalog)
        result["selected"] = selected
        train = history.iloc[-config.train_window:]
        for model in dict.fromkeys(["seasonal_naive", selected]):
            if reusable_final and model in reusable_final:
                fit = reusable_final[model]
                result.setdefault("reused_final", []).append(model)
            else:
                fit = _attempt(train, model, "final_refit", trace, config, forecaster)
            result["final_fits"][model] = fit
        baseline = result["final_fits"]["seasonal_naive"]
        if baseline["status"] != "ok":
            raise ValueError("The common baseline failed final fitting")
        fit = result["final_fits"][selected]
        delivered = selected if fit["status"] == "ok" else "seasonal_naive"
        fit = result["final_fits"][delivered]
        result.update(status="ok" if delivered == selected else "model_fallback", delivered=delivered,
                      forecast=fit["forecast"], dates=fit["dates"], baseline_forecast=baseline["forecast"])
    except ValueError as exc:
        result["error"] = str(exc)
    return result


def run_forecast_path(route, history, config=ForecastConfig(), forecaster=fit_forecast):
    if route not in FAMILIES:
        raise ValueError("No numerical path exists for this route")
    started = time.perf_counter()
    trace = []
    candidates = backtest_candidates(history, FAMILIES[route], config, forecaster, trace)
    result = _finish_path(history, route, candidates, trace, config, forecaster)
    result["seconds"] = time.perf_counter() - started
    result["config"] = asdict(config)
    return result


def run_unrestricted_reference(history, prior_results, config=ForecastConfig(), forecaster=fit_forecast):
    """Offline diagnostic, isolated from the main dispatcher and its prompts."""
    candidates, final_fits, trace, reused = {}, {}, [], []
    for prior in prior_results:
        numerical = prior.get("numerical") or {}
        if numerical.get("config") != asdict(config):
            continue
        candidates.update(numerical.get("candidates", {}))
        final_fits.update(numerical.get("final_fits", {}))
    reused = [m for m in CATALOG if m in candidates]
    missing = [m for m in CATALOG if m not in candidates]
    candidates.update(backtest_candidates(history, missing, config, forecaster, trace))
    result = _finish_path(history, "all_candidates_diagnostic", candidates, trace, config, forecaster, final_fits)
    result.update(reused_candidate_records=reused, config=asdict(config))
    return result


def score_frozen_forecasts(frozen, panel, config=ForecastConfig()):
    """Final outcomes are read only after every main and diagnostic forecast is frozen."""
    record = frozen.data
    for row in record["requests"]:
        sid, cutoff = row["request"]["series_id"], pd.Timestamp(row["request"]["as_of"])
        series = panel.loc[panel.series_id == sid].set_index("date").sales_units.sort_index()
        train = series.loc[:cutoff].iloc[-config.train_window:]
        for key, result in row["arms"].items():
            numerical = result.get("numerical")
            if numerical and numerical.get("forecast") is not None:
                dates = pd.DatetimeIndex(pd.to_datetime(numerical["dates"]))
                expected_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=config.horizon, freq="D")
                if not dates.equals(expected_dates):
                    raise ValueError("Incorrect final forecast dates")
                actual = series.reindex(dates)
                result["evaluation"] = {"delivered": score_forecast(actual, numerical["forecast"], train, config),
                                        "baseline": score_forecast(actual, numerical["baseline_forecast"], train, config)}
    record["forecast_fingerprint"] = frozen.fingerprint
    record["scored_after_freeze"] = True
    return FrozenJSON.of(record)


def package_versions():
    return {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "statsforecast")}


SHORT_CATALOG = ("seasonal_naive", "mean28")


def fit_simple(train, model, config=ForecastConfig()):
    """Add a trailing mean without changing the original numerical catalog."""
    if model != "mean28":
        return fit_forecast(train, model, config)
    started = time.perf_counter()
    values = np.asarray(train, dtype=float)
    result = {"model": model, "status": "failed", "warnings": [], "clip_count": 0,
              "train_start": train.index[0].date().isoformat(),
              "train_end": train.index[-1].date().isoformat(), "train_n": len(train)}
    if len(values) < 28 or not np.isfinite(values).all() or (values < 0).any():
        result["error"] = "Trailing mean needs 28 finite nonnegative observations"
    else:
        result.update(status="ok", forecast=[float(values[-28:].mean())] * config.horizon,
                      dates=pd.date_range(train.index[-1] + pd.Timedelta(days=1),
                                          periods=config.horizon, freq="D").strftime("%Y-%m-%d").tolist())
    result["seconds"] = time.perf_counter() - started
    return result


def run_short_path(history, config=ForecastConfig(), forecaster=fit_simple):
    """One 28-day validation block; final refit uses all visible observations."""
    if not 56 <= len(history) < config.required_history:
        raise ValueError("Short workflow requires 56–448 observed days")
    started, trace = time.perf_counter(), []
    validation_config = replace(config, train_window=len(history) - config.horizon, n_folds=1)
    candidates = backtest_candidates(history, SHORT_CATALOG, validation_config, forecaster, trace)
    final_config = replace(config, train_window=len(history), n_folds=1)
    result = _finish_path(history, "short_history", candidates, trace, final_config,
                          forecaster, catalog=SHORT_CATALOG)
    result.update(seconds=time.perf_counter() - started, config=asdict(final_config),
                  validation_config=asdict(validation_config))
    return result


def donor_prior(donors, origin, config=ForecastConfig()):
    """A fixed median of three 28-day donor means; no target data are accepted."""
    started = time.perf_counter()
    origin = pd.Timestamp(origin)
    expected = pd.date_range(origin - pd.Timedelta(days=27), origin, freq="D")
    if len(donors) != 3:
        raise ValueError("The prior requires exactly three authorized donors")
    means = {}
    for sid, series in donors.items():
        if not series.index.equals(expected) or not np.isfinite(series).all() or (series < 0).any():
            raise ValueError("Donor data must be the complete 28-day pre-origin window")
        means[sid] = float(series.mean())
    value = float(np.median(list(means.values())))
    return {"route": "new_product", "status": "ok", "selected": "donor_median28",
            "delivered": "donor_median28", "forecast": [value] * config.horizon,
            "dates": pd.date_range(origin + pd.Timedelta(days=1), periods=config.horizon,
                                   freq="D").strftime("%Y-%m-%d").tolist(),
            "donor_means": means, "fit_trace": [], "candidates": {},
            "seconds": time.perf_counter() - started, "config": asdict(config)}
