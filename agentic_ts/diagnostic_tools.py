"""A03 numerical tools. Instances own one dated prefix and one arm's caches."""
from collections import Counter
from importlib.metadata import version
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from .forecasting import FrozenJSON, digest, evaluate_forecast

CANDIDATES = ("seasonal_naive", "hw_full", "hw_recent60")
CASE_IDS = ("A3349335T", "A3349399C", "A3349337W", "A3349709X", "A3349565X", "A3349483V", "A3349416V", "A3349640L")
REQUEST = "Investigate this monthly retail series and produce its next twelve-month forecast within the available investigation budget."
NUMERICAL = {
    "version": "a03-v1", "available_end": "2024-06-01", "history_n": 294,
    "fold_ends": [258, 270, 282], "horizon": 12, "period": 12, "recent_n": 60,
    "fit": {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": 12,
            "initialization_method": "estimated", "use_boxcox": False},
    "optimizer": {"method": "L-BFGS-B", "optimized": True, "remove_bias": False,
                  "use_brute": True, "minimize_kwargs": {"options": {"maxiter": 2000}}},
    "diagnostics": {"stl_n": 120, "period": 12, "seasonal": 13, "trend": 21, "low_pass": 13,
                    "seasonal_deg": 1, "trend_deg": 1, "low_pass_deg": 1,
                    "seasonal_jump": 1, "trend_jump": 1, "low_pass_jump": 1, "robust": True,
                    "inner_iter": 2, "outer_iter": 15, "residual_n": 48, "discard_n": 12},
    "ranking": "mean three-fold RMSE; exact ties in candidate order; all folds must succeed",
    "forecast_policy": "finite, unclipped; flag negative values",
}
TOOL_MENU = {
    "decompose_history": "No arguments. Robust STL on the last 120 months; descriptive trend and seasonal strengths.",
    "compare_recent_history": "No arguments. Compare median annual differences in the last 24 vs previous 24 months; not a change-point test.",
    "check_residuals": "candidate_id: seasonal_naive or a successfully benchmarked competitor. Current-origin in-sample residual mean, SD and ACF lags 1/12 on the final 48 months.",
    "benchmark_candidate": "candidate_id: hw_full or hw_recent60. Execute three chronological twelve-month validation folds. Python ranks eligible candidates by mean RMSE.",
}


def numerical_versions():
    return {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")}


def available_series(panel, sid):
    """The only panel-to-controller gate. Never pass the complete series to a tool."""
    group = panel.loc[(panel.series_id == sid) & (panel.date <= NUMERICAL["available_end"])].sort_values("date")
    series = pd.Series(group.turnover_aud_millions.to_numpy(dtype=float), index=pd.DatetimeIndex(group.date), name=sid)
    validate_prefix(series)
    return series


def validate_prefix(series):
    expected = pd.date_range("2000-01-01", "2024-06-01", freq="MS")
    if not series.index.equals(expected) or len(series) != 294:
        raise ValueError("A03 requires exactly January 2000–June 2024 at monthly starts")
    if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
        raise ValueError("Available turnover must be finite and positive")


def series_value(series):
    return {"dates": series.index.strftime("%Y-%m-%d").tolist(), "values": series.to_list()}


def fit_current(prefix, candidate):
    """One numerical attempt, including fitted residuals; timings are kept elsewhere."""
    if candidate not in CANDIDATES:
        raise ValueError("Unknown candidate")
    train = prefix.iloc[-60:] if candidate == "hw_recent60" else prefix
    result = {"candidate_id": candidate, "status": "failed", "train_start": str(train.index[0].date()),
              "train_end": str(train.index[-1].date()), "train_n": len(train), "warnings": []}
    caught = []
    try:
        values = train.to_numpy(dtype=float)
        if len(train) < 24 or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid fit slice")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if candidate == "seasonal_naive":
                prediction = values[-12:].copy()
                residual = values[12:] - values[:-12]
            else:
                fitted = ExponentialSmoothing(values, **NUMERICAL["fit"]).fit(**NUMERICAL["optimizer"])
                if not bool(fitted.mle_retvals.get("success", False)):
                    raise ValueError("Optimizer did not converge")
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("Optimizer emitted a convergence warning")
                prediction = np.asarray(fitted.forecast(12), dtype=float)
                residual = values[12:] - np.asarray(fitted.fittedvalues, dtype=float)[12:]
        if prediction.shape != (12,) or not np.isfinite(prediction).all():
            raise ValueError("Invalid forecast")
        result.update(status="ok", forecast=prediction.tolist(),
                      dates=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS").strftime("%Y-%m-%d").tolist(),
                      negative_forecast=bool((prediction < 0).any()))
        # Residual failure must not invalidate an otherwise finite forecast.
        if np.isfinite(residual).all():
            result["residuals"] = {"dates": train.index[12:].strftime("%Y-%m-%d").tolist(), "values": residual.tolist()}
        else:
            result["residual_error"] = "Nonfinite fitted residuals"
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["warnings"] = [str(w.message) for w in caught]
    return result


def select_candidate(tested):
    eligible = [c for c in CANDIDATES if tested.get(c, {}).get("eligible")]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (tested[c]["mean_rmse"], CANDIDATES.index(c)))


def component_strength(component, residual):
    denominator = float(np.var(np.asarray(component) + residual, ddof=0))
    if denominator == 0:
        return None
    return float(max(0, 1 - np.var(residual, ddof=0) / denominator))


def residual_summary(residuals, expected):
    if residuals is None:
        raise ValueError("No finite residual sequence")
    dates = pd.DatetimeIndex(residuals["dates"])
    values = np.asarray(residuals["values"], dtype=float)
    if len(dates) != len(values) or not dates.is_unique:
        raise ValueError("Invalid residual alignment")
    mask = dates >= expected[0]
    if not dates[mask].equals(expected) or not np.isfinite(values[mask]).all():
        raise ValueError("Require the exact finite final 48 residuals")
    values = values[mask]
    constant = bool(np.ptp(values) == 0)
    correlations = None if constant else acf(values, nlags=12, fft=False, adjusted=False, missing="raise")
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=0)),
            "acf_lag1": None if constant else float(correlations[1]),
            "acf_lag12": None if constant else float(correlations[12]),
            "acf_null_reason": "Constant residuals" if constant else None,
            "start": str(expected[0].date()), "end": str(expected[-1].date()), "n": len(values),
            "definition": "In-sample one-step residuals after discarding the first 12 fit observations; not multi-step validation errors"}


class DiagnosticSession:
    """One arm/series. Caches do not cross arms; failed fits are cached too."""
    def __init__(self, prefix, metadata, fitter=fit_current):
        validate_prefix(prefix)
        self.prefix = prefix.copy()
        self.metadata = FrozenJSON.of(metadata).data
        self.prefix_hash = digest(series_value(prefix))
        self.fitter = fitter
        self.work = Counter()
        self.operations = []
        self.current_cache = {}
        self.tested = {}
        self.seen = set()
        self.observations = []
        self.baseline = self.benchmark("seasonal_naive")
        self.evidence = {
            "request": {"request": REQUEST, **metadata, "units": "nominal AUD millions",
                        "available_start": "2000-01-01", "available_end": "2024-06-01", "n": 294,
                        "forecast_start": "2024-07-01", "forecast_end": "2025-06-01",
                        "limitations": "Original turnover, one revised ABS vintage. No final outcomes or causal evidence are available."},
            "protocol": {"numerical": NUMERICAL, "versions": numerical_versions(), "tools": TOOL_MENU},
            "baseline": self.baseline,
        }

    def fit(self, prefix, candidate, purpose):
        self.work[purpose + "_fit_attempts"] += 1
        started = time.perf_counter()
        try:
            fit = self.fitter(prefix.copy(), candidate)
        except Exception as exc:
            fit = {"status": "failed", "candidate_id": candidate, "error": f"{type(exc).__name__}: {exc}", "warnings": []}
        self.work[purpose + "_fit_failures"] += fit["status"] != "ok"
        self.operations.append({"kind": purpose + "_fit", "candidate_id": candidate,
                                "prefix_start": str(prefix.index[0].date()), "prefix_end": str(prefix.index[-1].date()),
                                "seconds": time.perf_counter() - started, "status": fit["status"]})
        return fit

    def benchmark(self, candidate):
        folds = []
        for end in NUMERICAL["fold_ends"]:
            train, actual = self.prefix.iloc[:end], self.prefix.iloc[end:end + 12]
            fitted = self.fit(train, candidate, "validation")
            fold = {"available_start": str(train.index[0].date()), "available_end": str(train.index[-1].date()),
                    "available_n": len(train), "fit_start": str((train.iloc[-60:] if candidate == "hw_recent60" else train).index[0].date()),
                    "fit_n": 60 if candidate == "hw_recent60" else len(train),
                    "test_start": str(actual.index[0].date()), "test_end": str(actual.index[-1].date()),
                    "status": fitted["status"], "warnings": fitted.get("warnings", [])}
            try:
                if fitted["status"] != "ok":
                    raise ValueError(fitted.get("error", "Fit failed"))
                fold["metrics"] = evaluate_forecast(actual, fitted["forecast"], train)
                fold["negative_forecast"] = fitted.get("negative_forecast", False)
            except ValueError as exc:
                fold.update(status="failed", error=str(exc))
            folds.append(fold)
        eligible = all(f["status"] == "ok" for f in folds)
        result = {"candidate_id": candidate, "eligible": eligible, "folds": folds,
                  "mean_rmse": float(np.mean([f["metrics"]["rmse"] for f in folds])) if eligible else None}
        self.tested[candidate] = result
        return result

    def current_fit(self, candidate):
        key = (candidate, self.prefix_hash, digest(NUMERICAL))
        if key in self.current_cache:
            self.work["current_fit_reuses"] += 1
        else:
            self.current_cache[key] = self.fit(self.prefix, candidate, "current")
        return self.current_cache[key]

    def decompose_history(self):
        self.work["decompositions"] += 1
        values = self.prefix.iloc[-120:]
        config = {k: v for k, v in NUMERICAL["diagnostics"].items() if k not in {"stl_n", "inner_iter", "outer_iter", "residual_n", "discard_n"}}
        fitted = STL(values.to_numpy(), **config).fit(inner_iter=2, outer_iter=15)
        trend = component_strength(fitted.trend, fitted.resid)
        seasonal = component_strength(fitted.seasonal, fitted.resid)
        return {"start": str(values.index[0].date()), "end": str(values.index[-1].date()), "n": 120,
                "trend_strength": trend, "seasonal_strength": seasonal,
                "null_reason": "Zero variance denominator" if trend is None or seasonal is None else None,
                "definition": "max(0, 1 - var(residual)/var(component+residual)), population variances; descriptive STL features"}

    def compare_recent_history(self):
        annual = self.prefix.diff(12)
        previous, recent = annual.iloc[-48:-24], annual.iloc[-24:]
        scale = float(np.mean(np.abs(annual.iloc[12:].to_numpy())))
        contrast = float(recent.median() - previous.median())
        return {"previous_start": str(previous.index[0].date()), "previous_end": str(previous.index[-1].date()),
                "recent_start": str(recent.index[0].date()), "recent_end": str(recent.index[-1].date()),
                "required_source_start": "2019-07-01", "required_source_end": "2024-06-01",
                "previous_median": float(previous.median()), "recent_median": float(recent.median()),
                "contrast": contrast, "full_prefix_seasonal_scale": scale,
                "normalized_contrast": contrast / scale if scale > 0 else None,
                "null_reason": "Zero seasonal scale" if scale == 0 else None,
                "definition": "Median annual difference y[t]-y[t-12], recent 24 minus previous 24 months. Not a detected structural break."}

    def check_residuals(self, candidate_id):
        if not self.tested.get(candidate_id, {}).get("eligible") and candidate_id != "seasonal_naive":
            raise ValueError("Residual candidate must be successfully benchmarked")
        fitted = self.current_fit(candidate_id)
        if fitted["status"] != "ok":
            raise ValueError(fitted.get("error", "Current fit failed"))
        return {"candidate_id": candidate_id, "fit_start": fitted["train_start"],
                "warnings": fitted["warnings"],
                **residual_summary(fitted.get("residuals"), self.prefix.index[-48:])}

    def invoke(self, action, arguments):
        """Controller validates first. Every admitted call, including duplicates, counts."""
        self.work["tool_attempts"] += 1
        key = digest({"action": action, "arguments": arguments})
        started = time.perf_counter()
        observation = {"id": f"o{len(self.observations) + 1}", "tool": action, "arguments": arguments}
        if key in self.seen:
            self.work["duplicate_tools"] += 1
            observation.update(status="duplicate", result={"message": "Identical call already attempted; no recomputation"})
        else:
            self.seen.add(key)
            self.work["executed_tools"] += 1
            self.work["diagnostic_tools"] += action != "benchmark_candidate"
            try:
                if action == "benchmark_candidate":
                    result = self.benchmark(arguments["candidate_id"])
                else:
                    result = getattr(self, action)(**arguments)
                observation.update(status="ok", result=result)
            except Exception as exc:
                self.work["failed_tools"] += 1
                observation.update(status="failed", result={"error": f"{type(exc).__name__}: {exc}"})
        self.operations.append({"kind": "tool", "action": action, "seconds": time.perf_counter() - started, "status": observation["status"]})
        # Strict serialization prevents nonfinite values reaching the model.
        observation = FrozenJSON.of(observation).data
        self.observations.append(observation)
        self.evidence[observation["id"]] = observation
        return observation

    def finalize(self):
        selected = select_candidate(self.tested)
        if selected is None:
            return {"selected_candidate": None, "delivered_candidate": None, "forecast_status": "failed", "error": "No eligible candidate"}
        fitted = self.current_fit(selected)
        delivered = selected
        status = "ok"
        if fitted["status"] != "ok":
            delivered, status = "seasonal_naive", "fallback"
            fitted = self.current_fit(delivered)
        result = {"selected_candidate": selected, "delivered_candidate": delivered if fitted["status"] == "ok" else None,
                  "forecast_status": status if fitted["status"] == "ok" else "failed",
                  "selected_attempt_status": self.current_cache[(selected, self.prefix_hash, digest(NUMERICAL))]["status"]}
        if fitted["status"] == "ok":
            result.update(forecast=fitted["forecast"], dates=fitted["dates"], negative_forecast=fitted["negative_forecast"])
        else:
            result["error"] = fitted.get("error", "Baseline fallback failed")
        return result
