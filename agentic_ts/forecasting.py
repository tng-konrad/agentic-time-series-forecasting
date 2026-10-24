"""Numerical authority for module 1: a fixed, reproducible panel experiment."""
from collections import Counter
from dataclasses import dataclass
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

CATALOG = {
    "seasonal_naive": {"seasonal": None, "damped_trend": False},
    "hw_add": {"seasonal": "add", "damped_trend": False},
    "hw_add_damped": {"seasonal": "add", "damped_trend": True},
    "hw_mul": {"seasonal": "mul", "damped_trend": False},
    "hw_mul_damped": {"seasonal": "mul", "damped_trend": True},
}
FOLD_ENDS = (258, 270, 282)
DEVELOPMENT_END = 294
HORIZON = PERIOD = 12


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenJSON:
    """An immutable value; callers receive a fresh copy on every read."""
    payload: str

    @classmethod
    def of(cls, value):
        return cls(json_text(value))

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()


def load_panel(directory="data/a01"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = (directory / "abs_retail_panel.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("Panel checksum differs from its manifest")
    panel = pd.read_csv(directory / "abs_retail_panel.csv", parse_dates=["date"])
    validate_panel(panel, manifest)
    return panel, manifest


def validate_panel(panel, manifest):
    expected_columns = {"series_id", "state", "industry", "date", "turnover_aud_millions"}
    if set(panel.columns) != expected_columns:
        raise ValueError("Unexpected panel columns")
    expected = {s["series_id"]: s for s in manifest["series"]}
    if len(expected) != 16 or set(panel.series_id) != set(expected) or len(panel) != 4896:
        raise ValueError("Expected exactly the declared 16-series panel")
    if (manifest["units"], manifest["series_type"], manifest["frequency"]) != (
            "nominal AUD millions", "Original", "MS"):
        raise ValueError("Unexpected source units, type, or frequency")
    if panel.isna().any().any() or panel.duplicated(["series_id", "date"]).any():
        raise ValueError("Missing values or duplicate series-month keys")
    values = panel.turnover_aud_millions.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Turnover must be finite and positive")
    dates = pd.date_range("2000-01-01", "2025-06-01", freq="MS")
    for sid, group in panel.groupby("series_id"):
        if not pd.DatetimeIndex(group.sort_values("date").date).equals(dates):
            raise ValueError(f"Incomplete monthly calendar: {sid}")
        for label in ("state", "industry"):
            if set(group[label]) != {expected[sid][label]}:
                raise ValueError(f"Wrong {label} for {sid}")


def split_series(series, end, horizon=HORIZON):
    train, test = series.iloc[:end], series.iloc[end:end + horizon]
    if len(train) != end or len(test) != horizon or train.index.max() >= test.index.min():
        raise ValueError("Invalid chronological split")
    return train.copy(), test.copy()


def seasonal_scale(train):
    values = np.asarray(train, dtype=float)
    if len(values) <= PERIOD:
        raise ValueError("Insufficient training data for seasonal MASE")
    scale = float(np.mean(np.abs(values[PERIOD:] - values[:-PERIOD])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Seasonal MASE denominator must be finite and positive")
    return scale


def fit_forecast(train, model, horizon=HORIZON):
    """One fit attempt. An optimizer failure makes this candidate ineligible."""
    started = time.perf_counter()
    caught = []
    result = {"model": model, "status": "failed", "warnings": []}
    try:
        if model not in CATALOG:
            raise ValueError(f"Unknown candidate: {model}")
        values = np.asarray(train, dtype=float)
        if len(values) < 2 * PERIOD or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid training slice")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "seasonal_naive":
                predictions = np.resize(values[-PERIOD:], horizon)
            else:
                fitted = ExponentialSmoothing(
                    values, trend="add", seasonal_periods=PERIOD,
                    initialization_method="estimated", use_boxcox=False,
                    **CATALOG[model],
                ).fit(method="L-BFGS-B", optimized=True, remove_bias=False,
                      minimize_kwargs={"options": {"maxiter": 2000}})
                if not bool(fitted.mle_retvals.get("success", False)):
                    raise ValueError("Optimizer did not converge")
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("Optimizer emitted a convergence warning")
                predictions = np.asarray(fitted.forecast(horizon), dtype=float)
        if predictions.shape != (horizon,) or not np.isfinite(predictions).all():
            raise ValueError("Invalid forecast shape or nonfinite values")
        dates = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        result.update(status="ok", forecast=predictions.tolist(), dates=dates.strftime("%Y-%m-%d").tolist())
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["warnings"] = [str(w.message) for w in caught]
    result["seconds"] = time.perf_counter() - started
    return result


def evaluate_forecast(actual, prediction, train):
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Evaluation shapes must match")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Evaluation values must be finite")
    error = actual - prediction
    mae = float(np.mean(np.abs(error)))
    scale = seasonal_scale(train)
    metrics = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": mae, "mase": mae / scale, "scale": scale}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite evaluation metric")
    return metrics


def compare(left, right):
    return "lower" if left < right else "higher" if left > right else "equal"


def backtest_candidates(development, forecaster=fit_forecast):
    candidates = {}
    for model in CATALOG:
        folds = []
        for end in FOLD_ENDS:
            train, actual = split_series(development, end)
            scale = seasonal_scale(train)  # Invalid scaling is a data error, not a model failure.
            fitted = forecaster(train, model)
            fold = {"train_end": str(train.index[-1].date()), "train_n": len(train),
                    "validation_start": str(actual.index[0].date()), "validation_end": str(actual.index[-1].date()),
                    "status": fitted["status"], "warnings": fitted.get("warnings", []), "scale": scale}
            if fitted["status"] == "ok":
                try:
                    fold["metrics"] = evaluate_forecast(actual, fitted["forecast"], train)
                except ValueError as exc:
                    fold.update(status="failed", error=str(exc))
            else:
                fold["error"] = fitted.get("error", "Fit failed")
            folds.append(fold)
        eligible = all(f["status"] == "ok" for f in folds)
        candidates[model] = {"eligible": eligible, "folds": folds,
                             "mean_rmse": float(np.mean([f["metrics"]["rmse"] for f in folds])) if eligible else None}
    return candidates


def select_candidate(candidates):
    eligible = [m for m in CATALOG if candidates[m]["eligible"]]
    if not eligible:
        raise ValueError("No candidate completed all three folds")
    return min(eligible, key=lambda m: (candidates[m]["mean_rmse"], list(CATALOG).index(m)))


def forecast_selected(development, selected, forecaster=fit_forecast):
    baseline = forecaster(development, "seasonal_naive")
    if baseline["status"] != "ok":
        raise ValueError("Seasonal-naive baseline failed; panel is incomplete")
    chosen = baseline if selected == "seasonal_naive" else forecaster(development, selected)
    fallback = chosen["status"] != "ok"
    delivered = baseline if fallback else chosen
    return {"selected": selected, "delivered": "seasonal_naive" if fallback else selected,
            "status": "fallback" if fallback else "ok", "fallback_reason": chosen.get("error") if fallback else None,
            "selected_attempt": chosen, "baseline_attempt": baseline,
            "dates": delivered["dates"], "forecast": delivered["forecast"], "baseline": baseline["forecast"]}


def summarize_panel(rows):
    if len(rows) != 16 or len({r["series_id"] for r in rows}) != 16:
        raise ValueError("Portfolio summaries require all 16 series")
    final = float(np.mean([r["evaluation"]["delivered"]["mase"] for r in rows]))
    baseline = float(np.mean([r["evaluation"]["baseline"]["mase"] for r in rows]))
    directions = Counter(r["evaluation"]["mae_direction"] for r in rows)
    return {"selected_model_counts": dict(Counter(r["delivery"]["selected"] for r in rows)),
            "delivered_model_counts": dict(Counter(r["delivery"]["delivered"] for r in rows)),
            "fallback_count": sum(r["delivery"]["status"] == "fallback" for r in rows),
            "final_comparison": {"delivered_macro_mase": final, "baseline_macro_mase": baseline,
                                 "direction": compare(final, baseline), "wins": directions["lower"],
                                 "ties": directions["equal"], "losses": directions["higher"]},
            "review_series": [r["series_id"] for r in sorted(rows, key=lambda r: (-r["evaluation"]["delivered"]["mase"], r["series_id"]))[:3]]}


def run_experiment(panel, manifest):
    started = time.perf_counter()
    validate_panel(panel, manifest)
    rows = []
    for item in manifest["series"]:
        sid = item["series_id"]
        series = panel.loc[panel.series_id == sid].set_index("date").sort_index().turnover_aud_millions
        development, _ = split_series(series, DEVELOPMENT_END)
        candidates = backtest_candidates(development)
        selected = select_candidate(candidates)
        delivery = forecast_selected(development, selected)
        rows.append({**item, "backtest": candidates, "delivery": delivery})
    # All model choices and forecast arrays exist before any final outcomes are scored.
    for row in rows:
        series = panel.loc[panel.series_id == row["series_id"]].set_index("date").sort_index().turnover_aud_millions
        development, actual = split_series(series, DEVELOPMENT_END)
        delivered = evaluate_forecast(actual, row["delivery"]["forecast"], development)
        baseline = evaluate_forecast(actual, row["delivery"]["baseline"], development)
        row["evaluation"] = {"delivered": delivered, "baseline": baseline, "mae_direction": compare(delivered["mae"], baseline["mae"])}
    record = {"manifest": manifest, "rows": rows, "portfolio": summarize_panel(rows),
              "optimizer": {"method": "L-BFGS-B", "maxiter": 2000, "use_brute": True},
              "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels")},
              "numerical_seconds": time.perf_counter() - started}
    return FrozenJSON.of(record)


def build_evidence(experiment):
    result = experiment.data
    facts = {
        "data.source_vintage": {k: result["manifest"][k] for k in ("source_url", "release_date", "source_sha256", "csv_sha256")},
        "data.panel_manifest": result["manifest"]["series"],
        "data.development_span": ["2000-01-01", "2024-06-01"],
        "data.observation_count": {"series": 16, "months_per_series": 306, "total": 4896},
        "data.units": "nominal AUD millions",
        "protocol.validation_folds": [["2000-01", f"{y}-06", f"{y}-07", f"{y+1}-06"] for y in (2021, 2022, 2023)],
        "protocol.final_holdout": ["2024-07-01", "2025-06-01"],
        "protocol.selection_rule": "Per-series minimum mean validation RMSE over all three folds; exact ties use catalog order. Final outcomes do not change selection.",
        "protocol.scaling_rule": "MASE = MAE / mean absolute training lag-12 difference. Macro MASE weights all 16 series equally. Baseline wins use same-holdout MAE, not MASE < 1.",
        "models.catalog": CATALOG,
        "limitations.nominal_turnover": "Nominal sector turnover, not unit demand or inflation-adjusted sales; does not support inventory decisions.",
        "limitations.fixed_vintage": "One June 2025 release, not real-time vintages; original data retain pandemic disruptions. These results do not identify causes.",
        "limitations.panel_scope": "Four states and four industries, not all Australian retail; one 12-month final holdout and one portfolio reporting case.",
    }
    for row in result["rows"]:
        prefix = "series." + row["series_id"]
        selected = row["delivery"]["selected"]
        candidate = row["backtest"][selected]
        facts[prefix + ".selection"] = {"model": selected, "mean_validation_rmse": candidate["mean_rmse"],
            "baseline_mean_validation_rmse": row["backtest"]["seasonal_naive"]["mean_rmse"]}
        facts[prefix + ".backtest"] = {m: {"eligible": c["eligible"], "mean_rmse": c["mean_rmse"]} for m, c in row["backtest"].items()}
        facts[prefix + ".delivery"] = {k: row["delivery"][k] for k in ("selected", "delivered", "status", "fallback_reason")}
        facts[prefix + ".evaluation"] = row["evaluation"]
    facts.update({"portfolio." + k: v for k, v in result["portfolio"].items()})
    return FrozenJSON.of(facts)


def results_table(experiment):
    rows = []
    for r in experiment.data["rows"]:
        selected = r["delivery"]["selected"]
        rows.append({"series_id": r["series_id"], "state": r["state"], "industry": r["industry"],
                     "selected": selected, "validation_rmse": r["backtest"][selected]["mean_rmse"],
                     "delivered": r["delivery"]["delivered"], "status": r["delivery"]["status"],
                     **{k: r["evaluation"]["delivered"][k] for k in ("rmse", "mae", "mase")},
                     "baseline_mase": r["evaluation"]["baseline"]["mase"], "comparison": r["evaluation"]["mae_direction"]})
    return pd.DataFrame(rows)
