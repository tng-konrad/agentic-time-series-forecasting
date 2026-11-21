def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def load_json(path):
    return json.loads(Path(path).read_text())


def environment_record():
    return {"python": platform.python_version(), "os": platform.platform(), "architecture": platform.machine(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "nbclient", "nbformat", "ipykernel")},
            "numpy_backend": np.__config__.show(mode="dicts")}


def contract_settings():
    return {"episode_number": CFG.episode_number, "core_sha256": CFG.core_sha256,
            "stores": CFG.stores, "prefix_n": CFG.prefix_n, "horizon": CFG.horizon, "origins": CFG.origins,
            "diagnostic": [CFG.diagnostic_window, CFG.diagnostic_lag, CFG.diagnostic_alpha],
            "numerical": [CFG.annual_period, CFG.harmonics, CFG.maxiter, CFG.alpha], "method_order": CFG.method_order,
            "budgets": [CFG.max_requests, CFG.max_tools, CFG.request_seconds, CFG.workflow_seconds, CFG.max_input_characters],
            "model": [CFG.model, CFG.temperature, CFG.seed, CFG.num_ctx, CFG.num_predict, CFG.keep_alive],
            "schemas": [plan_schema(False), plan_schema(True)], "prompts": [system_prompt(False), system_prompt(True)]}


def load_panel():
    manifest = load_json(CFG.data_dir / "manifest.json")
    raw_bytes = gzip.decompress((CFG.data_dir / "retail.csv.gz").read_bytes())
    if hashlib.sha256(raw_bytes).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("Retail source checksum changed; rebuild provenance explicitly.")
    panel = pd.read_csv(CFG.data_dir / "store_daily.csv", parse_dates=["date"])
    if file_hash(CFG.data_dir / "store_daily.csv") != manifest["aggregate_sha256"]:
        raise ValueError("Store aggregation checksum changed.")
    assert panel.shape == (18260, 3)
    assert not panel.duplicated(["store", "date"]).any()
    assert not panel.isna().any().any()
    assert np.isfinite(panel.sales).all() and (panel.sales > 0).all()
    assert sorted(panel.store.unique().tolist()) == CFG.stores
    calendar = pd.date_range("2013-01-01", periods=1826, freq="D")
    for _, group in panel.groupby("store", sort=True):
        assert pd.DatetimeIndex(group.date).equals(calendar)
    return panel, manifest


def permitted_series(panel):
    # This function is the only gateway from the full panel to the executor.
    return {str(store): group.sort_values("date").iloc[:CFG.prefix_n].sales.to_numpy(float)
            for store, group in panel.groupby("store", sort=True)}


def profile_series(series):
    return {store: {"n": len(y), "mean": float(np.mean(y)),
                    "minimum": float(np.min(y)), "maximum": float(np.max(y)),
                    "acf7": float(acf(y, nlags=7, fft=True)[7])}
            for store, y in series.items()}


def fourier(days):
    # Absolute days since 2013-01-01, including when constructing future features.
    days = np.asarray(days, dtype=float)
    return np.column_stack([f(2 * np.pi * k * days / CFG.annual_period)
                            for k in range(1, CFG.harmonics + 1)
                            for f in (np.sin, np.cos)])


class AdmissionExpired(RuntimeError):
    pass


def admit(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise AdmissionExpired("No new fit/tool admitted after workflow deadline.")


def gaussian_output(mu, variance, transform):
    mu, variance = np.asarray(mu), np.asarray(variance)
    if not np.isfinite(mu).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Invalid predictive mean or variance")
    z = norm.ppf(1 - CFG.alpha / 2)
    low, high = mu - z * np.sqrt(variance), mu + z * np.sqrt(variance)
    mean = mu
    if transform == "log":
        mean, low, high = np.exp(mu + variance / 2), np.exp(low), np.exp(high)
    if not all(np.isfinite(v).all() for v in (mean, low, high)):
        raise ValueError("Nonfinite back-transformed forecast")
    return {"mean": mean.tolist(), "lower": low.tolist(), "upper": high.tolist()}


def fit_forecast(y, method, transform, work, deadline=None):
    admit(deadline)
    started = time.monotonic()
    event = {"method": method, "transform": transform, "n": len(y),
             "status": "started", "input_sha256": digest(np.asarray(y).tolist())}
    work["fits"].append(event)
    try:
        endog = np.log(y) if transform == "log" else np.asarray(y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if method == "trend":
                model = UnobservedComponents(endog, level="local linear deterministic trend")
            elif method == "bsm":
                model = UnobservedComponents(
                    endog, level="local linear trend", seasonal=7,
                    freq_seasonal=[{"period": CFG.annual_period, "harmonics": CFG.harmonics}])
            elif method == "sarimax":
                model = SARIMAX(endog, exog=fourier(np.arange(len(y))),
                                order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            else:
                raise ValueError("Unknown numerical method")
            fitted = model.fit(disp=False, method="lbfgs", maxiter=CFG.maxiter)
        event["warnings"] = [str(w.message) for w in caught]
        event["converged"] = bool(fitted.mle_retvals.get("converged", False))
        if not event["converged"]:
            raise ValueError("Optimizer did not converge")
        kwargs = {"exog": fourier(np.arange(len(y), len(y) + CFG.horizon))} if method == "sarimax" else {}
        fc = fitted.get_forecast(CFG.horizon, **kwargs)
        mu, variance = np.asarray(fc.predicted_mean), np.asarray(fc.var_pred_mean)
        gaussian_output(mu, variance, transform)  # finite-value check; not yet delivered
        innovations = np.asarray(fitted.filter_results.standardized_forecasts_error[0])
        burn = max(int(fitted.loglikelihood_burn), int(fitted.model.k_states))
        innovations = innovations[burn:][-CFG.diagnostic_window:]
        if len(innovations) != CFG.diagnostic_window or not np.isfinite(innovations).all():
            raise ValueError("Insufficient finite innovations after initialization")
        event["status"] = "completed"
        return {"mu": mu.tolist(), "variance": variance.tolist(),
                "innovations": innovations.tolist(), "parameters": np.asarray(fitted.params).tolist()}
    except Exception as exc:
        event["status"], event["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        event["seconds"] = time.monotonic() - started


def baseline_record(y):
    folds, absolute_errors = [], []
    for origin in CFG.origins:
        pred = np.resize(y[:origin][-7:], CFG.horizon)
        error = y[origin:origin + CFG.horizon] - pred
        folds.append({"origin": origin, "rmse": float(np.sqrt(np.mean(error ** 2))),
                      "prediction": pred.tolist()})
        absolute_errors.extend(np.abs(error).tolist())
    return {"status": "eligible", "rmse": float(np.mean([f["rmse"] for f in folds])),
            "folds": folds, "mean": np.resize(y[-7:], CFG.horizon).tolist(),
            "width": float(np.quantile(absolute_errors, 1 - CFG.alpha, method="linear"))}


def candidate_record(y, method, transform, work, deadline=None):
    record = {"status": "running", "folds": [], "rmse": None}
    try:
        for origin in CFG.origins:
            fit = fit_forecast(y[:origin], method, transform, work, deadline)
            pred = np.asarray(gaussian_output(fit["mu"], fit["variance"], transform)["mean"])
            error = y[origin:origin + CFG.horizon] - pred
            record["folds"].append({"origin": origin, "rmse": float(np.sqrt(np.mean(error ** 2))),
                                    "prediction": pred.tolist()})
        record["rmse"] = float(np.mean([f["rmse"] for f in record["folds"]]))
        record["validation_complete"] = True
        # Final-origin fit uses the permitted prefix only; final outcomes are absent.
        record["current"] = fit_forecast(y, method, transform, work, deadline)
        record["status"] = "eligible"
    except Exception as exc:
        record["status"] = "not_attempted_deadline" if isinstance(exc, AdmissionExpired) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["validation_complete"] = len(record["folds"]) == len(CFG.origins)
    return record


def monitor(record):
    if record["status"] != "eligible":
        return {"status": "unavailable", "reason": record.get("error", record["status"])}
    values = np.asarray(record["current"]["innovations"])
    test = acorr_ljungbox(values, lags=[CFG.diagnostic_lag], model_df=0, return_df=True)
    pvalue = float(test.lb_pvalue.iloc[0])
    correlation = acf(values, nlags=CFG.diagnostic_lag, fft=True)
    return {"status": "failed_check" if pvalue < CFG.diagnostic_alpha else "passed",
            "pvalue": pvalue, "acf7": float(correlation[7]), "acf28": float(correlation[28]),
            "statistic": float(test.lb_stat.iloc[0]), "n": len(values)}


def choose(catalog):
    eligible = [m for m in CFG.method_order if m in catalog and catalog[m]["status"] == "eligible"]
    if not eligible:
        return None
    return min(eligible, key=lambda m: (catalog[m]["rmse"], CFG.method_order.index(m)))


def deliver(catalog, selected, transform, controller, fallback_reason=None):
    chosen = selected
    if selected is None or catalog.get(selected, {}).get("status") != "eligible":
        chosen, fallback_reason = "seasonal_naive", fallback_reason or "selected_method_unavailable"
    rec = catalog.get(chosen)
    if rec is None or rec.get("status") != "eligible":
        return {"selected": selected, "delivered": None, "status": "missing", "controller": controller,
                "fallback": True, "reason": fallback_reason or "baseline_unavailable"}
    if chosen == "seasonal_naive":
        mean = np.asarray(rec["mean"])
        output = {"mean": mean.tolist(), "lower": (mean - rec["width"]).tolist(),
                  "upper": (mean + rec["width"]).tolist(), "interval_method": "empirical_validation_errors"}
    else:
        output = gaussian_output(rec["current"]["mu"], rec["current"]["variance"], transform)
        output["interval_method"] = "plug_in_gaussian" if transform == "identity" else "plug_in_lognormal"
    return dict(output, selected=selected, delivered=chosen, status="delivered", controller=controller,
                fallback=fallback_reason is not None, reason=fallback_reason)


INITIAL_TOOLS = ["check_data", "transform", "baseline", "candidates", "diagnostic", "uncertainty", "deliver"]
REVISION_TOOLS = ["repair", "diagnostic_v2", "uncertainty_v2", "deliver_v2"]


def expected_steps(revision=False):
    tools = REVISION_TOOLS if revision else INITIAL_TOOLS
    return [{"id": tool, "tool": tool,
             "depends_on": [tools[i - 1]] if i else (["diagnostic"] if revision else [])}
            for i, tool in enumerate(tools)]


def plan_schema(revision=False):
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    const = lambda v: {"type": "integer" if isinstance(v, int) else "number", "enum": [v]}
    fields = {
        "transform": {"type": "string", "enum": ["identity", "log"]},
        "horizon": const(CFG.horizon),
        "origins": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
        "candidates": {"type": "array", "items": {"type": "string", "enum": ["bsm", "sarimax"] if revision else ["trend"]},
                       "minItems": 1, "maxItems": 2 if revision else 1},
        "diagnostic": obj({"window": const(CFG.diagnostic_window), "lag": const(CFG.diagnostic_lag),
                           "alpha": const(CFG.diagnostic_alpha)}),
        "uncertainty": obj({"structural": {"type": "string", "enum": ["gaussian"]},
                            "baseline": {"type": "string", "enum": ["absolute_validation_errors"]},
                            "alpha": const(CFG.alpha)}),
        "steps": {"type": "array", "items": obj({"id": {"type": "string"}, "tool": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}}}),
                  "minItems": 4 if revision else 7, "maxItems": 4 if revision else 7},
        "reason": {"type": "string"}}
    if revision:
        fields["failed_evidence_id"] = {"type": "string"}
    return obj(fields)


def validate_plan(plan, revision=False, original=None, failed_evidence=None):
    required = set(plan_schema(revision)["required"])
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Plan fields do not match the contract")
    if plan["transform"] not in ("identity", "log") or plan["horizon"] != CFG.horizon or plan["origins"] != CFG.origins:
        raise ValueError("Invalid transform or changed chronological contract")
    if plan["diagnostic"] != {"window": CFG.diagnostic_window, "lag": CFG.diagnostic_lag, "alpha": CFG.diagnostic_alpha}:
        raise ValueError("Changed monitor contract")
    if plan["uncertainty"] != {"structural": "gaussian", "baseline": "absolute_validation_errors", "alpha": CFG.alpha}:
        raise ValueError("Changed uncertainty contract")
    allowed = {"bsm", "sarimax"} if revision else {"trend"}
    candidates = plan["candidates"]
    if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= allowed:
        raise ValueError("Invalid or duplicate candidates")
    reason = plan["reason"]
    if not isinstance(reason, str) or not 1 <= len(reason.split()) <= CFG.reason_words:
        raise ValueError("Public reason must contain 1–60 words")
    # The small allowed graph is explicit; arbitrary or cyclic task graphs are rejected.
    if plan["steps"] != expected_steps(revision):
        raise ValueError("Unknown/cyclic/missing steps or invalid prerequisites")
    if revision:
        if plan["transform"] != original["transform"]:
            raise ValueError("Revision must retain completed transformation")
        if plan["failed_evidence_id"] != failed_evidence:
            raise ValueError("Stale or missing failed-check evidence")
    return plan


def system_prompt(revision=False):
    role = "Revise the pending suffix after the recorded monitor failed." if revision else "Write the entire experiment plan before execution."
    return ("You are a forecasting experiment planner. " + role +
            " Return only the exact JSON contract. Give a brief public reason of at most 60 words, not private reasoning. "
            "The first experiment deliberately tests trend-only structure; this is an assigned teaching challenge. "
            "Code computes forecasts, checks dependencies and selects by validation RMSE. Never change cutoffs or invent measurements. "
            "Use the supplied exact required_steps, horizon, origins, diagnostic and uncertainty fields. "
            "Choose identity or log initially. After failure keep that transform and choose bsm, sarimax, or both as repair candidates. "
            "A small residual p-value signals a failed engineering check; it does not identify a cause or guarantee a repair.")


def http_json(path, payload=None, timeout=10):
    req = urllib.request.Request(CFG.endpoint + path, data=None if payload is None else canonical(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def server_identity():
    tags = http_json("/api/tags")
    found = [m for m in tags["models"] if m["name"] == CFG.model]
    if not found:
        raise RuntimeError(f"Install explicitly with ollama pull {CFG.model}; no model/provider substitution is allowed.")
    return {"model": found[0], "server": http_json("/api/version")}


def request_plan(semantic, revision, deadline, capture, run_dir, original=None, failed_id=None):
    admit(deadline)
    if len(capture["requests"]) >= CFG.max_requests:
        raise ValueError("Shared request budget exhausted")
    if len(canonical(semantic)) > CFG.max_input_characters:
        raise ValueError("Model-visible summary exceeds predeclared size cap")
    payload = {"model": CFG.model, "messages": [{"role": "system", "content": system_prompt(revision)},
                {"role": "user", "content": canonical(semantic)}], "format": plan_schema(revision),
               "stream": False, "think": False, "keep_alive": CFG.keep_alive,
               "options": {"temperature": CFG.temperature, "seed": CFG.seed,
                           "num_ctx": CFG.num_ctx, "num_predict": CFG.num_predict}}
    attempt = {"phase": "revision" if revision else "initial", "semantic_input": semantic,
               "input_sha256": digest(semantic), "payload": payload, "status": "started"}
    capture["requests"].append(attempt)
    save_json(run_dir / "capture.partial.json", capture)
    started = time.monotonic()
    try:
        raw = http_json("/api/chat", payload, timeout=min(CFG.request_seconds, max(.01, deadline-time.monotonic())))
        attempt["raw_response"] = raw
        if raw.get("done") is not True or raw.get("done_reason") != "stop":
            raise ValueError("Provider response did not complete normally")
        plan = validate_plan(json.loads(raw["message"]["content"]), revision, original, failed_id)
        attempt["plan"], attempt["status"] = plan, "accepted"
        return plan
    except Exception as exc:
        attempt["status"], attempt["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        attempt["seconds"] = time.monotonic() - started
        save_json(run_dir / "capture.partial.json", capture)


def settings_input(series, revision=False):
    return {"objective": "Forecast daily store-total demand for all ten stores; first test trend-only structure.",
            "case_order": list(series), "profile": profile_series(series), "horizon": CFG.horizon,
            "origins": CFG.origins, "diagnostic": {"window": CFG.diagnostic_window, "lag": CFG.diagnostic_lag,
                                                    "alpha": CFG.diagnostic_alpha},
            "uncertainty": {"structural": "gaussian", "baseline": "absolute_validation_errors", "alpha": CFG.alpha},
            "required_steps": expected_steps(revision),
            "candidate_catalog": {"trend": "stochastic local level + deterministic slope; no seasonality", "bsm": "local linear trend + weekly7 + annual365.25/3harmonics",
                                  "sarimax": "SARIMA(1,1,1)(1,1,1)7 + annual365.25/3harmonic regressors"},
            "permitted_input_sha256": digest({k: y.tolist() for k, y in series.items()})}


def summarize_catalog(catalogs):
    return {store: {m: {"status": r["status"], "rmse": r.get("rmse"),
                       "fold_rmse": [f["rmse"] for f in r.get("folds", [])], "sha256": digest(r)}
                    for m, r in cat.items()} for store, cat in catalogs.items()}


def initial_ledger(plan):
    return [dict(step, version=1, status="pending", output=None) for step in plan["steps"]]


def patch_ledger(ledger, revision):
    # Retain the failed event and completed prefix; supersede only the blocked suffix.
    for row in ledger:
        if row["status"] in ("pending", "blocked"):
            row["status"] = "superseded"
    ledger.extend(dict(step, version=2, status="pending", output=None) for step in revision["steps"])


def run_live(panel, protocol):
    series = permitted_series(panel)
    run_dir = CFG.evidence_dir / "runs" / CFG.run_id
    if run_dir.exists():
        raise FileExistsError("Refusing to overwrite an inference archive; choose a new A04_RUN_ID.")
    run_dir.mkdir(parents=True)
    identity = server_identity()
    if identity["model"]["digest"] != protocol["model_identity"]["model"]["digest"]:
        raise ValueError("Installed model digest differs from frozen protocol")
    started = time.monotonic()
    deadline = started + CFG.workflow_seconds
    capture = {"run_id": CFG.run_id, "execution_started_at": CFG.execution_started_at,
               "timezone": CFG.timezone, "evidence_mode": "genuine_local_inference",
               "protocol_sha256": digest(protocol), "model_identity": identity,
               "requests": [], "ledger": [], "catalogs": {s: {} for s in series},
               "work": {"fits": [], "cache_reuses": 0, "batch_tools": 0},
               "permitted_input_sha256": digest({k: y.tolist() for k, y in series.items()})}
    capture["environment"] = environment_record()
    catalogs, work = capture["catalogs"], capture["work"]
    initial_plan, revised_plan, controller = None, None, "failed_initial_plan"
    initial_deliveries, agent_deliveries = {}, {}
    initial_monitors, revised_monitors = {}, {}
    def checkpoint():
        save_json(run_dir / "capture.partial.json", capture)
    def tool(step, operation, allow_failed_prerequisite=False):
        admit(deadline)
        if work["batch_tools"] >= CFG.max_tools:
            raise AdmissionExpired("Plan tool budget exhausted")
        states = {r["id"]: r["status"] for r in capture["ledger"]}
        allowed = {"completed", "failed_check"} if allow_failed_prerequisite else {"completed"}
        if any(states.get(dep) not in allowed for dep in step["depends_on"]):
            raise ValueError("Executor prerequisite is not satisfied")
        work["batch_tools"] += 1
        step["status"] = "running"
        t = time.monotonic()
        checkpoint()
        try:
            output, status = operation()
            step["output"], step["output_sha256"], step["status"] = output, digest(output), status
        except Exception as exc:
            step["status"], step["error"] = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step["seconds"] = time.monotonic() - t
            checkpoint()
        return step["output"]
    try:
        initial_plan = request_plan(settings_input(series), False, deadline, capture, run_dir)
        capture["initial_plan"] = initial_plan
        capture["ledger"] = initial_ledger(initial_plan)
        transform = initial_plan["transform"]
        rows = capture["ledger"]
        tool(rows[0], lambda: (profile_series(series), "completed"))
        tool(rows[1], lambda: ({"transform": transform, "fitted_parameters": "none; pointwise transform"}, "completed"))
        def baselines():
            for store, y in series.items():
                admit(deadline)
                catalogs[store]["seasonal_naive"] = baseline_record(y)
            return summarize_catalog(catalogs), "completed"
        tool(rows[2], baselines)
        def candidates():
            for store, y in series.items():
                catalogs[store]["trend"] = candidate_record(y, "trend", transform, work, deadline)
                checkpoint()
            return summarize_catalog(catalogs), "completed"
        tool(rows[3], candidates)
        def diagnostics():
            initial_monitors.update({s: monitor(c["trend"]) for s, c in catalogs.items()})
            failed = any(r["status"] != "passed" for r in initial_monitors.values())
            return initial_monitors, "failed_check" if failed else "completed"
        tool(rows[4], diagnostics)
        capture["initial_monitors"] = initial_monitors
        initial_selected = {s: choose(c) for s, c in catalogs.items()}
        capture["initial_selected"] = initial_selected
        capture["initial_prefix_seconds"] = time.monotonic() - started
        # Explicit counterfactual comparator; not an executor bypass or agent success.
        initial_deliveries = {s: deliver(c, initial_selected[s], transform, "counterfactual_no_repair") for s, c in catalogs.items()}
        if rows[4]["status"] == "failed_check":
            for row in rows[5:]:
                row["status"] = "blocked"
            capture["initial_ledger"] = json.loads(canonical(rows))
            failed_id = "diagnostic:" + rows[4]["output_sha256"]
            semantic = settings_input(series, True)
            semantic.update({"initial_plan": initial_plan, "initial_plan_sha256": digest(initial_plan),
                             "failed_evidence_id": failed_id, "monitor_results": initial_monitors,
                             "completed_artifacts": {r["id"]: r.get("output_sha256") for r in rows[:5]},
                             "candidate_results": summarize_catalog(catalogs),
                             "instruction": "Preserve transform and all fixed settings. Choose seasonal repair candidates for all stores."})
            controller = "failed_replan"
            revised_plan = request_plan(semantic, True, deadline, capture, run_dir, initial_plan, failed_id)
            capture["revised_plan"] = revised_plan
            patch_ledger(rows, revised_plan)
            suffix = rows[-4:]
            def repair():
                for store, y in series.items():
                    for method in revised_plan["candidates"]:
                        catalogs[store][method] = candidate_record(y, method, transform, work, deadline)
                        checkpoint()
                work["cache_reuses"] += len(series) * 2  # retained baseline + trend candidate artifacts
                return summarize_catalog(catalogs), "completed"
            tool(suffix[0], repair, allow_failed_prerequisite=True)
            selected = {s: choose(c) for s, c in catalogs.items()}
            def recheck():
                for s, c in catalogs.items():
                    m = selected[s]
                    revised_monitors[s] = {"status": "not_applicable", "reason": "baseline has no fitted innovations"} if m == "seasonal_naive" else monitor(c[m])
                # Disposition is completed even if diagnostics are unresolved; status is explicit in output.
                return revised_monitors, "completed"
            tool(suffix[1], recheck)
            unresolved = any(r["status"] not in ("passed", "not_applicable") for r in revised_monitors.values())
            controller = "diagnostic_unresolved" if unresolved else "completed"
            tool(suffix[2], lambda: ({"structural": "plug_in_gaussian", "baseline": "empirical_validation_errors",
                                      "alpha": CFG.alpha, "disposition": controller}, "completed"))
            agent_deliveries = {s: deliver(c, selected[s], transform, controller) for s, c in catalogs.items()}
            tool(suffix[3], lambda: (agent_deliveries, "completed"))
        else:
            capture["initial_ledger"] = json.loads(canonical(rows))
            controller = "completed_without_replan"
            tool(rows[5], lambda: ({"alpha": CFG.alpha, "disposition": "passed"}, "completed"))
            agent_deliveries = {s: deliver(c, initial_selected[s], transform, controller) for s, c in catalogs.items()}
            tool(rows[6], lambda: (agent_deliveries, "completed"))
    except Exception as exc:
        controller = "failed_deadline" if isinstance(exc, AdmissionExpired) else controller
        capture["controller_error"] = f"{type(exc).__name__}: {exc}"
        for row in capture["ledger"]:
            if row["status"] in ("pending", "running"):
                row["status"] = "blocked"
        t = time.monotonic()
        for s, y in series.items():
            # Explicit no-fit safety exception; never calls a candidate after expiry.
            if "seasonal_naive" not in catalogs[s]:
                catalogs[s]["seasonal_naive"] = baseline_record(y)
            selected = choose(catalogs[s]) if initial_plan else "seasonal_naive"
            agent_deliveries[s] = deliver(catalogs[s], selected, initial_plan["transform"] if initial_plan else "identity",
                                           controller, "controller_failure")
        capture["safety_finalization_seconds"] = time.monotonic() - t
    capture.update(controller_status=controller, initial_monitors=initial_monitors,
                   revised_monitors=revised_monitors, agent_seconds=time.monotonic()-started,
                   admission_overshoot_seconds=max(0., time.monotonic()-deadline))
    arms = {"seasonal_naive": {}, "initial_plan": initial_deliveries, "agent_plan": agent_deliveries, "fixed_bsm": {}}
    fixed_work = {"fits": [], "cache_reuses": 0, "batch_tools": 0}
    base_t = time.monotonic()
    for s, y in series.items():
        arms["seasonal_naive"][s] = deliver({"seasonal_naive": baseline_record(y)}, "seasonal_naive", "identity", "fixed_baseline")
    capture["baseline_seconds"] = time.monotonic()-base_t
    fixed_t = time.monotonic()
    fixed_catalogs = {}
    for s, y in series.items():
        cat = {"seasonal_naive": baseline_record(y), "bsm": candidate_record(y, "bsm", "log", fixed_work)}
        fixed_catalogs[s] = cat
        arms["fixed_bsm"][s] = deliver(cat, choose(cat), "log", "fixed_plan")
        checkpoint()
    capture["fixed_seconds"], capture["fixed_work"] = time.monotonic()-fixed_t, fixed_work
    capture["fixed_catalogs"], capture["arms"] = fixed_catalogs, arms
    capture["terminal_record_complete"] = True
    capture["forecast_freeze_sha256"] = digest(arms)
    save_json(run_dir / "capture.json", capture)
    save_json(run_dir / "frozen_forecasts.json", {"arms": arms, "sha256": digest(arms)})
    save_json(run_dir / "freeze.json", {"run_id": CFG.run_id, "capture_sha256": file_hash(run_dir/"capture.json"),
                                       "forecasts_sha256": digest(arms), "protocol_sha256": digest(protocol),
                                       "frozen_at": datetime.now(ZoneInfo(CFG.timezone)).isoformat()})
    save_json(CFG.evidence_dir / "latest.json", {"run_id": CFG.run_id})
    return capture, run_dir


def verify_archive(panel, protocol, run_dir):
    freeze = load_json(run_dir / "freeze.json")
    capture = load_json(run_dir / "capture.json")
    forecasts = load_json(run_dir / "frozen_forecasts.json")
    if not capture.get("terminal_record_complete"):
        raise ValueError("Incomplete archive cannot be replayed")
    checks = [file_hash(run_dir / "capture.json") == freeze["capture_sha256"],
              digest(protocol) == freeze["protocol_sha256"] == capture["protocol_sha256"],
              digest(capture["arms"]) == freeze["forecasts_sha256"] == forecasts["sha256"],
              forecasts["arms"] == capture["arms"],
              digest({k: y.tolist() for k, y in permitted_series(panel).items()}) == capture["permitted_input_sha256"],
              protocol["core_sha256"] == CFG.core_sha256]
    if "contract" in protocol:
        checks.append(protocol["contract"] == contract_settings())
    for attempt in capture["requests"]:
        checks.extend([attempt["input_sha256"] == digest(attempt["semantic_input"]),
                       attempt["payload"]["messages"][1]["content"] == canonical(attempt["semantic_input"]),
                       attempt["payload"]["messages"][0]["content"] == system_prompt(attempt["phase"] == "revision"),
                       attempt["payload"]["format"] == plan_schema(attempt["phase"] == "revision")])
    if not all(checks):
        raise ValueError("Replay identity mismatch: use the archived inputs/protocol/code, never silently recapture.")
    return capture


def obtain_study(panel, protocol):
    if CFG.mode == "live":
        return run_live(panel, protocol)
    if CFG.mode != "replay":
        raise ValueError("A04_MODE must be live or replay")
    run_dir = CFG.evidence_dir / "runs" / load_json(CFG.evidence_dir/"latest.json")["run_id"]
    return verify_archive(panel, protocol, run_dir), run_dir


def score_study(panel, capture, run_dir):
    freeze = load_json(run_dir / "freeze.json")
    audit = load_json(run_dir / "audit.json")
    if audit["capture_sha256"] != freeze["capture_sha256"] or audit["forecast_sha256"] != freeze["forecasts_sha256"]:
        raise ValueError("Author audit must bind this capture and frozen forecasts before scoring.")
    records = []
    for arm, deliveries in capture["arms"].items():
        for store in CFG.stores:
            s = str(store)
            delivery = deliveries.get(s, {"status": "missing", "reason": "no_valid_initial_plan"})
            row = {"arm": arm, "store": store, "delivery": delivery["status"],
                   "selected": delivery.get("selected"), "delivered": delivery.get("delivered"),
                   "fallback": delivery.get("fallback"), "controller": delivery.get("controller"),
                   "reason": delivery.get("reason")}
            if delivery["status"] == "delivered":
                y = panel[panel.store == store].sort_values("date").sales.to_numpy(float)
                actual = y[CFG.prefix_n:]
                mean, lower, upper = [np.asarray(delivery[k]) for k in ("mean", "lower", "upper")]
                assert len(actual) == len(mean) == CFG.horizon
                error = actual-mean
                scale = float(np.mean(np.abs(y[7:CFG.prefix_n]-y[:CFG.prefix_n-7])))
                mae = float(np.mean(np.abs(error)))
                score = upper-lower + 2/CFG.alpha*(lower-actual)*(actual < lower) + 2/CFG.alpha*(actual-upper)*(actual > upper)
                row.update(rmse=float(np.sqrt(np.mean(error**2))), mae=mae,
                           mase=mae/scale if scale > 0 else None, mase_reason=None if scale > 0 else "zero seasonal scale",
                           coverage=float(np.mean((actual >= lower) & (actual <= upper))),
                           width=float(np.mean(upper-lower)), interval_score=float(np.mean(score)))
            records.append(row)
    summaries = []
    for arm in capture["arms"]:
        rows = [r for r in records if r["arm"] == arm and r["delivery"] == "delivered"]
        summaries.append({"arm": arm, "delivered": len(rows), "eligible_cases": [r["store"] for r in rows],
                          "denominator": len(CFG.stores),
                          "rmse": float(np.sqrt(np.mean([r["rmse"]**2 for r in rows]))) if rows else None,
                          **{k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
                             for k in ("mae", "mase", "coverage", "width", "interval_score")}})
    result = {"run_id": capture["run_id"], "capture_sha256": freeze["capture_sha256"],
              "audit_sha256": file_hash(run_dir/"audit.json"), "records": records, "summaries": summaries,
              "scored_at": datetime.now(ZoneInfo(CFG.timezone)).isoformat()}
    path = run_dir / "results.json"
    if path.exists():
        old = load_json(path)
        if old["records"] != records or old["summaries"] != summaries:
            raise ValueError("Frozen results changed")
        return old
    save_json(path, result)
    return result
