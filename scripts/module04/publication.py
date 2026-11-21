SECTIONS = ["Why write the experiment before running it?", "Put the future behind a boundary",
            "Give the planner an executable contract", "Execute until a check fails",
            "Revise the remaining work", "Freeze forecasts, then open the outcome",
            "Check the controller when things go wrong", "Closing time"]

VISUALS = [
    (1, "graph", SECTIONS[1], "Store 1 before the forecast origin", "Permitted daily sales for store 1; shading marks the three 90-day validation blocks. No final outcomes are shown."),
    (2, "table", SECTIONS[2], "The initial executable plan", "The genuine initial plan, including each prerequisite. Transformation and other settings are recorded in the full plan."),
    (3, "graph", SECTIONS[3], "What the trend model leaves behind", "Store 1 standardized one-step innovation autocorrelations over the last 364 permitted days. The dashed bounds are descriptive approximations."),
    (4, "table", SECTIONS[3], "Execution pauses at the failed check", "Initial shared-plan status before revision. The monitor tests the trend candidate, independently of validation selection."),
    (5, "table", SECTIONS[4], "The revised suffix and its outcome", "The accepted revision replaces pending work and retains the completed prefix and failed event. Full per-store monitor values are archived."),
    (6, "graph", SECTIONS[5], "Ninety days after the forecast freeze", "Predeclared store 1: frozen point forecasts and the agent's nominal 80% marginal interval against the final 90 daily outcomes."),
    (7, "table", SECTIONS[5], "Forecast error and uncertainty across ten stores", "All four arms: delivery counts out of ten, pooled daily RMSE, equal-store mean MASE, and mean interval diagnostics over delivered cases. Full 40-row scores retain individual statuses and MAE."),
    (8, "table", SECTIONS[5], "Actual work in the genuine run", "Original measured agent and comparator workloads, excluding synthetic development and publication export. Retained artifacts are logical reuse, not new fits."),
    (9, "table", SECTIONS[6], "Software checks outside the live cohort", "Deterministic and numerical development checks. These do not estimate live model failure probabilities.")]


def table_figure(columns, rows, widths=None):
    widths = widths or [1 / len(columns)] * len(columns)
    wrap_limits = [max(9, int(76 * w)) for w in widths]
    shown = [[textwrap.fill(str(v), width=wrap_limits[j], break_long_words=True) for j, v in enumerate(row)] for row in rows]
    heights = [max(len(cell.splitlines()) for cell in row) for row in shown]
    height = .75 + sum(max(.47, .26 * h + .12) for h in heights)
    fig, ax = plt.subplots(figsize=(CFG.figure_width, height))
    ax.axis("off")
    table = ax.table(cellText=shown, colLabels=columns, colWidths=widths, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(CFG.table_font)
    total_height = sum(max(.47, .26 * h + .12) for h in heights) + .6
    for (i, j), cell in table.get_celld().items():
        cell.get_text().set_color("black")
        cell.set_edgecolor("#FFFFFF")
        cell.set_facecolor("#D8E5EF" if i == 0 else ("#EEF3F7" if i % 2 else "#FFFFFF"))
        cell.set_height((.6 if i == 0 else max(.47, .26 * heights[i-1] + .12))/total_height*.94)
        if i == 0:
            cell.get_text().set_weight("bold")
    return fig, shown


def export_visual(number, kind, data, fig, capture, result, columns=None, rows=None, displayed=None):
    spec = next(v for v in VISUALS if v[0] == number)
    filename = f"{CFG.asset_prefix}{kind}_{number:02d}.png"
    path = CFG.graph_dir / filename
    sidecar = CFG.export_dir / f"{filename[:-4]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve previous stable-name exports before replacing them.
    if path.exists():
        old = CFG.evidence_dir / "exports" / "archived" / CFG.export_id
        old.mkdir(parents=True, exist_ok=True)
        (old / filename).write_bytes(path.read_bytes())
    fig.savefig(path, dpi=CFG.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_json(sidecar, {"visual": number, "data": data, "columns": columns, "rows": rows, "displayed_strings": displayed})
    entry = {"episode_number": CFG.episode_number, "asset_prefix": CFG.asset_prefix,
             "visual_number": f"{number:02d}", "type": kind, "filename": "graphs/" + filename,
             "section": spec[2], "title": spec[3], "caption": spec[4],
             "alt_text": f"{kind.capitalize()} {number:02d}: {spec[3]}. {spec[4]}",
             "sidecar": str(sidecar.relative_to(CFG.root)), "sidecar_sha256": file_hash(sidecar),
             "png_sha256": file_hash(path), "source_result_sha256": digest(data),
             "source_capture_sha256": file_hash(CFG.run_dir/"capture.json"),
             "execution_started_at": CFG.execution_started_at, "timezone": CFG.timezone,
             "inference_run_id": capture["run_id"], "export_run_id": CFG.export_id,
             "dpi": CFG.dpi, "figure_width_inches": CFG.figure_width,
             "pixel_dimensions": list(plt.imread(path).shape[:2][::-1]),
             "visual_review": "pending"}
    CFG.assets.append(entry)
    display(Image(filename=str(path)))
    return entry


def make_visual(number, panel, capture, result, tests):
    series = permitted_series(panel)
    data, columns, rows, shown = {}, None, None, None
    kind = next(v[1] for v in VISUALS if v[0] == number)
    if number == 1:
        y = series["1"]
        dates = pd.date_range("2013-01-01", periods=len(y), freq="D")
        fig, ax = plt.subplots(figsize=(CFG.figure_width, CFG.figure_height))
        ax.plot(dates, y, lw=.8, color="#356A9A", label="Permitted sales")
        for i, n in enumerate(CFG.origins):
            ax.axvspan(dates[n], dates[n+CFG.horizon-1], color=["#C9DCEB", "#E5DBC4", "#D6E8D2"][i], alpha=.6,
                       label=f"Validation {i+1}")
        ax.set(xlabel="Date", ylabel="Store 1 sales (units/day)")
        ax.legend(fontsize=10, loc="upper left")
        data = {"date": dates.astype(str).tolist(), "sales": y.tolist(), "origins": CFG.origins}
    elif number == 2:
        plan = capture.get("initial_plan")
        columns = ["Step", "Requires", "Operation"]
        rows = [[s["id"], ", ".join(s["depends_on"]) or "none", s["tool"]] for s in plan["steps"]] if plan else [["none", "none", "initial planner failed; see raw capture"]]
        data = plan
    elif number == 3:
        record = capture["catalogs"]["1"].get("trend")
        fig, ax = plt.subplots(figsize=(CFG.figure_width, CFG.figure_height))
        if record and record["status"] == "eligible":
            values = np.asarray(record["current"]["innovations"])
            corr = acf(values, nlags=28, fft=True)[1:]
            ax.bar(np.arange(1, 29), corr, color="#356A9A")
            bound = 1.96/np.sqrt(len(values))
            ax.axhline(bound, color="#A3463C", ls="--", label="Approximate ±1.96/√n")
            ax.axhline(-bound, color="#A3463C", ls="--")
            ax.axhline(0, color="black", lw=.7)
            data = {"acf": corr.tolist(), "n": len(values), "monitor": capture["initial_monitors"]["1"]}
        else:
            ax.text(.5, .5, "Trend fit unavailable; see archived failure", transform=ax.transAxes, ha="center")
        ax.set(xlabel="Lag (days)", ylabel="Innovation autocorrelation", xticks=[1, 7, 14, 21, 28])
        ax.legend()
    elif number in (4, 5):
        columns = ["Step", "Requires", "Status", "Recorded result"]
        ledger = capture.get("initial_ledger", capture["ledger"]) if number == 4 else [r for r in capture["ledger"] if r["version"] == 2]
        notes = {"check_data": "10 stores; 1,736 days each", "transform": capture.get("initial_plan", {}).get("transform", "unavailable"),
                 "baseline": "Weekly seasonal naive; 3 folds", "candidates": "Trend candidates evaluated", "diagnostic": f"{sum(v['status'] == 'failed_check' for v in capture['initial_monitors'].values())}/10 failed trend checks",
                 "uncertainty": "Waiting for monitor disposition", "deliver": "Waiting for monitor disposition",
                 "repair": ", ".join(capture.get("revised_plan", {}).get("candidates", [])),
                 "diagnostic_v2": f"{sum(v['status'] == 'failed_check' for v in capture['revised_monitors'].values())}/10 unresolved checks",
                 "uncertainty_v2": "80% plug-in or empirical intervals", "deliver_v2": capture["controller_status"]}
        rows = [[r["id"], ", ".join(r["depends_on"]) or "none", r["status"], notes.get(r["id"], "see archive")] for r in ledger]
        if not rows:
            rows = [["No revision", "see trace", capture["controller_status"], "No genuine revision claimed"]]
        data = ledger
    elif number == 6:
        observed = panel[panel.store == 1].sort_values("date")
        dates = pd.DatetimeIndex(observed.date.iloc[CFG.prefix_n:])
        fig, ax = plt.subplots(figsize=(CFG.figure_width, CFG.figure_height))
        ax.plot(observed.date.iloc[CFG.prefix_n-60:], observed.sales.iloc[CFG.prefix_n-60:], color="#333333", lw=1.5, label="Observed")
        style = {"seasonal_naive": ("#888888", ":", "s"), "initial_plan": ("#C47A30", "--", "x"),
                 "fixed_bsm": ("#8A54A2", "--", "^"), "agent_plan": ("#287B61", "-", "o")}
        for arm in ["seasonal_naive", "initial_plan", "fixed_bsm", "agent_plan"]:
            rec = capture["arms"][arm].get("1", {})
            if rec.get("status") == "delivered":
                color, ls, marker = style[arm]
                ax.plot(dates, rec["mean"], color=color, ls=ls, lw=1.3, marker=marker, markevery=9, ms=4, label=arm.replace("_", " "))
                if arm == "agent_plan":
                    ax.fill_between(dates, rec["lower"], rec["upper"], color=color, alpha=.15, label="Agent nominal 80%")
        ax.axvline(dates[0], color="#333333", lw=1, ls=":")
        ax.set(xlabel="Date", ylabel="Store 1 sales (units/day)")
        ax.legend(fontsize=9, ncol=2)
        data = {"date": observed.date.astype(str).tolist(), "sales": observed.sales.tolist(),
                "arms": {a: v.get("1") for a, v in capture["arms"].items()}}
    elif number == 7:
        columns = ["Arm", "Delivered", "RMSE", "MASE", "Coverage", "Width", "Int. score"]
        data = result["summaries"]
        rows = [[r["arm"].replace("_", " "), f"{r['delivered']}/10", f"{r['rmse']:.1f}" if r['rmse'] is not None else "N/A",
                 f"{r['mase']:.3f}" if r['mase'] is not None else "N/A", f"{r['coverage']:.1%}" if r['coverage'] is not None else "N/A",
                 f"{r['width']:.1f}" if r['width'] is not None else "N/A", f"{r['interval_score']:.1f}" if r['interval_score'] is not None else "N/A"] for r in data]
    elif number == 8:
        columns = ["Workload", "Fits", "Requests", "Time", "Output tokens", "Retained"]
        tokens = sum(r.get("raw_response", {}).get("eval_count", 0) for r in capture["requests"])
        data = [{"workload": "Shared agent", "fits": len(capture["work"]["fits"]), "requests": len(capture["requests"]),
                 "seconds": capture["agent_seconds"], "output_tokens": tokens, "retained": capture["work"]["cache_reuses"]},
                {"workload": "Fixed BSM", "fits": len(capture["fixed_work"]["fits"]), "requests": 0,
                 "seconds": capture["fixed_seconds"], "output_tokens": 0, "retained": 0},
                {"workload": "Seasonal naive", "fits": 0, "requests": 0,
                 "seconds": capture["baseline_seconds"], "output_tokens": 0, "retained": 0}]
        rows = [[r["workload"], r["fits"], r["requests"],
                 f"{r['seconds']*1000:.2f} ms" if r["seconds"] < .01 else f"{r['seconds']:.2f} s",
                 r["output_tokens"], r["retained"]] for r in data]
    else:
        columns = ["Check", "Outcome", "Evidence"]
        rows = [[r["name"], r["status"], r["summary"]] for r in tests["checks"]]
        data = tests
    if kind == "table":
        widths = {2: [.29, .34, .37], 4: [.2, .22, .17, .41], 5: [.2, .22, .17, .41],
                  7: [.2, .13, .13, .12, .14, .13, .15], 8: [.23, .1, .14, .15, .22, .16], 9: [.25, .12, .63]}[number]
        fig, shown = table_figure(columns, rows, widths)
    return export_visual(number, kind, data, fig, capture, result, columns, rows, shown)


def finish_exports(capture, result):
    manifest = {"episode_number": CFG.episode_number, "asset_prefix": CFG.asset_prefix,
                "execution_started_at": CFG.execution_started_at, "timezone": CFG.timezone,
                "inference_run_id": capture["run_id"], "export_id": CFG.export_id,
                "replay_scope": "analysis-only; does not recompute tools or forecasts",
                "assets": CFG.assets, "result_records_sha256": digest(result["records"])}
    save_json(CFG.export_dir / "asset-manifest.json", manifest)
    save_json(CFG.evidence_dir / "asset-manifest.json", manifest)
    return manifest
