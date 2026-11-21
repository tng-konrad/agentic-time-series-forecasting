from pathlib import Path
import json
import time
import nbformat

ROOT = Path(__file__).resolve().parents[2]


def namespace():
    nb = nbformat.read(ROOT / "a04-plan-and-execute.ipynb", as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.source.startswith("panel, data_manifest"):
                break
            exec(compile(cell.source, "<inline-notebook>", "exec"), ns)
    return ns


if __name__ == "__main__":
    n = namespace()
    cfg, np = n["CFG"], n["np"]
    checks = []
    def check(name, summary, function):
        t = time.monotonic()
        function()
        checks.append({"name": name, "summary": summary, "status": "passed", "seconds": time.monotonic()-t})
        print("PASS", name, flush=True)
    def rejects(function, exception=ValueError):
        try:
            function()
        except exception:
            return
        raise AssertionError("Expected rejection")
    panel, manifest = n["load_panel"]()
    allowed = n["permitted_series"](panel)
    def boundary():
        changed = panel.copy(deep=True)
        changed.loc[changed.date >= "2017-10-03", "sales"] += 100000
        other = n["permitted_series"](changed)
        assert all(np.array_equal(allowed[s], other[s]) for s in allowed)
        assert n["settings_input"](allowed) == n["settings_input"](other)
        assert n["baseline_record"](allowed["1"]) == n["baseline_record"](other["1"])
        # Actual structural predictions, not merely file hashes.
        a = n["fit_forecast"](allowed["1"][:1466], "trend", "log", {"fits": []})
        b = n["fit_forecast"](other["1"][:1466], "trend", "log", {"fits": []})
        assert a == b
        other["1"][0] += 1
        assert n["digest"](n["settings_input"](allowed)) != n["digest"](n["settings_input"](other))
    check("Future isolation", "Mutated final values leave requests and real trend/baseline forecasts unchanged", boundary)
    initial = {"transform": "log", "horizon": cfg.horizon, "origins": cfg.origins,
               "candidates": ["trend"], "diagnostic": {"window": 364, "lag": 28, "alpha": .05},
               "uncertainty": {"structural": "gaussian", "baseline": "absolute_validation_errors", "alpha": .2},
               "steps": n["expected_steps"](), "reason": "Start with the assigned simple structure."}
    clone = lambda obj: json.loads(json.dumps(obj))
    def contracts():
        n["validate_plan"](initial)
        for edit in [lambda p: p["steps"].pop(), lambda p: p["steps"][0].update(depends_on=["deliver"]),
                     lambda p: p.update(origins=[1466, 1556, 1736]), lambda p: p.update(candidates=["trend", "trend"]),
                     lambda p: p["steps"][3].update(tool="arbitrary_code")]:
            bad = clone(initial); edit(bad); rejects(lambda: n["validate_plan"](bad))
    check("Plan contracts", "Unknown tools, cycles, missing steps and changed cutoffs rejected", contracts)
    revised = clone(initial)
    revised.update(candidates=["bsm"], steps=n["expected_steps"](True), failed_evidence_id="diagnostic:test")
    def revision():
        n["validate_plan"](revised, True, initial, "diagnostic:test")
        rejects(lambda: n["validate_plan"](revised, True, initial, "diagnostic:stale"))
        bad = clone(revised); bad["transform"] = "identity"
        rejects(lambda: n["validate_plan"](bad, True, initial, "diagnostic:test"))
        ledger = n["initial_ledger"](initial)
        for r in ledger[:4]: r["status"] = "completed"
        ledger[4]["status"] = "failed_check"
        for r in ledger[5:]: r["status"] = "blocked"
        old = clone(ledger[:5]); n["patch_ledger"](ledger, revised)
        assert ledger[:5] == old and all(r["status"] == "superseded" for r in ledger[5:7])
        assert len(ledger) == 11
    check("Revision history", "Stale evidence rejected; failed event retained and blocked suffix superseded", revision)
    def phase():
        a = n["fourier"](np.arange(1466 + 90))
        b = np.vstack([n["fourier"](np.arange(1466)), n["fourier"](np.arange(1466, 1466+90))])
        assert np.array_equal(a, b)
        assert not np.allclose(a[1466], a[0])
        assert np.allclose(n["fourier"]([0]), n["fourier"]([365.25]), atol=1e-12)
    check("Fourier alignment", "Future annual phases continue from absolute training dates", phase)
    def transformations():
        out = n["gaussian_output"]([np.log(100)], [1.], "log")
        assert np.isclose(out["mean"][0], 100*np.exp(.5))
        assert out["lower"][0] < 100 < out["mean"][0] < out["upper"][0]
        rejects(lambda: n["gaussian_output"]([0], [-1], "log"))
        y = np.arange(30.)
        assert np.mean(np.abs(y[7:]-y[:-7])) == 7
        assert np.mean(np.abs(np.diff(y, n=7))) == 0
    check("Means and scales", "Lognormal mean/quantiles and seven-day MASE lag validated", transformations)
    def deadline():
        work = {"fits": []}
        r = n["candidate_record"](allowed["1"], "trend", "log", work, time.monotonic()-1)
        assert r["status"] == "not_attempted_deadline" and work["fits"] == []
        rejects(lambda: n["request_plan"]({}, False, time.monotonic()+1, {"requests": [1, 2]}, ROOT/"examples/module04/development"))
    check("Admission and budgets", "Expiry admits no fit; exhausted shared requests reject without HTTP", deadline)
    def fit_failure():
        old = n["UnobservedComponents"]
        class FailedFit:
            mle_retvals = {"converged": False}
            def fit(self, **kwargs): return self
        n["UnobservedComponents"] = lambda *args, **kwargs: FailedFit()
        try:
            work = {"fits": []}
            rec = n["candidate_record"](allowed["1"], "trend", "log", work)
            assert rec["status"] == "failed" and work["fits"][0]["status"] == "failed"
            baseline = n["baseline_record"](allowed["1"])
            assert n["choose"]({"seasonal_naive": baseline, "trend": rec}) == "seasonal_naive"
            d = n["deliver"]({"seasonal_naive": baseline, "trend": rec}, "trend", "log", "failed_fit")
            assert d["selected"] == "trend" and d["delivered"] == "seasonal_naive" and d["fallback"]
        finally:
            n["UnobservedComponents"] = old
    check("Fit failure and fallback", "Nonconverged fit excluded; selected and fallback delivery stay distinct", fit_failure)
    def transport_failure():
        old = n["http_json"]
        def timeout(*args, **kwargs): raise TimeoutError("scripted transport timeout")
        n["http_json"] = timeout
        try:
            cap = {"requests": []}
            rejects(lambda: n["request_plan"]({}, False, time.monotonic()+5, cap, ROOT/"examples/module04/development"), TimeoutError)
            assert len(cap["requests"]) == 1 and cap["requests"][0]["status"] == "failed"
            assert "raw_response" not in cap["requests"][0]
        finally:
            n["http_json"] = old
    check("Transport accounting", "One timeout consumes one attempt, without retry or fabricated response", transport_failure)
    def integrity():
        d = ROOT/"examples/module04/development/integrity-fixture"; d.mkdir(exist_ok=True)
        proto = {"core_sha256": cfg.core_sha256}
        cap = {"terminal_record_complete": True, "protocol_sha256": n["digest"](proto),
               "arms": {"fixture": {}}, "permitted_input_sha256": n["digest"]({k: y.tolist() for k, y in allowed.items()}), "requests": []}
        n["save_json"](d/"capture.json", cap)
        n["save_json"](d/"frozen_forecasts.json", {"arms": cap["arms"], "sha256": n["digest"](cap["arms"])})
        n["save_json"](d/"freeze.json", {"capture_sha256": n["file_hash"](d/"capture.json"),
                                             "protocol_sha256": n["digest"](proto), "forecasts_sha256": n["digest"](cap["arms"])})
        n["verify_archive"](panel, proto, d)
        cap["arms"]["fixture"]["changed"] = True; n["save_json"](d/"capture.json", cap)
        rejects(lambda: n["verify_archive"](panel, proto, d))
        rejects(lambda: n["verify_archive"](panel, proto, d/"missing"), FileNotFoundError)
    check("Archive integrity", "Changed capture or missing archive fails closed; no model fallback", integrity)
    n["save_json"](ROOT/"examples/module04/development/checks.json", {"evidence_mode": "development_only", "checks": checks})
    # Full-length synthetic timing/stimulus pilot, excluded from the ten-store cohort.
    rng = np.random.default_rng(42); days = np.arange(cfg.prefix_n)
    synthetic = np.exp(7.5 + .0001*days + .2*np.sin(2*np.pi*days/7) + .15*np.sin(2*np.pi*days/365.25) + rng.normal(0, .035, len(days)))
    pilot = {"evidence_mode": "synthetic_development_only", "n": len(days), "records": {}, "work": {"fits": []}}
    for method in ["trend", "bsm", "sarimax"]:
        t = time.monotonic()
        try:
            fit = n["fit_forecast"](synthetic, method, "log", pilot["work"])
            pilot["records"][method] = {"status": "completed", "monitor": n["monitor"]({"status": "eligible", "current": fit})}
        except Exception as exc:
            pilot["records"][method] = {"status": "failed", "error": str(exc)}
        pilot["records"][method]["seconds"] = time.monotonic()-t
        print("PILOT", method, pilot["records"][method], flush=True)
    n["save_json"](ROOT/"examples/module04/development/numerical-pilot.json", pilot)
    assert pilot["records"]["trend"]["monitor"]["status"] == "failed_check"
    prototypes = []
    for number, columns, rows, widths in [
        (2, ["Step", "Requires", "Operation"], [[r["id"], ",".join(r["depends_on"]) or "none", r["tool"]] for r in initial["steps"]], [.29, .34, .37]),
        (7, ["Arm", "Delivered", "RMSE", "MASE", "Coverage", "Width", "Int. score"], [["Synthetic prototype", "10/10", "123.4", "0.987", "80.0%", "200.1", "330.2"]]*4, [.2,.13,.13,.12,.14,.13,.15])]:
        fig, shown = n["table_figure"](columns, rows, widths)
        path = ROOT/f"examples/module04/development/atsf04_table_{number:02d}.png"
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white"); n["plt"].close(fig)
        prototypes.append({"filename": str(path.relative_to(ROOT)), "sha256": n["file_hash"](path), "evidence_mode": "synthetic_layout_prototype"})
    n["save_json"](ROOT/"examples/module04/development/draft-manifest.json", {"assets": prototypes})
