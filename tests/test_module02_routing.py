"""Contract, causal-boundary, selection and recovery tests using labeled fixtures."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agentic_ts import module02_workflow as w
from agentic_ts import routing_forecasting as n
from agentic_ts.module01_assets import format_cell
from agentic_ts.module02_publication import shape_valid


class FixtureClient:
    def __init__(self, route="regular", raw=None):
        self.calls = 0
        self.raw = raw or {"text": json.dumps({"route": route, "evidence_ids": ["history.available_span", "history.current_window"],
                                              "reason": "The supplied history supports this workflow.",
                                              "clarification_question": "Which metadata is missing?" if route == "clarify" else None}),
                           "usage": {}, "metadata": {"fixture": True}}
    def complete(self, messages, schema):
        self.calls += 1
        return deepcopy(self.raw)


class Module02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.source, cases = w.load_inputs()
        cls.cases = {c["case_id"]: c for c in cases}

    def view(self, cid="M02-F01", panel=None):
        return w.request_view(self.cases[cid], self.panel if panel is None else panel, self.source)

    def test_suite_and_temporal_windows(self):
        self.assertEqual(len(self.cases), 20)
        self.assertEqual(len(w.as_series(self.view()["history"])), 449)
        short = w.as_series(self.view("M02-S01")["history"])
        self.assertEqual(len(short), 60)
        self.assertEqual(short.index[0], pd.Timestamp("2016-01-28"))
        dates = [(train.index[-1].date().isoformat(), valid.index[-1].date().isoformat())
                 for train, valid in n.validation_windows(w.as_series(self.view()["history"]))]
        self.assertEqual(dates, [("2016-01-03", "2016-01-31"), ("2016-01-31", "2016-02-28"), ("2016-02-28", "2016-03-27")])

    def test_four_routes_and_missing_metadata(self):
        for cid, route in [("M02-F01", "regular"), ("M02-F05", "intermittent"), ("M02-S01", "short_history"),
                           ("M02-N01", "new_product"), ("M02-X01", "clarify"), ("M02-X02", "clarify")]:
            self.assertEqual(w.rule_route(w.profile(self.view(cid))), route)

    def test_gates_make_no_model_calls(self):
        for cid, expected in [("M02-X03", "clarify"), ("M02-X04", "invalid_input")]:
            client = FixtureClient()
            self.assertEqual(w.choose("C", self.view(cid), mode="injected", client=client)["status"], expected)
            self.assertEqual(client.calls, 0)
        view = self.view(); del view["request"]["horizon"]
        self.assertEqual(w.gate(view)["status"], "clarify")

    def test_history_boundary_and_integer_policy(self):
        base = w.profile(self.view())
        for count, route in [(55, "clarify"), (56, "short_history"), (448, "short_history"), (449, "regular")]:
            e = deepcopy(base); e["history.available_span"]["observations"] = count
            self.assertEqual(w.rule_route(e), route)
        e = deepcopy(base); e["history.current_window"].update(observed_count=100, zero_count=50)
        self.assertEqual(w.rule_route(e), "intermittent")
        e["history.current_window"]["zero_count"] = 49
        self.assertEqual(w.rule_route(e), "regular")
        for count in (55, 56):
            view = self.view(); view["history"]["dates"] = view["history"]["dates"][-count:]
            view["history"]["values"] = view["history"]["values"][-count:]
            view["request"]["visible_start"] = view["history"]["dates"][0]
            self.assertEqual(w.choose("A", view)["executed_route"], None if count == 55 else "seasonal_naive")

    def test_zero_and_empty_are_distinct(self):
        view = self.view(); view["history"]["values"] = [0.] * 449
        self.assertEqual(w.rule_route(w.profile(view)), "intermittent")
        e = w.profile(self.view("M02-N01"))
        self.assertEqual(e["history.available_span"]["observations"], 0)
        self.assertIsNone(e["history.current_window"]["mean_sales"])
        self.assertEqual(w.rule_route(e), "new_product")
        view["request"]["lifecycle"] = "new"
        self.assertEqual(w.rule_route(w.profile(view)), "clarify")

    def test_holdout_and_masked_history_cannot_enter_inputs(self):
        changed = self.panel.copy()
        changed.loc[changed.date > pd.Timestamp(w.ORIGIN), "sales_units"] += 1000
        for cid in ("M02-F01", "M02-S01", "M02-N01"):
            self.assertEqual(self.view(cid), self.view(cid, changed))
        changed = self.panel.copy()
        mask = (changed.series_id == self.cases["M02-S01"]["series_id"]) & (changed.date < pd.Timestamp("2016-01-28"))
        changed.loc[mask, "sales_units"] += 1000
        for cid in ("M02-S01", "M02-N01"):
            self.assertEqual(self.view(cid), self.view(cid, changed))
        view = self.view("M02-N01"); other = deepcopy(view)
        first = next(iter(other["donors"]))
        other["donors"][first]["history"]["values"][0] += 1
        self.assertNotEqual(w.inputs(view, w.profile(view))[1], w.inputs(other, w.profile(other))[1])

    def test_median_prior_sensitivity_and_hand_calculation(self):
        dates = pd.date_range("2016-02-29", "2016-03-27")
        donors = {str(i): pd.Series(np.repeat(v, 28), index=dates) for i, v in enumerate([1., 3., 9.])}
        self.assertEqual(n.donor_prior(donors, w.ORIGIN)["forecast"], [3.] * 28)
        donors["0"][:] = 2.
        self.assertEqual(n.donor_prior(donors, w.ORIGIN)["forecast"], [3.] * 28)
        donors["1"][:] = 4.
        self.assertEqual(n.donor_prior(donors, w.ORIGIN)["forecast"], [4.] * 28)
        with self.assertRaises(ValueError): n.donor_prior({"0": donors["0"]}, w.ORIGIN)

    def test_actual_dispatch_disagreement_and_abstention(self):
        view = self.view()
        decision = w.choose("C", view, mode="injected", client=FixtureClient("intermittent"))
        self.assertEqual(decision["executed_route"], "intermittent")
        self.assertEqual(decision["source"], "llm")
        self.assertEqual(w.execute(decision["executed_route"], view)["route"], "intermittent")
        decision = w.choose("C", view, mode="injected", client=FixtureClient("clarify"))
        self.assertIsNone(decision["executed_route"])
        self.assertEqual(decision["status"], "abstained")

    def test_ineligible_malformed_truncated_and_unknown_fields(self):
        decision = w.choose("C", self.view("M02-S01"), mode="injected", client=FixtureClient("regular"))
        self.assertTrue(decision["schema_valid"])
        self.assertEqual(decision["source"], "rule_fallback")
        self.assertEqual(decision["executed_route"], "short_history")
        for raw in [{"text": "{", "metadata": {}}, {"text": "{}", "metadata": {"done_reason": "length"}}]:
            client = FixtureClient(raw=raw)
            result = w.choose("C", self.view(), mode="injected", client=client)
            self.assertEqual(result["source"], "rule_fallback")
            self.assertEqual(client.calls, 1)
        value = json.loads(FixtureClient().raw["text"]); value["extra"] = "forbidden"
        with self.assertRaises(ValueError): w.validate(value, w.profile(self.view()))
        value.pop("extra"); value["evidence_ids"] = ["missing"]
        with self.assertRaises(ValueError): w.validate(value, w.profile(self.view()))

    def test_exact_replay_and_failure_separation(self):
        view = self.view()
        original = w.choose("C", view, mode="live", client=FixtureClient())
        replay = w.choose("C", view, recorded=original)
        self.assertEqual(replay["llm_attempts"], 0)
        self.assertEqual(replay["accepted_route"], original["accepted_route"])
        changed = deepcopy(original); changed["response"]["text"] += " "
        with self.assertRaisesRegex(ValueError, "integrity"): w.choose("C", view, recorded=changed)
        changed = deepcopy(view); changed["history"]["values"][0] += 1
        with self.assertRaisesRegex(ValueError, "input mismatch"): w.choose("C", changed, recorded=original)
        bad = w.choose("C", view, mode="live", client=FixtureClient(raw={"text": "{", "metadata": {}}))
        self.assertEqual(w.choose("C", view, recorded=bad)["source"], "rule_fallback")

    def test_json_shape_does_not_imply_cross_field_contract(self):
        evidence = w.profile(self.view())
        value = json.loads(FixtureClient().raw["text"])
        value["clarification_question"] = "Unnecessary question?"
        self.assertTrue(shape_valid(value, evidence))
        with self.assertRaisesRegex(ValueError, "null question"):
            w.validate(value, evidence)
        value["evidence_ids"] = [["invalid"]]
        self.assertFalse(shape_valid(value, evidence))

    def test_short_selection_refit_and_simple_forecasts(self):
        history = pd.Series(np.arange(60.), index=pd.date_range("2016-01-28", periods=60))
        mean = n.fit_simple(history, "mean28")
        self.assertEqual(mean["forecast"], [45.5] * 28)
        naive = n.fit_forecast(history, "seasonal_naive")
        self.assertEqual(naive["forecast"], list(np.tile(np.arange(53., 60.), 4)))
        out = n.run_short_path(history)
        self.assertEqual(out["config"]["train_window"], 60)
        self.assertEqual(out["validation_config"]["train_window"], 32)
        self.assertTrue(all(f["train_n"] == 60 for f in out["fit_trace"] if f["phase"] == "final_refit"))
        zero = history * 0
        for method in ("seasonal_naive", "tsb"):
            self.assertEqual(n.fit_forecast(zero, method)["forecast"], [0.] * 28)

    def test_selection_failure_tie_and_final_recovery(self):
        history = w.as_series(self.view("M02-S01")["history"])
        def fake(train, model, config):
            fit = n.fit_simple(train, model, config)
            if len(train) == 60 and model == "mean28": fit.update(status="failed", error="injected final failure")
            if len(train) == 32 and model == "seasonal_naive": fit["forecast"] = [10000.] * 28
            return fit
        result = n.run_short_path(history, forecaster=fake)
        self.assertEqual(result["selected"], "mean28")
        self.assertEqual(result["delivered"], "seasonal_naive")
        self.assertEqual(result["status"], "model_fallback")
        candidates = {m: {"eligible": True, "mean_rmse": 1.} for m in n.SHORT_CATALOG}
        self.assertEqual(n.select_candidate(candidates, n.SHORT_CATALOG), "seasonal_naive")
        candidates["seasonal_naive"]["eligible"] = False
        with self.assertRaises(ValueError): n.select_candidate(candidates, n.SHORT_CATALOG)

    def test_scaled_metric_denominator_and_typed_formatting(self):
        item = {"metrics": {"mae": 1., "rmse": 1., "mase": None}}
        row = {"case": {"cohort": "main", "scenario": "full", "case_id": "test"},
               "arms": {a: deepcopy(item) for a in "ABCD"}}
        stats = w.summary({"rows": [row]})["full_history"]
        self.assertEqual(stats["A"]["common_n"], 1)
        self.assertEqual(stats["A"]["scaled_n"], 0)
        self.assertIsNone(stats["A"]["macro_mase"])
        self.assertEqual(format_cell(1.), "1.00")
        self.assertEqual(format_cell(1), "1")
        self.assertEqual(format_cell(None), "N/A")

    def test_corrupted_source_rejected_before_views(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("manifest.json", "m5_routing_panel.csv"):
                shutil.copy(w.ROOT / "data/a02" / name, directory)
            with (Path(directory) / "m5_routing_panel.csv").open("a") as stream: stream.write("\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"): n.load_routing_panel(directory)

    def test_frozen_forecasts_ignore_outcomes_and_masked_values(self):
        capture = json.loads((w.ROOT / "examples/module02/recorded_run.json").read_text())
        selected = [self.cases[cid] for cid in ("M02-F01", "M02-S01", "M02-N01")]
        original = w.freeze_run(self.panel, self.source, selected, capture)
        changed = self.panel.copy()
        changed.loc[changed.date > pd.Timestamp(w.ORIGIN), "sales_units"] += 100
        later = w.freeze_run(changed, self.source, selected, capture)
        self.assertEqual(w.semantic(original.data), w.semantic(later.data))
        old_scores, new_scores = w.score_run(original, self.panel), w.score_run(later, changed)
        for old, new in zip(old_scores["rows"], new_scores["rows"]):
            self.assertNotEqual(old["arms"]["B"]["metrics"], new["arms"]["B"]["metrics"])
        changed = self.panel.copy()
        mask = (changed.series_id == selected[0]["series_id"]) & (changed.date < pd.Timestamp("2016-01-28"))
        changed.loc[mask, "sales_units"] += 100
        restricted = w.freeze_run(changed, self.source, selected[1:], capture)
        for row in restricted.data["rows"]:
            old = next(r for r in original.data["rows"] if r["case"]["case_id"] == row["case"]["case_id"])
            self.assertEqual(w.semantic(row), w.semantic(old))


if __name__ == "__main__": unittest.main()
