"""Causal boundaries, bank equivalence, masks and evidence for the panel study."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentic_ts import module02_panel as p
from agentic_ts import module02_workflow as w
from agentic_ts import routing_forecasting as n
from agentic_ts.module01_assets import PublicationAssets


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = p.load_data()
        cls.case = cls.data.cases[0]

    def changed_data(self, case, selector):
        series = dict(self.data.series); sid = case["series_id"]
        series[sid] = series[sid].copy()
        mask = selector(series[sid].index)
        series[sid].loc[mask] += 100
        return p.StudyData(series,self.data.items,self.data.cases,self.data.manifest)

    def test_exact_grid_and_selection(self):
        self.assertEqual(len({m["item_id"] for m in self.data.items.values()}),100)
        self.assertEqual(len(self.data.series),300)
        self.assertEqual(sum(len(s) for s in self.data.series.values()),210300)
        protocol = p.record("protocol.json")
        self.assertEqual(set(protocol["excluded_original_items"]) & set(protocol["item_ids"]),set())
        self.assertEqual(sum(c["role"]=="development" for c in self.data.cases),600)
        self.assertEqual(sum(c["role"]=="main" for c in self.data.cases),1800)
        self.assertFalse(any(c["store"]=="WI_1" for c in self.data.cases if c["role"]=="development"))

    def test_calendar_and_inner_outer_separation(self):
        for case in self.data.cases[::100]:
            h = p.history(self.data,case)
            self.assertEqual(len(h),449)
            for train,valid in n.validation_windows(h):
                self.assertEqual(len(train),365);self.assertEqual(len(valid),28)
                self.assertLess(train.index[-1],valid.index[0]);self.assertLessEqual(valid.index[-1],h.index[-1])
        origins = sorted({pd.Timestamp(c["as_of"]) for c in self.data.cases if c["role"]=="main"})
        self.assertTrue(all((b-a).days==28 for a,b in zip(origins,origins[1:])))

    def test_future_and_masked_history_do_not_enter_view(self):
        case = self.case; origin = pd.Timestamp(case["as_of"])
        changed = self.changed_data(case,lambda dates:dates>origin)
        self.assertEqual(p.view(self.data,case),p.view(changed,case))
        later = next(c for c in self.data.cases if c["series_id"]==case["series_id"] and pd.Timestamp(c["as_of"])>origin)
        self.assertNotEqual(p.view(self.data,later),p.view(changed,later))
        last = next(c for c in self.data.cases if c["series_id"]==case["series_id"] and c["day"]==1885)
        before = pd.Timestamp(last["as_of"])-pd.Timedelta(days=448)
        changed = self.changed_data(last,lambda dates:dates<before)
        self.assertEqual(p.view(self.data,last),p.view(changed,last))

    def test_frozen_predictions_ignore_changed_outcomes(self):
        original = p.freeze_cases(self.data,[self.case],p.FitBank(),"R")
        changed = self.changed_data(self.case,lambda dates:dates>pd.Timestamp(self.case["as_of"]))
        second = p.freeze_cases(changed,[self.case],p.FitBank(),"R")
        self.assertEqual(w.semantic(original.data),w.semantic(second.data))
        a,b = p.score(original,self.data),p.score(second,changed)
        self.assertNotEqual(a["rows"][0]["arms"]["B"]["metrics"],b["rows"][0]["arms"]["B"]["metrics"])
        self.assertNotIn("actual",original.data["rows"][0])

    def test_fit_bank_cache_is_training_slice_specific(self):
        bank = p.FitBank();p.freeze_case(self.data,self.case,bank,"R")
        self.assertEqual((len(bank.fits),bank.lookups,bank.hits),(12,12,0))
        p.freeze_case(self.data,self.case,bank,"R")
        self.assertEqual((len(bank.fits),bank.hits),(12,12))
        later = next(c for c in self.data.cases if c["series_id"]==self.case["series_id"] and c["day"]==1689)
        p.freeze_case(self.data,later,bank,"R")
        self.assertEqual((len(bank.fits),bank.hits),(15,21))
        train=p.history(self.data,self.case).tail(365);other=train.copy();other.iloc[0]+=1
        self.assertNotEqual(bank.key(train,"seasonal_naive"),bank.key(other,"seasonal_naive"))

    def test_bank_paths_equal_independent_executors(self):
        row=p.freeze_case(self.data,self.case,p.FitBank(),"R");h=p.history(self.data,self.case)
        for arm,route in [("R","regular"),("I","intermittent"),("D",None)]:
            actual=n.run_forecast_path(route,h) if route else n.run_unrestricted_reference(h,[])
            for key in ("selected","delivered","status","forecast","dates"):
                self.assertEqual(row["arms"][arm][key],actual[key])
            self.assertEqual(row["arms"][arm]["logical_fit_attempts"],len(actual["fit_trace"]))
        self.assertEqual(row["arms"]["F"],row["arms"]["R"])

    def test_route_agreement_is_a_construction_check(self):
        row=p.freeze_case(self.data,self.case,p.FitBank(),"I")
        expected=w.rule_route(w.profile(p.view(self.data,self.case)))
        self.assertEqual(row["policy_route"],expected)
        self.assertEqual(row["arms"]["B"],row["arms"]["R" if expected=="regular" else "I"])

    def test_zero_scale_and_empty_eligible_selection(self):
        series=dict(self.data.series);series[self.case["series_id"]]=series[self.case["series_id"]]*0
        data=p.StudyData(series,self.data.items,self.data.cases,self.data.manifest)
        frozen=p.freeze_cases(data,[self.case],p.FitBank(),"R");out=p.score(frozen,data)
        stats=p.arm_summary(out["rows"])
        self.assertEqual(stats["B"]["scaled_n"],0);self.assertIsNone(stats["B"]["macro_mase"])
        self.assertEqual(stats["B"]["delivered_n"],1)
        with self.assertRaisesRegex(ValueError,"No common positive-scale"):p.select_fixed(out)

    def test_development_selection_rejects_main_or_wi(self):
        out=p.score(p.freeze_cases(self.data,[self.case],p.FitBank()),self.data)
        self.assertIn(p.select_fixed(out)["selected_arm"],("R","I"))
        changed=deepcopy(out);changed["rows"][0]["case"]["store"]="WI_1"
        with self.assertRaises(ValueError):p.select_fixed(changed)
        changed=deepcopy(out);changed["rows"][0]["case"]["role"]="main"
        with self.assertRaises(ValueError):p.select_fixed(changed)
        for r in out["rows"]:
            for arm in ("R","I"):r["arms"][arm]["metrics"]["mase"]=1.
        self.assertEqual(p.select_fixed(out)["selected_arm"],"R")

    def test_wi_sales_cannot_influence_fixed_comparator(self):
        cases=[c for c in self.data.cases if c["role"]=="development"][:2]
        series=dict(self.data.series)
        for sid,meta in self.data.items.items():
            if meta["store"]=="WI_1":series[sid]=series[sid]+100
        changed=p.StudyData(series,self.data.items,self.data.cases,self.data.manifest)
        a=p.score(p.freeze_cases(self.data,cases,p.FitBank()),self.data)
        b=p.score(p.freeze_cases(changed,cases,p.FitBank()),changed)
        self.assertEqual(p.select_fixed(a),p.select_fixed(b))

    def test_last_horizon_cannot_change_any_origin_forecast(self):
        cases=[c for c in self.data.cases if c["role"]=="main" and c["series_id"]==self.case["series_id"]]
        end=pd.Timestamp(cases[-1]["as_of"])
        changed=self.changed_data(cases[-1],lambda dates:dates>end)
        first=p.freeze_cases(self.data,cases,p.FitBank(),"I")
        second=p.freeze_cases(changed,cases,p.FitBank(),"I")
        self.assertEqual(w.semantic(first.data),w.semantic(second.data))
        a,b=p.score(first,self.data),p.score(second,changed)
        self.assertEqual(a["rows"][0]["actual"],b["rows"][0]["actual"])
        self.assertNotEqual(a["rows"][-1]["actual"],b["rows"][-1]["actual"])

    def test_model_fallback_and_failed_validation_are_retained(self):
        real=n.fit_forecast
        final_end=pd.Timestamp(self.case["as_of"])
        def fake(train,model,config):
            fit=real(train,model,config)
            if model=="hw_add_damped" and train.index[-1]==final_end:fit.update(status="failed",error="injected final failure")
            return fit
        with patch.object(n,"fit_forecast",side_effect=fake):
            row=p.freeze_case(self.data,self.case,p.FitBank(),"R")
        # Force candidate ranking to demonstrate the preserved final-fit recovery.
        row["candidates"]["hw_add_damped"].update(eligible=True,mean_rmse=-1.)
        bank=p.FitBank()
        with patch.object(n,"fit_forecast",side_effect=fake):
            h=p.history(self.data,self.case);finals={m:bank.forecast(h.tail(365),m) for m in n.CATALOG}
        out=p.reconstruct_path(h,"regular",row["candidates"],finals,row["fit_keys"])
        self.assertEqual((out["selected"],out["delivered"],out["status"]),("hw_add_damped","seasonal_naive","model_fallback"))

    def test_common_mask_and_block_bootstrap(self):
        frozen=p.freeze_cases(self.data,[self.case],p.FitBank(),"R");out=p.score(frozen,self.data)
        row=out["rows"][0]
        for a in p.ARMS:row["arms"][a]["metrics"]["mase"]=1. if a=="B" else 2.
        copy=deepcopy(row);copy["case"]["store"]="TX_1"
        values=p.bootstrap_items([row,copy],"F",repetitions=20)
        self.assertEqual((values["item_blocks"],values["low"],values["high"]),(1,-1.,-1.))
        copy["arms"]["D"]["metrics"]=None
        self.assertEqual(len(p.common_rows([row,copy])[0]),1)

    def test_predeclared_timing_subset(self):
        cases=p.timing_cases(self.data.cases,p.record("protocol.json"))
        self.assertEqual(len(cases),36)
        self.assertEqual(len({c["item_id"] for c in cases}),2)
        self.assertTrue(all(c["role"]=="main" for c in cases))

    def test_corrupted_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("manifest.json","sales.csv.gz","items.json","cases.json"):
                shutil.copy(p.DATA/name,Path(directory)/name)
            with (Path(directory)/"sales.csv.gz").open("ab") as stream:stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError,"checksum"):p.load_data(directory)

    def test_nested_export_and_original_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            class Config:
                graph_folder=root/"graphs/module02-panel";example_folder=root/"examples/panel"
                export_dpi=180
            Config.root=root
            with patch("agentic_ts.module01_assets.display"):
                export=PublicationAssets(Config,"fixture",prefix="atsf02",sidecar_dir="examples/panel")
                fig,ax=plt.subplots();ax.plot([0,1],[0,1]);export.save(fig,"graph","Fixture","fixture","outside image")
            self.assertEqual(export.assets[0]["path"],"graphs/module02-panel/atsf02-graph1.png")
            Config.graph_folder=root/"graphs"
            with patch("agentic_ts.module01_assets.display"):
                original=PublicationAssets(Config,"fixture")
                fig,ax=plt.subplots();original.save(fig,"graph","Fixture","fixture","outside image")
            self.assertEqual(original.assets[0]["path"],"graphs/atsf01-graph1.png")


if __name__=="__main__":unittest.main()
