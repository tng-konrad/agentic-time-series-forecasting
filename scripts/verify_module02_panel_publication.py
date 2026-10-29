"""Offline arithmetic, chronology, provenance and publication checks for the panel study."""
import base64
from collections import Counter
import hashlib
import json
import re

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from agentic_ts import module02_panel as p
from agentic_ts import module02_workflow as w
from agentic_ts import routing_forecasting as n
from scripts.verify_module02_publication import require, prose, local_links


def evidence_value(reference):
    path,pointer=reference.split("#",1);value=p.read(p.ROOT/path)
    for key in pointer.strip("/").split("/") if pointer.strip("/") else []:
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def main():
    nbpath=p.ROOT/"a02-routing-panel.ipynb";postpath=p.ROOT/"a02-post-new.md"
    notebook=nbformat.read(nbpath,as_version=4);nbformat.validate(notebook);post=postpath.read_text()
    codes=[c for c in notebook.cells if c.cell_type=="code"]
    require(len(codes)==31 and [c.execution_count for c in codes]==list(range(1,32)),"Incomplete clean-kernel execution")
    headings=[];sections={};section=None
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=="markdown":
            for title in re.findall(r"^## (.+)$",prose(cell.source),re.M):
                headings.append(title);section=title
            local_links(cell.source,p.ROOT)
        else:
            sections[cell.id]=section
            require(i+1<len(notebook.cells) and notebook.cells[i+1].cell_type=="markdown","Code lacks immediate explanation")
            require(all(o.output_type!="error" for o in cell.outputs),"Notebook has an execution error")
            require(all("text/html" not in o.get("data",{}) for o in cell.outputs),"Uninventoried rendered table")
    require(headings==p.HEADINGS==re.findall(r"^## (.+)$",prose(post),re.M),"Substantive headings differ")
    subheadings=[title for cell in notebook.cells if cell.cell_type=="markdown"
                 for title in re.findall(r"^### (.+)$",prose(cell.source),re.M)]
    require(subheadings==re.findall(r"^### (.+)$",prose(post),re.M),"Substantive subheadings differ")
    local_links(post,p.ROOT)

    data=p.load_data();protocol=p.record("protocol.json");author=p.record("author_run.json")
    require(w.digest(protocol)==author["protocol_hash"]==data.manifest["protocol_hash"],"Protocol binding changed")
    require(w.digest(data.cases)==protocol["case_manifest_hash"],"Case manifest changed")
    require(set(protocol["excluded_original_items"]).isdisjoint(protocol["item_ids"]),"Original products leaked into new sample")
    for role,origins,stores,count in [("development",protocol["development_origins"],p.STORES[:2],600),("main",protocol["main_origins"],p.STORES,1800)]:
        cases=[c for c in data.cases if c["role"]==role]
        require(len(cases)==count and {(c["item_id"],c["store"],c["as_of"]) for c in cases}==
                {(i,s,o) for i in protocol["item_ids"] for s in stores for o in origins},"Incomplete product/store/origin grid")
    frozen=p.record("frozen_forecasts.json.gz");reader=p.record("reader_frozen_forecasts.json.gz")
    result=p.record("results.json.gz");fresh=p.record("reader_results.json.gz");bank=p.record("fit_bank.json.gz")
    require(w.semantic(frozen)==w.semantic(reader) and w.semantic(result)==w.semantic(fresh),"Fresh numerical results differ")
    require(w.digest(w.semantic(frozen))==author["frozen_semantic_hash"]==result["frozen_semantic_hash"],"Prediction snapshot drift")
    require(w.digest(w.semantic(result))==author["result_semantic_hash"],"Scored-result drift")
    require(all("actual" not in r and "actual_dates" not in r for r in frozen["rows"]),"Outcomes entered frozen predictions")
    require(w.digest(w.semantic(bank))==author["bank_semantic_hash"],"Bank provenance drift")
    require(w.semantic(bank)==w.semantic(p.record("reader_fit_bank.json.gz")),"Reader bank differs")
    development=p.record("development_results.json.gz");fixed=p.record("fixed_workflow.json")
    require(p.select_fixed(development)==fixed==p.record("reader_fixed_workflow.json"),"Fixed control was not reproduced")
    require(max(r["actual_dates"][-1] for r in development["rows"])<=min(protocol["main_origins"]),"Development uses main future outcomes")

    checked_metrics=0;checked_folds=0;fits=bank["fits"];key_builder=p.FitBank()
    for outcome in (development,result):
        for row in outcome["rows"]:
            case=row["case"];h=p.history(data,case)
            require(row["view_hash"]==w.digest(p.view(data,case)),"Restricted-view mismatch")
            require(row["policy_route"]==w.rule_route(w.profile(p.view(data,case))),"Router used a different policy")
            dates=pd.date_range(pd.Timestamp(case["as_of"])+pd.Timedelta(days=1),periods=28)
            actual=data.series[case["series_id"]].reindex(dates).to_numpy(float)
            require(row["actual_dates"]==dates.strftime("%Y-%m-%d").tolist() and row["actual"]==actual.tolist(),"Incorrect final actuals")
            for model in n.CATALOG:
                candidate=row["candidates"][model]
                for i,(train,valid) in enumerate(n.validation_windows(h)):
                    key=row["fit_keys"][model][i];fit=fits[key]["fit"]
                    require(key==key_builder.key(train,model)==fits[key]["key"],"Numerical fit key does not match permitted data")
                    require(fit["train_end"]<valid.index[0].date().isoformat() and valid.index[-1]<=h.index[-1],"Validation leaks future data")
                    fold=candidate["folds"][i]
                    require(fold["status"]==fit["status"],"Fold status changed")
                    if fit["status"]=="ok":
                        errors=valid.to_numpy(float)-np.array(fit["forecast"])
                        require(np.isclose(fold["metrics"]["mae"],np.abs(errors).mean()) and
                                np.isclose(fold["metrics"]["rmse"],np.sqrt(np.mean(errors**2))),"Internal validation error arithmetic failed")
                    checked_folds+=1
                require(candidate["eligible"]==all(f["status"]=="ok" for f in candidate["folds"]),"Failed candidate was eligible")
                if candidate["eligible"]:
                    require(candidate["mean_rmse"]==float(np.mean([f["metrics"]["rmse"] for f in candidate["folds"]])),"Candidate ranking score changed")
                require(row["fit_keys"][model][-1]==key_builder.key(h.tail(365),model),"Final fit key mismatch")
            for arm,item in row["arms"].items():
                if item["forecast"] is None:
                    require(item["metrics"] is None,"Missing forecast received zero error");continue
                forecast=np.array(item["forecast"]);errors=actual-forecast
                require(len(forecast)==28 and np.isfinite(forecast).all() and (forecast>=0).all(),"Invalid forecast")
                require(item["dates"]==row["actual_dates"],"Prediction dates differ")
                mae=np.abs(errors).mean();rmse=np.sqrt(np.mean(errors**2));train=h.tail(365).to_numpy(float)
                scale=np.abs(train[7:]-train[:-7]).mean();m=item["metrics"]
                require(np.isclose(m["mae"],mae) and np.isclose(m["rmse"],rmse) and np.isclose(m["scale"],scale),"Final error/scale arithmetic failed")
                require(np.isclose(m["mase"],mae/scale) if scale>0 else m["mase"] is None,"Incorrect MASE or zero-scale treatment")
                require(item["logical_fit_attempts"]==len(item["logical_fit_keys"]),"Fit budget mismatch")
                if arm!="A":
                    models=n.CATALOG if arm=="D" else n.FAMILIES[item["route"]]
                    selected=n.select_candidate({m:row["candidates"][m] for m in models})
                    require(item["selected"]==selected,"Selected method ignores past-validation scores")
                checked_metrics+=1
            if case["role"]=="main":
                require(row["arms"]["F"]==row["arms"][fixed["selected_arm"]],"F changed during main evaluation")
                family="R" if row["policy_route"]=="regular" else "I"
                require(row["arms"]["B"]==row["arms"][family],"B did not execute its selected family")

    summary=p.summarize(result)
    require(summary==p.record("summary.json")==p.record("reader_summary.json"),"Summary or bootstrap does not reproduce")
    require(all(s["eligible_n"]==s["delivered_n"]==s["common_n"]==s["scaled_n"]==1800 for s in summary["arms"].values()),"Main denominators changed")
    timings=p.record("timings.json");cases=p.timing_cases(data.cases,protocol)
    require([r["case_id"] for r in timings["rows"]]==[c["case_id"] for c in cases] and len(cases)==36,"Timing cohort changed")
    by_id={r["case"]["case_id"]:r for r in result["rows"]}
    for source in (timings,p.record("reader_timings.json")):
        for i,row in enumerate(source["rows"]):
            require(row["order"]==(["B","D"] if i%2==0 else ["D","B"]),"Timing order did not alternate")
            for a in ("B","D"):
                require(row["arms"][a]["forecast_hash"]==w.digest(by_id[row["case_id"]]["arms"][a]["forecast"]),"Uncached benchmark changed forecasts")
                require(row["arms"][a]["fit_attempts"]==by_id[row["case_id"]]["arms"][a]["logical_fit_attempts"],"Uncached work differs from declared execution")
    work=p.work_record(bank,result,timings)
    require(work==p.record("work.json"),"Work accounting differs")
    require(work["bank"]["lookups"]==28800 and work["bank"]["actual_fit_attempts"]==9900 and work["bank"]["failed_fits"]==6,"Shared-bank accounting changed")

    assets=p.record("assets.json");require(len(assets)==9,"Require nine exports")
    pngs={c.id:[base64.b64decode(o["data"]["image/png"]) for o in c.outputs if "image/png" in o.get("data",{})] for c in codes}
    require(sum(map(len,pngs.values()))==9,"Uninventoried notebook figure")
    post_images=[];section=None
    for line in prose(post).splitlines():
        if line.startswith("## "):section=line[3:]
        post_images.extend((x,section) for x in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",line))
    require(Counter(x for x,_ in post_images)==Counter(a["path"] for a in assets),"Post assets differ")
    order={c.id:i for i,c in enumerate(codes)}
    require([order[a["cell_id"]] for a in assets]==sorted(order[a["cell_id"]] for a in assets),"Asset appearance order changed")
    for kind,count in [("graph",5),("table",4)]:
        require([a["number"] for a in assets if a["kind"]==kind]==list(range(1,count+1)),"Nonsequential filenames")
    table_cells=0
    for a in assets:
        name=f"atsf02-{a['kind']}{a['number']}.png";path=p.ROOT/a["path"]
        require(a["filename"]==name and a["path"]=="graphs/module02-panel/"+name,"Invalid export path")
        require(a["post"]==postpath.name,"Export is bound to a different post")
        require(w.file_hash(path)==a["sha256"],"Stale exported image")
        require(a["section"]==sections[a["cell_id"]] and (a["path"],a["section"]) in post_images,"Section/asset correspondence differs")
        require(len(pngs[a["cell_id"]])==1 and hashlib.sha256(pngs[a["cell_id"]][0]).hexdigest()==a["sha256"],"Notebook image bytes differ")
        expected=w.digest(work) if a["cell_id"]=="work-table" else w.digest(w.semantic(result)) if a["cell_id"] in {"errors-by-origin","errors-by-band","accuracy-table"} else w.digest(protocol)
        require(a["source_result_hash"]==expected,"Asset source is stale")
        with Image.open(path) as im:
            require(im.format=="PNG" and im.width>1000 and im.height>200,"Invalid dimensions")
            require("dpi" in im.info and all(abs(d-180)<.1 for d in im.info["dpi"]),"Invalid resolution");im.verify()
        require(a["figsize"][0]==13.0 and a["dpi"]==180 and a["font"],"Style metadata changed")
        if a["kind"]=="table":
            require(a["free_text_artists"]==0,"Table contains title/footer/text")
            require(w.file_hash(p.ROOT/a["table_sidecar"])==a["table_sidecar_sha256"],"Table source changed")
            for block in p.read(p.ROOT/a["table_sidecar"]):
                for raw,display,types in zip(block["raw"],block["display"],block["types"],strict=True):
                    require(len(raw)==len(display)==len(types)==len(block["headers"]),"Malformed table")
                    for value,rendered,typename in zip(raw,display,types,strict=True):
                        expected=f"{value:.2f}" if isinstance(value,float) else "N/A" if value is None else str(value)
                        require(type(value).__name__==typename and rendered==expected,"Table precision/type mismatch")
                        if isinstance(value,float):require(re.fullmatch(r"-?\d+\.\d{2}",rendered),"Float lacks two decimals")
                        table_cells+=1
    claims=p.record("post_claims.json")
    require(claims["post"]==postpath.name and claims["post_sha256"]==w.file_hash(postpath) and claims["result_semantic_hash"]==author["result_semantic_hash"],"Editorial review stale")
    paragraphs=post.strip().split("\n\n")
    for claim in claims["paragraphs"]:
        require(paragraphs[claim["paragraph_index"]]==claim["text"] and claim["evidence"],"Reviewed paragraph changed")
        for reference in claim["evidence"]:evidence_value(reference)
    expected=["**1.09184 for A, 0.96875 for B, 0.96996 for R, 0.97396 for I/F, and 0.96719 for D**",
              f"**{100*(1-summary['arms']['B']['macro_mase']/summary['arms']['F']['macro_mase']):.2f}%**",
              f"**{work['logical_main']['B']:,} fit attempts**",f"**{work['logical_main']['D']:,} for D**",
              f"**{100*(1-work['logical_main']['B']/work['logical_main']['D']):.2f}% fewer**",
              f"**{work['uncached_timing']['B']['seconds']:.2f} seconds**",f"**{work['uncached_timing']['D']['seconds']:.2f} seconds**",
              "Always-regular R is close to B", "does not demonstrate equivalence", "by construction", "February"]
    for fragment in expected:require(fragment in post,f"Numerical claim or qualification changed: {fragment}")
    visual=p.record("visual_qa.json")
    require(visual["status"]=="passed" and visual["widths"]==["native",760] and
            visual["asset_hashes"]=={a["filename"]:a["sha256"] for a in assets},"Visual review missing or stale")
    execution=p.record("recompute_verification.json")
    require(execution["new_llm_calls"]==0 and execution["main_cases"]==1800 and execution["development_cases"]==600,"Incomplete recomputation")
    verification={"status":"passed","executed_code_cells":len(codes),"matching_sections":len(headings),
                  "matching_subsections":len(subheadings),
                  "main_cases":1800,"development_cases":600,"independently_checked_forecast_metrics":checked_metrics,
                  "checked_validation_folds":checked_folds,"graphs":5,"tables":4,"checked_table_cells":table_cells,
                  "reviewed_empirical_paragraphs":len(claims["paragraphs"]),"new_llm_calls":0,
                  "notebook_sha256":w.file_hash(nbpath),"post":postpath.name,"post_sha256":w.file_hash(postpath),
                  "scope":"Arithmetic/provenance/chronology checks plus explicitly recorded semantic and visual author review; no guarantee of population generalization."}
    p.write(p.EXAMPLES/"publication_verification.json",verification)
    print(json.dumps(verification,indent=2))


if __name__=="__main__":main()
