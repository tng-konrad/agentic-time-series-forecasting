"""Prepare the predeclared balanced cohort using metadata-only selection."""
from dataclasses import asdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

from agentic_ts import module02_panel as p
from agentic_ts import module02_workflow as w


def main():
    if (p.DATA / "manifest.json").exists() or (p.EXAMPLES / "protocol.json").exists():
        raise FileExistsError("Preserving the original panel protocol and extraction")
    source = p.ROOT.parent / "time-series-with-Konrad/data"
    pinned = json.loads((p.ROOT / "data/a02/manifest.json").read_text())["source_sha256"]
    for name, expected in pinned.items():
        if w.file_hash(source / name) != expected:
            raise ValueError(f"Unexpected raw source: {name}")
    with (source / "calendar.csv").open() as stream:
        calendar = {int(r["d"][2:]): r["date"] for r in csv.DictReader(stream)}
    excluded = {s["series_id"].split("_CA_1")[0] for s in p.read(p.ROOT / "data/a02/manifest.json")["series"]}
    catalog = {}; memberships = {}
    with zipfile.ZipFile(source / "sales_train_validation.csv.zip") as archive:
        with archive.open("sales_train_validation.csv") as stream:
            reader = csv.reader(io.TextIOWrapper(stream)); header = next(reader)
            for r in reader:
                if r[4] in p.STORES and r[1] not in excluded:
                    catalog[r[1]] = r[2]; memberships.setdefault(r[1], set()).add(r[4])
    rank = lambda item: (hashlib.sha256(f"{p.VERSION}|42|{item}".encode()).hexdigest(), item)
    selected = []; quotas = {}
    for department in sorted(set(catalog.values())):
        quota = 15 if department in ("FOODS_1", "FOODS_2") else 14
        candidates = sorted([i for i,d in catalog.items() if d == department and memberships[i] == set(p.STORES)], key=rank)
        if len(candidates) < quota: raise ValueError("Insufficient declared department membership")
        selected += candidates[:quota]; quotas[department] = quota
    selected = sorted(selected); selected_set = set(selected)
    items = []
    # Metadata preparation does not use sales values in selecting cohort or examples.
    with zipfile.ZipFile(source / "sales_train_validation.csv.zip") as archive:
        with archive.open("sales_train_validation.csv") as stream:
            reader = csv.reader(io.TextIOWrapper(stream)); next(reader)
            for r in reader:
                if r[1] in selected_set and r[4] in p.STORES:
                    items.append({"series_id": r[0], "item_id": r[1], "department": r[2], "store": r[4], "state": r[5]})
    items.sort(key=lambda m: m["series_id"])
    cases = []
    for role, days, stores in [("development", p.DEV_DAYS, p.STORES[:2]), ("main", p.MAIN_DAYS, p.STORES)]:
        for day in days:
            for item in items:
                if item["store"] not in stores: continue
                cases.append({**item, "role": role, "day": day, "as_of": calendar[day],
                              "case_id": f"P02-{role[0].upper()}-{day}-{item['item_id']}-{item['store']}"})
    protocol = {"version": p.VERSION, "created_at": w.timestamp(), "source_hashes": pinned,
                "selection": "Department quotas, SHA256(module02-panel-v1|42|item_id), then item_id; no outcome-based selection",
                "quotas": quotas, "item_ids": selected, "excluded_original_items": sorted(excluded), "stores": list(p.STORES),
                "development_origins": [calendar[d] for d in p.DEV_DAYS], "main_origins": [calendar[d] for d in p.MAIN_DAYS],
                "numerical_config": asdict(p.CFG), "routing_policy": w.POLICY,
                "fixed_selection": "R/I development macro MASE on common positive-scale CA_1/TX_1 cases; tie R before I",
                "primary_contrasts": ["B-F", "B-D"], "zero_band_edges": [0,.25,.5,.75,1],
                "bootstrap": {"unit":"item_id including all stores/origins", "repetitions":2000, "seed":42},
                "timing_item_ids": sorted(selected, key=rank)[:2],
                "raw_plot_item_ids": [min([i for i in selected if catalog[i] == dept], key=rank) for dept in ("FOODS_1", "HOBBIES_1")],
                "case_manifest_hash": w.digest(cases), "new_llm_calls": 0,
                "limitations": "Retrospective, department-balanced teaching cohort; observed sales; not WRMSSE, true launch or stock-availability evaluation."}
    # Freeze design before any numerical evaluation and before extracting target values.
    p.write(p.EXAMPLES / "protocol.json", protocol, exclusive=True)
    p.DATA.mkdir(parents=True, exist_ok=True)
    p.write(p.DATA / "items.json", items, exclusive=True); p.write(p.DATA / "cases.json", cases, exclusive=True)
    out = io.StringIO(); writer = csv.writer(out, lineterminator="\n"); writer.writerow(["series_id", "date", "sales_units"])
    with zipfile.ZipFile(source / "sales_train_validation.csv.zip") as archive:
        with archive.open("sales_train_validation.csv") as stream:
            reader = csv.reader(io.TextIOWrapper(stream)); next(reader)
            for r in reader:
                if r[1] in selected_set and r[4] in p.STORES:
                    for day in range(1213, 1914): writer.writerow([r[0], calendar[day], int(r[day+5])])
    (p.DATA / "sales.csv.gz").write_bytes(gzip.compress(out.getvalue().encode(), mtime=0))
    p.write(p.DATA / "manifest.json", {"version":p.VERSION,"source_hashes":pinned,"protocol_hash":w.digest(protocol),
             "files":{name:w.file_hash(p.DATA/name) for name in ("sales.csv.gz","items.json","cases.json")},
             "items":100,"series":300,"days":701,"rows":210300,"development_cases":600,"main_cases":1800,
             "span":[calendar[1213],calendar[1913]],"source":"https://github.com/Mcompetitions/M5-methods"}, exclusive=True)
    p.load_data()
    print("Prepared 100 products × 3 stores × 701 days; 600 development and 1800 main cases")


if __name__ == "__main__": main()
