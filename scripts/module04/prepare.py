from pathlib import Path
import gzip
import hashlib
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
source = ROOT.parent / "time-series-with-Konrad/data/train.csv"
target = ROOT / "data/module04"
target.mkdir(parents=True, exist_ok=True)
raw = source.read_bytes()
frame = pd.read_csv(source, parse_dates=["date"])
assert frame.shape == (913000, 4)
assert list(frame.columns) == ["date", "store", "item", "sales"]
assert not frame.isna().any().any() and not frame.duplicated(["date", "store", "item"]).any()
assert sorted(frame.store.unique()) == list(range(1, 11))
assert sorted(frame.item.unique()) == list(range(1, 51))
assert (frame.sales >= 0).all()
calendar = pd.date_range("2013-01-01", periods=1826, freq="D")
for _, group in frame.groupby(["store", "item"]):
    assert pd.DatetimeIndex(group.sort_values("date").date).equals(calendar)
compressed = gzip.compress(raw, mtime=0)
(target / "retail.csv.gz").write_bytes(compressed)
aggregate = frame.groupby(["store", "date"], as_index=False).sales.sum()
aggregate.to_csv(target / "store_daily.csv", index=False, date_format="%Y-%m-%d")
sha = lambda value: hashlib.sha256(value).hexdigest()
manifest = {"source": "https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data",
            "source_file": "time-series-with-Konrad/data/train.csv", "copied_at": "2026-11-12",
            "original_retrieval_date": "unavailable; existing local series material",
            "release": "local competition train.csv snapshot; release identifier unavailable",
            "raw_sha256": sha(raw), "compressed_sha256": sha(compressed),
            "aggregate_sha256": sha((target / "store_daily.csv").read_bytes()),
            "raw_shape": [913000, 4], "aggregation": "sum sales over all50 items by store/date",
            "stores": 10, "items_per_store": 50, "dates": 1826, "store_days": 18260,
            "units": "daily sales units; competition dataset, retailer authenticity not established",
            "calendar": ["2013-01-01", "2017-12-31"],
            "limitations": "historical competition snapshot, no real-time-vintage or representativeness claim"}
(target / "manifest.json").write_text(json.dumps(manifest, indent=2))
for name in ["request", "response"]:
    shutil.copyfile(f"/tmp/a04-pilot-{name}.json", ROOT / f"examples/module04/development/transport-{name}.json")
print(json.dumps(manifest, indent=2))
