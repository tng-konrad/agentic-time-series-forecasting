from pathlib import Path
import argparse
import os
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["live", "replay"])
parser.add_argument("--run-id")
args = parser.parse_args()
if args.mode == "live" and not args.run_id:
    parser.error("live mode requires --run-id so the author audit can target the frozen archive")
os.chdir(ROOT)
os.environ["A04_MODE"] = args.mode
os.environ["MPLCONFIGDIR"] = "/tmp/a04-matplotlib"
if args.run_id:
    os.environ["A04_RUN_ID"] = args.run_id
nb = nbformat.read(ROOT/"a04-plan-and-execute.ipynb", as_version=4)
if args.mode == "replay":
    # Execution-only cell; removed from final notebook, leaving computational code intact.
    nb.cells.insert(0, nbformat.v4.new_code_cell("""import socket
def reject_network(*args, **kwargs):
    raise RuntimeError('Network/model access blocked for analysis-only replay verification')
socket.create_connection = reject_network
socket.socket.connect = reject_network
"""))

def before_cell(cell, cell_index, **kwargs):
    if args.mode == "live" and cell.cell_type == "code" and cell.source.startswith("result = score_study"):
        audit = ROOT/"examples/module04/runs"/args.run_id/"audit.json"
        print("FORECASTS FROZEN; waiting for step-local author audit:", audit, flush=True)
        deadline = time.monotonic()+1800
        while not audit.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Unscored capture retained; author audit not supplied within30 minutes")
            time.sleep(.5)

def after_cell(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print("Executed cell", cell_index, flush=True)

client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=before_cell, on_cell_executed=after_cell)
try:
    client.execute()
finally:
    if args.mode == "replay":
        nb.cells.pop(0)
    nbformat.write(nb, ROOT/"a04-plan-and-execute.ipynb")
print("FINISHED", args.mode, flush=True)
