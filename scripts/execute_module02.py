"""Execute Module 2 in a fresh kernel; prohibit network use in default replay."""
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    root = Path(__file__).resolve().parents[1]
    for variable, folder in [("MPLCONFIGDIR", "module02-mpl"), ("JUPYTER_RUNTIME_DIR", "module02-jupyter"), ("NUMBA_CACHE_DIR", "module02-numba")]:
        os.environ.setdefault(variable, str(Path(tempfile.gettempdir()) / folder))
    path = root / "a02-routing.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code": cell.outputs, cell.execution_count = [], None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=300, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells in a fresh kernel")


if __name__ == "__main__": main()
