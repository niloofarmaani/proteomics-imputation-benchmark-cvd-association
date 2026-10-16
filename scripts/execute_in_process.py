"""Execute notebook cells in order when local socket restrictions block kernels.

This executes real code and captures rich outputs, but it does not validate a
Jupyter kernel or a Google Colab installation. Run nbconvert in Colab/local
Jupyter before a tagged release as well.
"""
import argparse
import os
from pathlib import Path
import sys
import traceback
import nbformat
import matplotlib
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="notebooks/benchmark_TCGA_reproducible.ipynb")
parser.add_argument("--output", default="notebooks/benchmark_TCGA_executed.ipynb")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
matplotlib.use("module://matplotlib_inline.backend_inline")
shell = InteractiveShell.instance()
notebook = nbformat.read(args.input, as_version=4)
count = 0
for i, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        continue
    count += 1
    print(f"Executing cell {i} ...", flush=True)
    with capture_output(display=True) as captured:
        result = shell.run_cell(cell.source, store_history=True)
        from matplotlib_inline.backend_inline import flush_figures
        flush_figures()
    cell.execution_count = count
    cell.outputs = []
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    error = result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exception(type(error), error, error.__traceback__)))
    nbformat.write(notebook, args.output)
    print(f"Finished cell {i}: {len(cell.outputs)} output(s)", flush=True)
    if error:
        raise RuntimeError(f"Notebook cell {i} failed: {error}") from error
notebook.metadata["validation"] = {"execution": "sequential in-process IPython", "colab_validated": False}
nbformat.validate(notebook)
nbformat.write(notebook, args.output)
print("NOTEBOOK_SUCCESS", flush=True)
