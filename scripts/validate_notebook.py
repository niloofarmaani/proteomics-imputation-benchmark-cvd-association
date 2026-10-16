from pathlib import Path
import ast
import nbformat

root = Path(__file__).resolve().parents[1]
for path in (root / "notebooks").glob("benchmark_TCGA*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    print(f"Valid notebook and Python cell syntax: {path.name}")
