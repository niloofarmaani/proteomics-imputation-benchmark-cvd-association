"""Generate Figure 2 style plots from a completed run, without any refitting."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.paper_figures import paper_rmse_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True, help="Directory containing run.json and reconstruction.csv")
args = parser.parse_args()
results = pd.read_csv(args.run / "reconstruction.csv")
outputs, values, summary = paper_rmse_figure(results, args.run, show=False)
print("Seeds:", sorted(values.seed.unique().tolist()))
print("PDF:", outputs["main_all_conditions_figure"]["pdf"])
print("PNG:", outputs["main_all_conditions_figure"]["png"])
print("Summary:", outputs["summary_csv"])
