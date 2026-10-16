"""Prevent publication plots from concealing incomplete or duplicated runs."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from reproducibility.paper_figures import paper_rmse_figure


class PaperFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.settings = dict(methods=["KNN"], mechanisms=["MCAR", "MAR", "MNAR"],
                             fractions=[0.1, 0.3, 0.5], final_seeds=[42, 100])
        (self.out / "run.json").write_text(json.dumps({"settings": self.settings}))
        # Synthetic values are used only for failure-mode and summary tests.
        self.scores = pd.DataFrame([
            dict(method="KNN", mechanism=m, fraction=f, seed=s, rmse=r)
            for m in self.settings["mechanisms"] for f in self.settings["fractions"]
            for s, r in [(42, 0.2), (100, 0.4)]
        ])

    def test_incomplete_or_duplicate_grids_rejected(self):
        for scores in [self.scores.iloc[1:], pd.concat([self.scores, self.scores.iloc[:1]])]:
            with self.assertRaisesRegex(ValueError, "grid"):
                paper_rmse_figure(scores, self.out, show=False)

    def test_summary_uses_seed_replicates_and_sample_sd(self):
        outputs, values, summary = paper_rmse_figure(self.scores, self.out, show=False)
        self.assertEqual(len(values), 18)
        np.testing.assert_allclose(summary["mean"], 0.3)
        np.testing.assert_allclose(summary["std"], np.sqrt(0.02))
        self.assertTrue((summary["count"] == 2).all())
        self.assertTrue(Path(outputs["main_all_conditions_figure"]["pdf"]).is_file())
        with self.assertRaises(ValueError):
            paper_rmse_figure(self.scores.assign(rmse=np.nan), self.out, show=False)

    def test_single_seed_does_not_invent_variability(self):
        self.settings["final_seeds"] = [42]
        (self.out / "run.json").write_text(json.dumps({"settings": self.settings}))
        _, _, summary = paper_rmse_figure(self.scores[self.scores.seed == 42], self.out, show=False)
        self.assertTrue(summary["std"].isna().all())
        manifest = json.loads((self.out / "figures/figure2_TCGA_UCEC/plot_manifest.json").read_text())
        self.assertEqual(manifest["uncertainty"], "not estimated")


if __name__ == "__main__":
    unittest.main()
