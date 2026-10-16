"""Checks target scientific failure modes: leakage, masks, alignment, and metrics."""
import unittest
import numpy as np
import pandas as pd
from reproducibility.missingness import corrupt, validation_mask, masked_rmse
from reproducibility.associations import preservation, covariate_design, scan


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(np.random.default_rng(7).normal(size=(100, 60)),
                              index=[f"sample_{i}" for i in range(100)], columns=[f"p{i}" for i in range(60)])

    def test_fixed_masks_and_disjoint_validation(self):
        for mechanism in ["MCAR", "MAR", "MNAR"]:
            bad, mask = corrupt(self.x, mechanism, .3, 42)
            again, repeated = corrupt(self.x, mechanism, .3, 42)
            np.testing.assert_array_equal(mask, repeated)
            np.testing.assert_array_equal(mask.sum(axis=0), np.full(60, 30))
            holdout = validation_mask(bad, 999)
            self.assertFalse((holdout & mask).any())
            np.testing.assert_array_equal(bad.to_numpy()[~mask], self.x.to_numpy()[~mask])
            self.assertFalse(np.array_equal(mask, corrupt(self.x, mechanism, .3, 100)[1]))

    def test_rmse_does_not_discard_failed_predictions(self):
        bad, mask = corrupt(self.x, "MCAR", .1, 42)
        with self.assertRaises(ValueError):
            masked_rmse(self.x, bad, mask)
        self.assertEqual(masked_rmse(self.x, self.x, mask), 0)

    def test_identical_associations_and_empty_fdr(self):
        table = pd.DataFrame({"beta": np.arange(60) + .1, "z": np.arange(60) + .2,
                              "p": np.full(60, .8), "q": np.full(60, .9)}, index=self.x.columns)
        metrics = preservation(table, table)
        self.assertAlmostEqual(metrics["z_spearman"], 1)
        self.assertEqual(metrics["top50_overlap"], 1)
        self.assertEqual(metrics["beta_rmse"], 0)
        self.assertTrue(np.isnan(metrics["fdr_jaccard"]))
        self.assertTrue(np.isnan(preservation(table.iloc[:10], table.iloc[:10])["top50_overlap"]))

    def test_covariates_and_sample_order(self):
        meta = pd.DataFrame({"outcome": np.tile([0, 1], 50), "age": np.linspace(40, 80, 100)}, index=self.x.index)
        design = covariate_design(meta, ["age"])
        with self.assertRaises(ValueError):
            scan(self.x.iloc[::-1], meta, "outcome", design)
        reference = scan(self.x.iloc[:, :3], meta, "outcome", design)
        self.assertEqual(len(reference), 3)
        self.assertTrue(reference.p.notna().all())


if __name__ == "__main__":
    unittest.main()
