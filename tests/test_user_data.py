"""Input, reference-truth, export and notebook checks for the own-data companion."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

from reproducibility.pipeline import Settings
from reproducibility.user_data import (
    load_uploaded_data, reference_for_benchmark, impute_selected_dataset, select_by_validation,
)


class UploadedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.path = self.out / "input.csv"
        self.path.write_text("sample_id,label,Protein A,protein/b\n00001,case,1.2300,\n00002,control,NA,4.0\n00003,,3,6\n")
        self.data = load_uploaded_data(self.path, metadata_columns=["label"])
        self.settings = Settings(methods=("KNN", "Iterative"))
        self.frozen = {"KNN": {"validation_rmse": .2, "params": {"k": 3}, "best_trial": 0, "completed_trials": 1},
                       "Iterative": {"validation_rmse": .4, "params": {}, "best_trial": 0, "completed_trials": 1}}

    def test_headers_ids_and_numeric_values_are_preserved(self):
        self.assertEqual(self.data.original.sample_id.tolist(), ["00001", "00002", "00003"])
        self.assertEqual(self.data.feature_columns, ["Protein A", "protein/b"])
        self.assertEqual(self.data.matrix.shape, (3, 2))
        self.assertEqual(int(self.data.matrix.isna().sum().sum()), 2)
        self.assertNotIn("label", self.data.matrix.columns)
        for text in ["a,a\n1,2\n", "a,b\ntext,2\n", "a,b\n,2\n", "a,b\ninf,2\n"]:
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                load_uploaded_data(self.path)

    def test_unknown_values_are_not_manufactured_into_reference_truth(self):
        before = self.data.matrix.copy()
        with self.assertRaisesRegex(ValueError, "complete reference rows"):
            reference_for_benchmark(self.data)
        pd.testing.assert_frame_equal(before, self.data.matrix)

    def test_separate_reference_aligns_names_and_rejects_missing_truth(self):
        ref_path = self.out / "reference.csv"
        pd.DataFrame({"protein/b": np.arange(55) + 2., "Protein A": np.arange(55) + 1.}).to_csv(ref_path, index=False)
        ref = load_uploaded_data(ref_path)
        x, audit = reference_for_benchmark(self.data, ref)
        self.assertEqual(audit["reference_samples"], 55)
        np.testing.assert_array_equal(x.iloc[:, 0], np.arange(55) + 1.)
        ref.matrix.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "contains missing"):
            reference_for_benchmark(self.data, ref)

    def test_selected_export_uses_original_missing_data_and_keeps_formatting(self):
        completed = self.data.matrix.fillna(self.data.matrix.mean())
        with patch("reproducibility.user_data.impute", return_value=completed) as fit:
            path, result = impute_selected_dataset(self.data, self.out, self.settings, self.frozen, "KNN")
        pd.testing.assert_frame_equal(fit.call_args.args[0], self.data.matrix)
        self.assertEqual(fit.call_args.args[1], "KNN")
        self.assertEqual(result.loc[0, "Protein A"], "1.2300")
        self.assertEqual(result.sample_id.tolist(), ["00001", "00002", "00003"])
        pd.testing.assert_series_equal(result.label, self.data.original.label)
        self.assertTrue(path.is_file())
        self.assertTrue(np.isfinite(result[self.data.feature_columns].apply(pd.to_numeric)).all().all())
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            impute_selected_dataset(self.data, self.out, self.settings, self.frozen, "Iterative")

    def test_failed_completion_cannot_be_exported(self):
        changed = self.data.matrix.fillna(0.) + 1.
        with patch("reproducibility.user_data.impute", return_value=changed):
            with self.assertRaisesRegex(ValueError, "preserve every observed"):
                impute_selected_dataset(self.data, self.out, self.settings, self.frozen, "KNN")

    def test_selection_requires_all_requested_finite_validation_results(self):
        winner, ranking = select_by_validation(self.frozen, self.settings.methods)
        self.assertEqual(winner, "KNN")
        self.assertEqual(ranking.validation_rank.tolist(), [1, 2])
        self.frozen["KNN"]["validation_rmse"] = np.nan
        with self.assertRaises(ValueError):
            select_by_validation(self.frozen, self.settings.methods)

    def test_own_data_notebook_structure_and_syntax(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/impute_your_own_data.ipynb"
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        sources = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        self.assertIn("tune(reference, OUT, settings)", sources)
        self.assertIn("evaluate(reference, OUT, settings, frozen)", sources)
        self.assertNotIn("association_analysis(", sources)


if __name__ == "__main__":
    unittest.main()
