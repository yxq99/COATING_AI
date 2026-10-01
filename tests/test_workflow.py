"""仅用临时合成数据测试，不生成真实实验数据。"""
import tempfile
import unittest
from pathlib import Path

import torch

from coating_ai.config import feasible
from coating_ai.data import frame, initial_points, load, save, simulate
from coating_ai.learning import cross_validate
from coating_ai.optimization import pareto_hv


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_initial_constraints_and_reproducibility(self):
        x = initial_points(1024)
        self.assertTrue(feasible(x).all())
        self.assertTrue(torch.equal(x, initial_points(1024)))

    def test_no_overwrite_and_missing_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.csv"
            save(frame(initial_points()), path)
            with self.assertRaises(FileExistsError):
                save(frame(initial_points()), path)
            with self.assertRaisesRegex(ValueError, "空白"):
                load(path)

    def test_demo_never_enters_real_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = initial_points()
            path = Path(tmp) / "demo.csv"
            save(frame(x, simulate(x), kind="demo"), path)
            with self.assertRaisesRegex(ValueError, "不可混合"):
                load(path)
            _, read_x, _ = load(path, allow_demo=True)
            self.assertTrue(torch.allclose(x, read_x))

    def test_constraint_violation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = initial_points()
            df = frame(x, simulate(x), kind="demo")
            df.loc[0, "mxene_wt"] = 99
            path = Path(tmp) / "invalid.csv"
            save(df, path)
            with self.assertRaisesRegex(ValueError, "违反"):
                load(path, allow_demo=True)

    def test_hv_direction(self):
        y = torch.tensor([[7., 2., 30.], [8., 3., 40.]], dtype=torch.double)
        mask, hv = pareto_hv(y)
        self.assertEqual(mask.tolist(), [False, True])
        self.assertAlmostEqual(hv, (8-5.5)*(3+0.5)*(40+5))

    def test_leave_one_out(self):
        x = initial_points(6)
        metrics = cross_validate(x, simulate(x))
        self.assertEqual(len(metrics), 3)
        self.assertTrue((metrics["MAE"] >= 0).all())
        self.assertFalse(metrics.isna().any().any())


if __name__ == "__main__":
    unittest.main()
