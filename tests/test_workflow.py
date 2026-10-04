from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from coating_ai.artifacts import load_training_run, save_training_run
from coating_ai.config import feasible, load_project_config, resolve_search_bounds
from coating_ai.data import frame, initial_points, load_data, simulate
from coating_ai.learning import build_model
from coating_ai.optimization import pareto_summary, recommend_target


def settings(*, mode="target", observed_bounds=False, pool_size=128):
    return {
        "project": {"name": "test", "initial_design_size": 8},
        "input_variables": {
            "temperature": {
                "type": "continuous", "enabled": True,
                "bounds": {
                    "enabled": not observed_bounds, "lower": 0.0, "upper": 10.0,
                    "observed_margin": 0.1,
                },
            },
            "layers": {
                "type": "integer", "enabled": True,
                "bounds": {"enabled": True, "lower": 1, "upper": 5},
            },
        },
        "target_variables": {
            "strength": {
                "enabled": True, "pareto_direction": "maximize", "reference_value": None,
                "goal": {
                    "enabled": True, "mode": "target", "value": 1.0,
                    "tolerance": 0.2, "weight": 1.0,
                },
            },
            "cost": {
                "enabled": True, "pareto_direction": "minimize", "reference_value": None,
                "goal": {
                    "enabled": True, "mode": "at_most", "value": 0.5,
                    "tolerance": 0.2, "weight": 1.0,
                },
            },
        },
        "constraints": {
            "enabled": True,
            "rules": [{
                "name": "limit", "enabled": True,
                "coefficients": {"temperature": 1.0, "layers": 1.0},
                "operator": "<=", "rhs": 12.0,
            }],
        },
        "recommendation": {
            "mode": mode, "batch_size": 3, "candidate_pool_size": pool_size,
            "seed": 7, "risk_aversion": 0.1,
            "diversity_min_distance": 0.02, "auto_reference_margin": 0.1,
        },
    }


class Posterior:
    def __init__(self, x):
        self.mean = torch.stack((x[:, 0] / 10.0, x[:, 1] / 10.0), dim=-1)
        self.variance = torch.full_like(self.mean, 0.01)


class PredictableModel:
    def posterior(self, x):
        return Posterior(x)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def config(self, **kwargs):
        path = self.temp_path / f"settings_{len(list(self.temp_path.glob('settings_*')))}.yaml"
        path.write_text(yaml.safe_dump(settings(**kwargs), sort_keys=False), encoding="utf-8")
        return load_project_config(path)

    def test_dynamic_schema_integer_inputs_and_constraints(self):
        config = self.config()
        self.assertEqual(config.input_names, ["temperature", "layers"])
        self.assertEqual(config.target_names, ["strength", "cost"])
        self.assertEqual(config.integer_indices, [1])
        points = initial_points(config)
        bounds = resolve_search_bounds(config)
        self.assertEqual(points.shape, (8, 2))
        self.assertTrue(torch.equal(points[:, 1], points[:, 1].round()))
        self.assertTrue(feasible(config, points, bounds).all())

    def test_disabled_bounds_use_observed_range_and_margin(self):
        config = self.config(observed_bounds=True)
        observed = torch.tensor([[2.0, 1.0], [8.0, 4.0]], dtype=torch.double)
        bounds = resolve_search_bounds(config, observed)
        self.assertAlmostEqual(float(bounds[0, 0]), 1.4)
        self.assertAlmostEqual(float(bounds[1, 0]), 8.6)
        self.assertEqual(bounds[:, 1].tolist(), [1.0, 5.0])
        with self.assertRaisesRegex(ValueError, "没有观测范围"):
            initial_points(config)

    def test_real_data_must_be_complete_and_not_mixed_with_demo(self):
        config = self.config()
        x = initial_points(config)
        bounds = resolve_search_bounds(config)
        y = simulate(config, x, bounds, noise=0)
        table = frame(config, x, y)
        path = self.temp_path / "real.csv"
        table.to_csv(path, index=False)
        _, loaded_x, loaded_y = load_data(config, path)
        self.assertTrue(torch.allclose(x, loaded_x))
        self.assertTrue(torch.allclose(y, loaded_y))

        table.loc[0, "strength"] = float("nan")
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "空白"):
            load_data(config, path)

        table = frame(config, x, y)
        table.loc[0, "data_kind"] = "demo"
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "不可混合"):
            load_data(config, path)

    def test_pareto_supports_mixed_directions(self):
        config = self.config(mode="pareto")
        y = torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.5, 0.5]], dtype=torch.double)
        mask, hypervolume, reference = pareto_summary(config, y)
        self.assertEqual(mask.tolist(), [False, True, True])
        self.assertGreater(hypervolume, 0)
        self.assertEqual(reference.shape, (2,))

    def test_target_mode_returns_feasible_named_recommendations(self):
        config = self.config()
        train_x = initial_points(config)
        bounds = resolve_search_bounds(config)
        train_y = simulate(config, train_x, bounds, noise=0)
        candidates, report, metadata = recommend_target(
            config, PredictableModel(), train_x, train_y, bounds
        )
        self.assertEqual(candidates.shape, (3, 2))
        self.assertTrue(feasible(config, candidates, bounds).all())
        self.assertTrue(torch.equal(candidates[:, 1], candidates[:, 1].round()))
        expected = {"temperature", "layers", "strength_predicted", "cost_predicted",
                    "target_match_score"}
        self.assertTrue(expected.issubset(report.columns))
        self.assertEqual(metadata["score_definition"], "lower_is_better")

    def test_saved_model_rejects_a_different_variable_schema(self):
        config = self.config()
        x = initial_points(config)
        bounds = resolve_search_bounds(config)
        y = simulate(config, x, bounds, noise=0)
        table = frame(config, x, y)
        model = build_model(x, y, bounds)
        run = self.temp_path / "run"
        run.mkdir()
        save_training_run(config, run, table, model, bounds)
        loaded, loaded_table, loaded_x, loaded_y, loaded_bounds = load_training_run(config, run)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded_table), len(table))
        self.assertTrue(torch.allclose(loaded_x, x))
        self.assertTrue(torch.allclose(loaded_y, y))
        self.assertTrue(torch.equal(loaded_bounds, bounds))

        raw = settings()
        raw["input_variables"]["temperature_renamed"] = raw["input_variables"].pop("temperature")
        raw["constraints"]["rules"][0]["coefficients"] = {
            "temperature_renamed": 1.0, "layers": 1.0,
        }
        other_path = self.temp_path / "other.yaml"
        other_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        other = load_project_config(other_path)
        with self.assertRaisesRegex(ValueError, "变量名称、类型或顺序"):
            load_training_run(other, run)


if __name__ == "__main__":
    unittest.main()
