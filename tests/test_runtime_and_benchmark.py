import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.run_stage_benchmark import _aggregate_metrics
from run_experiment import _train_command
from runtime_config import RuntimeConfigError, load_runtime_config
from training.policy import align_next_token_logits


class RuntimeConfigTests(unittest.TestCase):
    def test_repository_runtime_config_is_valid(self):
        config = load_runtime_config()
        self.assertIn("gpt2", config["models"])
        self.assertEqual(config["version"], 1)

    def test_unknown_campaign_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.yaml"
            path.write_text(
                """
project: {root: ., bootstrap_python: python3, python: .runtime/python, mlflow_python: .runtime/mlflow, airflow_python: .runtime/airflow, runtime_dir: .runtime}
services: {mlflow: {}, airflow: {}, tensorboard: {}}
campaign:
  name: test
  models: [missing]
  pretrain_methods: [lora]
  rl_methods: [grpo]
  training: {}
  benchmark: {}
models: {gpt2: {name: gpt2, torch_dtype: float32}}
""",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeConfigError):
                load_runtime_config(path)

    def test_unknown_training_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path("configs/runtime.yaml").read_text(encoding="utf-8")
            path = Path(temp_dir) / "runtime.yaml"
            path.write_text(
                source.replace("    - lora\n", "    - unsupported_method\n", 1),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeConfigError):
                load_runtime_config(path)


class StageBenchmarkAggregationTests(unittest.TestCase):
    def test_metrics_use_their_own_denominators(self):
        metrics, counts = _aggregate_metrics(
            [
                {"accuracy": 1.0},
                {"accuracy": 0.0},
                {"safety": 1.0},
                {"word_overlap": 0.25, "actionability": 1.0},
            ]
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["safety"], 1.0)
        self.assertEqual(metrics["word_overlap"], 0.25)
        self.assertEqual(counts, {
            "accuracy": 2,
            "actionability": 1,
            "safety": 1,
            "word_overlap": 1,
        })


class ExperimentCommandTests(unittest.TestCase):
    def test_transition_adapter_override_is_hydra_add_or_replace(self):
        class Args:
            tracking_uri = "http://127.0.0.1:5000"
            mlflow_experiment = "test"
            max_steps = 1
            train_limit = 1
            eval_limit = 1
            train_batch_size = 1
            eval_batch_size = 1
            gradient_accumulation_steps = 1
            full_optimizer = "paged_adamw_8bit"
            no_eval = False

        command = _train_command(
            "gpt2", "lora", "test-run", Args(), init_adapter="adapter"
        )
        self.assertIn("++peft.peft.init_adapter_path=adapter", command)
        self.assertIn("training.training.save.save_strategy=no", command)

    def test_full_training_uses_memory_efficient_optimizer(self):
        class Args:
            tracking_uri = "http://127.0.0.1:5000"
            mlflow_experiment = "test"
            max_steps = 1
            train_limit = 1
            eval_limit = 1
            train_batch_size = 1
            eval_batch_size = 1
            gradient_accumulation_steps = 1
            full_optimizer = "paged_adamw_8bit"
            no_eval = False

        command = _train_command("gpt2", "full", "test-run", Args())
        self.assertIn(
            "training.training.optimization.optim=paged_adamw_8bit", command
        )


class PolicyAlignmentTests(unittest.TestCase):
    def test_virtual_prompt_positions_are_removed_before_kl(self):
        policy = torch.zeros(2, 12, 7)
        reference = torch.ones(2, 8, 7)
        aligned_policy, aligned_reference = align_next_token_logits(
            policy, reference
        )
        self.assertEqual(aligned_policy.shape, (2, 7, 7))
        self.assertEqual(aligned_reference.shape, (2, 7, 7))


if __name__ == "__main__":
    unittest.main()
