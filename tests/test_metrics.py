import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import calculate_security_metrics, calculate_perplexity


class TestMetrics(unittest.TestCase):
    def test_zero_reward_is_preserved(self):
        metrics = calculate_security_metrics(
            [{"reward": 0}],
            [{"generated_output": "reward: 5", "response_time_seconds": 0.1}],
        )
        self.assertEqual(metrics["average_reward"], 0.0)

    def test_empty_perplexity_is_nan(self):
        class DummyModel:
            def eval(self):
                return self

        self.assertTrue(math.isnan(calculate_perplexity(DummyModel(), [])))


if __name__ == "__main__":
    unittest.main()
