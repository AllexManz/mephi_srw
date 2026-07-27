import os
import sys
import tempfile
import unittest
import shutil
from pathlib import Path

# Add src to Python path so we can import modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.utils import format_training_text


@unittest.skipUnless(
    os.environ.get("RUN_MODEL_TESTS") == "1",
    "Model snapshot tests are opt-in; set RUN_MODEL_TESTS=1 to run them.",
)

class TestModelBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a single shared temporary directory for the tests
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_dir = Path(cls.temp_dir.name)
        
        # Create dummy dataset directory
        cls.dataset_dir = cls.base_dir / "data" / "processed"
        cls.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy dataset files
        import json
        cls.dummy_data = [
            {
                "instruction": "Проанализируй уязвимость и предоставь рекомендации по исправлению",
                "input": "CVE ID: CVE-2025-1234\nДата публикации: 2025-01-01\nОписание: Test vulnerability description.\nCVSS Score: 7.5\nВектор атаки: NETWORK",
                "output": "Уязвимость CVE-2025-1234 имеет следующие характеристики:\nРекомендации по исправлению:\n1. Обновить ПО."
            },
            {
                "instruction": "Проанализируй уязвимость и предоставь рекомендации по исправлению",
                "input": "CVE ID: CVE-2025-5678\nДата публикации: 2025-02-02\nОписание: Another test vulnerability.\nCVSS Score: 5.0\nВектор атаки: LOCAL",
                "output": "Уязвимость CVE-2025-5678 имеет следующие характеристики:\nРекомендации по исправлению:\n1. Ограничить локальный доступ."
            }
        ]
        
        with open(cls.dataset_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(cls.dummy_data, f, indent=2, ensure_ascii=False)
            
        with open(cls.dataset_dir / "eval.json", "w", encoding="utf-8") as f:
            json.dump(cls.dummy_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        cls.temp_dir.cleanup()

    def _run_model_test(self, model_id: str):
        """Helper to run train and evaluate for a specific model."""
        import subprocess
        
        # 1. Run training
        train_args = [
            sys.executable,
            "src/train.py",
            f"model={model_id}",
            "peft=lora",
            f"paths.base_dir={self.base_dir}",
            f"paths.dataset_dir={self.dataset_dir}",
            "+training.training.max_steps=2",
            "training.training.save.save_steps=2",
            "training.training.evaluation.eval_steps=2",
            "model.model.training.per_device_train_batch_size=1",
            "model.model.training.per_device_eval_batch_size=1",
            "model.model.training.gradient_accumulation_steps=1",
            "logging.tensorboard.enabled=false",
            "logging.wandb.enabled=false",
            "logging.mlflow.enabled=false"
        ]
        
        print(f"\nRunning SFT LoRA training for model: {model_id}...")
        # We set a timeout of 180 seconds (3 minutes) per model to prevent hanging
        train_result = subprocess.run(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=180
        )
        
        # Check if gated repo error occurred
        stderr_lower = train_result.stderr.lower()
        stdout_lower = train_result.stdout.lower()
        is_gated = "gated" in stderr_lower or "gated" in stdout_lower or "403 client error" in stderr_lower or "forbidden" in stderr_lower
        
        if train_result.returncode != 0:
            if is_gated:
                self.skipTest(f"Model '{model_id}' is gated and access was denied (403 Forbidden).")
            else:
                print(f"Training failed for {model_id}!")
                print(f"STDOUT:\n{train_result.stdout}")
                print(f"STDERR:\n{train_result.stderr}")
                self.fail(f"Training failed with exit code {train_result.returncode}. Error: {train_result.stderr[:500]}")
                
        print(f"Training succeeded for {model_id}!")
        
        # 2. Run evaluation
        eval_args = [
            sys.executable,
            "src/evaluate.py",
            f"model={model_id}",
            "peft=lora",
            f"paths.base_dir={self.base_dir}",
            f"paths.dataset_dir={self.dataset_dir}",
            "model.model.training.per_device_eval_batch_size=1",
            "evaluation.evaluation.generation.max_examples=1"
        ]
        
        print(f"Running evaluation for model: {model_id}...")
        eval_result = subprocess.run(
            eval_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=180
        )
        
        if eval_result.returncode != 0:
            print(f"Evaluation failed for {model_id}!")
            print(f"STDOUT:\n{eval_result.stdout}")
            print(f"STDERR:\n{eval_result.stderr}")
            self.fail(f"Evaluation failed with exit code {eval_result.returncode}. Error: {eval_result.stderr[:500]}")
            
        print(f"Evaluation succeeded for {model_id}!")
        
        # Verify that evaluation output files were created
        eval_dir = self.base_dir / "logs" / model_id
        self.assertTrue(eval_dir.exists(), f"Evaluation logs directory {eval_dir} was not created.")
        
        # Find evaluation_metrics.json recursively in logs
        metrics_files = list(eval_dir.glob("**/evaluation/evaluation_metrics.json"))
        self.assertTrue(len(metrics_files) > 0, f"evaluation_metrics.json was not created for {model_id}.")
        
        responses_files = list(eval_dir.glob("**/evaluation/generated_responses.json"))
        self.assertTrue(len(responses_files) > 0, f"generated_responses.json was not created for {model_id}.")

    # Individual test cases for each model
    def test_gpt2(self):
        self._run_model_test("gpt2")

    def test_qwen2_5(self):
        self._run_model_test("qwen2_5")

    def test_deepseek_r1_1_5b(self):
        self._run_model_test("deepseek_r1_1_5b")

    def test_qwen2(self):
        self._run_model_test("qwen2")

    def test_phi3(self):
        self._run_model_test("phi3")

    def test_gemma2(self):
        self._run_model_test("gemma2")

    def test_mistral(self):
        self._run_model_test("mistral")

    def test_llama3_2(self):
        self._run_model_test("llama3_2")

if __name__ == "__main__":
    unittest.main()
