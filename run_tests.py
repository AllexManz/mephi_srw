#!/usr/bin/env python3
"""Run the repository test suite.

Model snapshot tests are opt-in because they download large Hugging Face
models. Use ``RUN_MODEL_TESTS=1 python run_tests.py`` for those tests.
"""

import sys
from pathlib import Path

# Add src and tests directories to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "tests"))

import unittest
def main():
    print("Starting Security LLM test suite...")
    suite = unittest.TestLoader().discover("tests")
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    if result.wasSuccessful():
        print(
            f"Passed: {result.testsRun - len(result.skipped)} | "
            f"Skipped: {len(result.skipped)}"
        )
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
