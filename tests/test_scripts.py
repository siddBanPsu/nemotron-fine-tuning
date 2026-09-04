from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_preflight_direct_execution_bootstraps_src_layout(self):
        result = subprocess.run(
            [sys.executable, "-I", "scripts/preflight.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)
        self.assertIn("--model-profile", result.stdout)

    def test_data_and_vllm_entrypoints_expose_help_without_gpu_imports(self):
        for script, option in (
            ("scripts/prepare_text2sql.py", "--evaluation-only"),
            ("scripts/evaluate_vllm.py", "--tensor-parallel-size"),
            ("scripts/show_model_profile.py", "--list"),
        ):
            result = subprocess.run(
                [sys.executable, "-I", script, "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(option, result.stdout)

    def test_vllm_evaluator_checkpoints_predictions_before_scoring(self):
        source = (ROOT / "scripts/evaluate_vllm.py").read_text(encoding="utf-8")
        self.assertLess(source.index("write_jsonl(predictions_path"), source.index("score_predictions("))
        self.assertIn("--force-generation", source)
        self.assertIn('parser.add_argument("--max-num-seqs", type=int, default=64)', source)
        self.assertIn('engine_kwargs["mamba_ssm_cache_dtype"]', source)


if __name__ == "__main__":
    unittest.main()
