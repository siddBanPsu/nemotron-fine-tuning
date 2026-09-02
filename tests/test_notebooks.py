from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_four_notebooks_are_valid_clean_and_compilable(self):
        expected = [
            "01_cloud_api_baseline.ipynb",
            "02_local_bf16_baseline.ipynb",
            "03_peft_lora.ipynb",
            "04_full_finetuning_design.ipynb",
        ]
        self.assertEqual(sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb")), expected)
        for name in expected:
            payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

    def test_cloud_notebook_uses_25_rows_per_model_and_execution_scoring(self):
        text = (ROOT / "notebooks/01_cloud_api_baseline.ipynb").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_API_EVALUATION_SIZE", text)
        self.assertIn("2 * len(cloud_rows)", text)
        self.assertIn("score_predictions(generated, data_dir=DATA_DIR)", text)
        self.assertIn("API_PROFILE_OVERRIDE = None", text)
        self.assertIn("API_CONFIG.ultra", text)
        self.assertIn("requests_per_minute=API_CONFIG.requests_per_minute", text)
        self.assertIn("endpoint_model_fingerprint", text)
        self.assertIn("input('Paste NVIDIA API key", text)
        self.assertNotIn("ipywidgets", text)

    def test_local_and_peft_use_vllm_on_the_identical_holdout(self):
        local = (ROOT / "notebooks/02_local_bf16_baseline.ipynb").read_text(encoding="utf-8")
        peft = (ROOT / "notebooks/03_peft_lora.ipynb").read_text(encoding="utf-8")
        baseline = "baseline_local_bf16_text2sql.json"
        self.assertIn("scripts/evaluate_vllm.py", local)
        self.assertIn(baseline, local)
        self.assertIn(baseline, peft)
        self.assertIn("paired_execution_comparison(baseline, tuned)", peft)
        self.assertIn("--max-train-samples', '4096'", peft)
        self.assertIn("--max-steps', '64'", peft)
        self.assertIn("NEMOTRON_PEFT_NUM_GPUS", peft)
        self.assertIn("TP=1, EP={N_GPUS}", peft)
        self.assertNotIn("AutoModelForCausalLM", local)
        self.assertNotIn("transformers.generate", peft)
        for text in (local, peft):
            self.assertIn("host port **8889**", text)
            self.assertIn("print('Python:', sys.executable)", text)
            self.assertIn("NEMOTRON_ARTIFACTS_DIR", text)

    def test_full_training_notebook_is_design_only(self):
        text = (ROOT / "notebooks/04_full_finetuning_design.ipynb").read_text(encoding="utf-8")
        self.assertIn("DESIGN_ONLY = True", text)
        self.assertIn("validate_full_sft_hardware", text)
        self.assertIn("paired execution-accuracy delta", text)
        self.assertNotIn("subprocess.run(full_cmd", text)


if __name__ == "__main__":
    unittest.main()
