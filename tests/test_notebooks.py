from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_four_notebooks_are_valid_and_code_compiles(self):
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
                    source = "".join(cell["source"])
                    compile(source, f"{name}:cell-{index}", "exec")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

    def test_full_training_notebook_is_design_only(self):
        text = (ROOT / "notebooks/04_full_finetuning_design.ipynb").read_text(encoding="utf-8")
        self.assertIn("DESIGN_ONLY = True", text)
        self.assertIn("validate_full_sft_hardware", text)
        self.assertNotIn("subprocess.run(full_cmd", text)

    def test_peft_uses_the_exact_local_baseline(self):
        api_text = (ROOT / "notebooks/01_cloud_api_baseline.ipynb").read_text(encoding="utf-8")
        api_payload = json.loads(api_text)
        api_code = "\n".join(
            "".join(cell["source"])
            for cell in api_payload["cells"]
            if cell["cell_type"] == "code"
        )
        local_text = (ROOT / "notebooks/02_local_bf16_baseline.ipynb").read_text(encoding="utf-8")
        peft_text = (ROOT / "notebooks/03_peft_lora.ipynb").read_text(encoding="utf-8")
        self.assertIn("load_nvidia_api_config", api_text)
        self.assertIn("API_PROFILE_OVERRIDE = None", api_text)
        self.assertIn("profile=API_PROFILE_OVERRIDE", api_text)
        self.assertIn("API_CONFIG.ultra.model_id", api_text)
        self.assertIn("input('Paste NVIDIA API key", api_code)
        self.assertNotIn("ipywidgets", api_code)
        self.assertNotIn("widgets.Password", api_code)
        self.assertNotIn("widgets.Text", api_code)
        self.assertNotIn("getpass(", api_code)
        self.assertIn("max_retries=0", api_code)
        self.assertIn("API_EXAMPLES_PER_LABEL = 1", api_text)
        self.assertIn("API_EXPERIMENT_PROFILE = 'retrieval'", api_text)
        self.assertIn("NVIDIA_API_REQUESTS_PER_MINUTE = API_CONFIG.requests_per_minute", api_text)
        self.assertIn("'opaque_zero_shot', 'taxonomy', 'retrieved_few_shot'", api_text)
        self.assertIn("LexicalDemonstrationRetriever", api_text)
        self.assertIn("balanced_evaluation_subset", api_text)
        self.assertIn("baseline_api_{artifact_model_name}_nvfp4_", api_text)
        self.assertIn("baseline_local_bf16.json", local_text)
        self.assertIn("BASELINE_PATH = ARTIFACTS_DIR / 'evaluation/baseline_local_bf16.json'", peft_text)
        for text in (api_text, local_text, peft_text):
            self.assertIn("NEMOTRON_ARTIFACTS_DIR", text)
        self.assertIn("CLOUD_BASELINE_GLOB = 'baseline_api_*_nvfp4_*_per_label.json'", peft_text)
        self.assertIn("peft_vs_cloud_targets.json", peft_text)
        self.assertIn("tuned Lightning is better", peft_text)

    def test_gpu_notebooks_require_the_container_jupyter(self):
        for name in ("02_local_bf16_baseline.ipynb", "03_peft_lora.ipynb"):
            text = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
            self.assertIn("host port **8889**", text)
            self.assertIn("print('Python:', sys.executable)", text)


if __name__ == "__main__":
    unittest.main()
