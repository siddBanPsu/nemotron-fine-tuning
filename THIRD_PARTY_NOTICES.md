# Third-party inputs and attribution

## NVIDIA Nemotron Nano 9B v2

- Workshop model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- Pinned model revision: `6533e8de2c68e4536bf7c411d7a3ce5734111476`
- Model terms: NVIDIA Open Model License Agreement, linked from the model card
- Megatron-Bridge runtime revision: `8bc33cd2ca1cd044e1520130f4c5e1f5e0181434`

The model card specifies `/no_think` for reasoning-off prompts, greedy decoding
in that mode, and a float32 Mamba SSM cache for accurate vLLM inference. This
repository applies those settings to both the Nano BF16 baseline and merged
LoRA evaluation. The pinned Bridge revision contains the one-GPU Nano PEFT
recipe; no model weights are redistributed.

## NVIDIA Nemotron 3.5 Lightning and Ultra

- Customization model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Pinned model revision: `b3caaabed0263651a17dc1f2d4ce97e794f76c44`
- Model terms: OpenMDW License Agreement 1.1, linked from the model card
- Official Text2SQL cookbook: `NVIDIA-NeMo/Nemotron`
- Cookbook revision followed by the lab: `ccbea41e1ccb8a9bda9169ca83f19d18e39b9cdc`
- Megatron-Bridge runtime revision: `8bc33cd2ca1cd044e1520130f4c5e1f5e0181434`

The preparation/training drivers are workshop glue around NVIDIA's public
Megatron-Bridge recipe APIs. They preserve the official Text2SQL prompt layout,
packed-sequence LoRA design, recipe-owned target modules, checkpoint conversion,
merge, and vLLM serving path. No NVIDIA model weights are redistributed.

Notebook 01 optionally calls NVIDIA's hosted
`nvidia/nemotron-3.5-lightning-30b-a3b` and
`nvidia/nemotron-3-ultra-550b-a55b` endpoints. Endpoint use is subject to
NVIDIA's applicable API trial/service terms. API keys and generated responses
are not included in the repository.

## BIRD Text2SQL

- Authors: Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li,
  Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, and collaborators
- Paper: “Can LLM Already Serve as A Database Interface? A BIg Bench for
  Large-Scale Database Grounded Text-to-SQLs,” NeurIPS 2023
- Direct train mirror: `xu3kev/BIRD-SQL-data-train`
- Pinned direct-train revision: `9122256f9d14752ed80fb9b7d158e21d9f9261aa`
- Reasoning train mirror: `meowterspace45/bird-sql-train-with-reasoning`
- Pinned reasoning revision: `9e351e0057819f1b0917debb83c8e12f321157a4`
- Evaluation dataset: official BIRD Mini-Dev SQLite package
- Mini-Dev terms: Creative Commons Attribution-ShareAlike 4.0

The BIRD project distributes the Mini-Dev questions and SQLite databases in one
official package. The preparation script downloads that package from BIRD's
Google Drive file ID, uses its own `mini_dev_sqlite.json` as the sole evaluation
source, records its byte size and SHA-256 in the local manifest, and does not
commit it.

The `xu3kev` direct-train mirror does not declare a license in its Hugging Face
metadata at the pinned revision. The reasoning mirror's card describes its
reasoning traces as Apache-2.0, but its structured license metadata is unset.
Users remain responsible for reviewing upstream dataset/model terms for their
intended use. This repository does not redistribute either training mirror.

## SQLGlot

SQLGlot is used under its MIT license to parse and normalize generated SQLite
queries before read-only execution. See `sqlglot==30.17.0` in
`requirements-lab.txt`.
