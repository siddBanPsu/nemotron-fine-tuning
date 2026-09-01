# Third-party inputs and attribution

## NVIDIA Nemotron 3.5 Lightning

- Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Governing terms: OpenMDW License Agreement 1.1, linked from the model card
- Model revision used by the lab: `b3caaabed0263651a17dc1f2d4ce97e794f76c44`

The training drivers are original workshop glue around public NVIDIA
Megatron-Bridge recipe APIs. Their structure follows the public Lightning 3.5
LoRA/full-SFT cookbooks and uses no redistributed NVIDIA model weights.

Notebook 01 calls NVIDIA's hosted
`nvidia/nemotron-3.5-lightning-30b-a3b` and
`nvidia/nemotron-3-ultra-550b-a55b` trial endpoints. Their API references
identify the served checkpoints as NVFP4 variants. Nemotron 3 Ultra is governed
by OpenMDW 1.1, and endpoint use is subject to NVIDIA's API Trial Terms. API
keys and generated responses are not included in the repository.

## BANKING77

- Authors: Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson,
  and Ivan Vulić
- Paper: “Efficient Intent Detection with Dual Sentence Encoders,” NLP4ConvAI
  2020
- Source: `PolyAI-LDN/task-specific-datasets`
- Pinned source revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- License: Creative Commons Attribution 4.0 International

The repository does not redistribute the dataset. The preparation script
downloads the pinned author-provided `train.csv`, `test.csv`, and
`categories.json`, then records their SHA-256 digests in the generated local
manifest.
