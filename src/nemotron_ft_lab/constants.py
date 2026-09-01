"""Pinned public inputs and workshop defaults."""

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_LIGHTNING_MODEL_ID = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_API_LIGHTNING_MODEL_VARIANT = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
NVIDIA_API_ULTRA_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_API_ULTRA_MODEL_VARIANT = "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
# Backward-compatible aliases for code that expects the customization model's API identity.
NVIDIA_API_MODEL_ID = NVIDIA_API_LIGHTNING_MODEL_ID
NVIDIA_API_MODEL_VARIANT = NVIDIA_API_LIGHTNING_MODEL_VARIANT
DATASET_ID = "PolyAI/banking77"
DATASET_SOURCE_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
DATASET_SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
DATASET_SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{DATASET_SOURCE_REVISION}/banking_data"
)
DATASET_SOURCE_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}

ROUTE_PREFIX = "B77_"
ROUTE_PERMUTATION_SEED = 20_260_829
DEFAULT_SEED = 1234
DEFAULT_TRAIN_PER_LABEL = 25
DEFAULT_VALIDATION_PER_LABEL = 5
DEFAULT_TEST_PER_LABEL = 3
DEFAULT_MAX_SEQUENCE_LENGTH = 512
