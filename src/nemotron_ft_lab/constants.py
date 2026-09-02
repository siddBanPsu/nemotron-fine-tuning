"""Pinned inputs and workshop defaults for the Text2SQL lab."""

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"

NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_LIGHTNING_MODEL_ID = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_API_LIGHTNING_MODEL_VARIANT = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
NVIDIA_API_ULTRA_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_API_ULTRA_MODEL_VARIANT = "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
NVIDIA_API_MODEL_ID = NVIDIA_API_LIGHTNING_MODEL_ID
NVIDIA_API_MODEL_VARIANT = NVIDIA_API_LIGHTNING_MODEL_VARIANT

# NVIDIA's official Nemotron 3.5 Lightning LoRA Text2SQL cookbook uses both of
# these BIRD train mirrors. Revisions are pinned so reruns do not silently move.
OFFICIAL_COOKBOOK_REPOSITORY = "NVIDIA-NeMo/Nemotron"
OFFICIAL_COOKBOOK_REVISION = "ccbea41e1ccb8a9bda9169ca83f19d18e39b9cdc"
TRAIN_DATASET_ID = "xu3kev/BIRD-SQL-data-train"
TRAIN_DATASET_REVISION = "9122256f9d14752ed80fb9b7d158e21d9f9261aa"
REASONING_DATASET_ID = "meowterspace45/bird-sql-train-with-reasoning"
REASONING_DATASET_REVISION = "9e351e0057819f1b0917debb83c8e12f321157a4"

# BIRD Mini-Dev is disjoint from the train split and includes executable SQLite
# databases. The official archive is ~800 MB compressed.
EVAL_DATASET_NAME = "BIRD Mini-Dev official SQLite package"
EVAL_EXCLUDED_QUESTION_IDS = (701,)
EVALUATION_PROTOCOL_VERSION = 1
MINIDEV_ARCHIVE_FILE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
MINIDEV_ARCHIVE_URL = (
    f"https://drive.usercontent.google.com/download?id={MINIDEV_ARCHIVE_FILE_ID}&export=download&confirm=t"
)
MINIDEV_ARCHIVE_BYTES = 799_944_582
MINIDEV_ARCHIVE_SHA256 = "aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be"

DEFAULT_SEED = 1234
DEFAULT_EVALUATION_SIZE = 100
DEFAULT_API_EVALUATION_SIZE = 25
DEFAULT_MAX_TRAIN_SAMPLES = 4096
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_MAX_STEPS = 64
TRAINING_PROTOCOL_VERSION = 1
