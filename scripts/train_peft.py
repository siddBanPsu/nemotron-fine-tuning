#!/usr/bin/env python3
"""Profile-aware LoRA SFT for the Nano workshop and Lightning advanced paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_5_lightning_peft_config,
    nemotron_nano_9b_v2_peft_config,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

from nemotron_ft_lab.constants import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    MEGATRON_BRIDGE_REVISION,
    OFFICIAL_COOKBOOK_REVISION,
)
from nemotron_ft_lab.data import sha256_file
from nemotron_ft_lab.megatron_compat import configure_expert_bias_padding_mask_compatibility
from nemotron_ft_lab.model_profiles import LIGHTNING35_ADVANCED, get_model_profile


def count_packed_sequences(data_dir: Path, sequence_length: int) -> tuple[int, int]:
    total_tokens = 0
    examples = 0
    with (data_dir / "training.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total_tokens += int(row["length"])
            examples += 1
    return examples, max(1, math.ceil(total_tokens / sequence_length))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    artifacts_dir = Path(os.environ.get("NEMOTRON_ARTIFACTS_DIR", "artifacts"))
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--hf-model")
    parser.add_argument("--revision")
    parser.add_argument("--megatron-checkpoint")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = get_model_profile(args.model_profile)
    args.model_profile = profile.name
    args.hf_model = args.hf_model or profile.model_id
    args.revision = profile.revision if args.revision is None else args.revision
    args.megatron_checkpoint = args.megatron_checkpoint or str(
        Path("/workspace/storage/checkpoints") / profile.megatron_checkpoint_name
    )
    args.data_dir = args.data_dir or str(
        artifacts_dir / "data/bird-text2sql/profiles" / profile.name
    )
    args.output_dir = args.output_dir or str(
        Path("/workspace/storage/checkpoints") / profile.lora_checkpoint_name
    )
    args.global_batch_size = args.global_batch_size or profile.default_global_batch_size
    args.max_steps = profile.default_max_steps if args.max_steps is None else args.max_steps
    return args


def build_config(args: argparse.Namespace):
    profile = get_model_profile(args.model_profile)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in profile.peft_world_sizes:
        allowed = ", ".join(str(value) for value in profile.peft_world_sizes)
        raise RuntimeError(
            f"{profile.name} supports torchrun world sizes [{allowed}]; received {world_size}."
        )
    tp = 1
    ep = world_size if profile == LIGHTNING35_ADVANCED else 1
    data_dir = Path(args.data_dir).resolve()
    examples, packed_sequences = count_packed_sequences(data_dir, args.sequence_length)
    steps_for_epoch = max(1, math.ceil(packed_sequences / args.global_batch_size))
    train_steps = steps_for_epoch * args.epochs
    if args.max_steps > 0:
        train_steps = min(args.max_steps, train_steps)

    training_path = data_dir / "training.jsonl"
    training_manifest_path = data_dir / "training_manifest.json"
    if not training_manifest_path.is_file():
        raise RuntimeError(f"Missing training data manifest: {training_manifest_path}")
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_sha256 = sha256_file(training_path)
    if training_manifest.get("training_sha256") != training_sha256:
        raise RuntimeError("training.jsonl no longer matches its preparation manifest.")
    expected_training_identity = {
        "model_profile": profile.name,
        "model": args.hf_model,
        "revision": args.revision,
        "system_prompt": profile.system_prompt,
        "include_reasoning": profile.include_reasoning,
    }
    mismatched = {
        key: (training_manifest.get(key), value)
        for key, value in expected_training_identity.items()
        if training_manifest.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            f"Training data does not match model profile {profile.name}: {mismatched}. "
            "Regenerate the profile-specific training directory."
        )

    output_dir = Path(args.output_dir).resolve()
    checkpoint_markers = (
        output_dir / "latest_checkpointed_iteration.txt",
        output_dir / "latest_train_state.pt",
    )
    has_checkpoint = any(path.exists() for path in checkpoint_markers)
    checkpoint_save_interval = min(8, train_steps) if profile.name == "nano9b_workshop" else train_steps
    run_contract = {
        "model_profile": profile.name,
        "recipe_name": profile.recipe_name,
        "model": args.hf_model,
        "revision": args.revision,
        "cookbook_revision": OFFICIAL_COOKBOOK_REVISION,
        "megatron_bridge_revision": MEGATRON_BRIDGE_REVISION,
        "training_sha256": training_sha256,
        "system_prompt": profile.system_prompt,
        "include_reasoning": profile.include_reasoning,
        "examples": examples,
        "estimated_packed_sequences": packed_sequences,
        "sequence_length": args.sequence_length,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "train_steps": train_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank,
        "world_size": world_size,
        "tensor_parallel_size": tp,
        "expert_parallel_size": ep,
        "single_gpu_mtp_reduction": profile == LIGHTNING35_ADVANCED and world_size == 1,
        "checkpoint_save_interval": checkpoint_save_interval,
    }
    run_manifest_path = output_dir / "nemotron_ft_lab_run.json"
    if run_manifest_path.exists():
        existing_contract = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise RuntimeError(
                f"Existing checkpoint contract differs from this run: {run_manifest_path}. "
                "Choose a new --output-dir rather than resuming incompatible training."
            )
    elif has_checkpoint:
        raise RuntimeError(
            f"Existing checkpoint has no lab run contract: {output_dir}. "
            "Choose a new --output-dir to avoid an ambiguous resume."
        )
    elif int(os.environ.get("RANK", "0")) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path.write_text(json.dumps(run_contract, indent=2) + "\n", encoding="utf-8")

    if has_checkpoint and not args.resume:
        raise RuntimeError(
            f"A checkpoint already exists at {output_dir}. Pass --resume to continue it, "
            "or choose a new --output-dir for a fresh experiment."
        )

    if profile == LIGHTNING35_ADVANCED:
        cfg = nemotron_3_5_lightning_peft_config("lora")
    else:
        cfg = nemotron_nano_9b_v2_peft_config("lora")
    cfg.model.hf_model_id = args.hf_model
    cfg.model.hf_model_revision = args.revision or None
    cfg.tokenizer.tokenizer_model = args.hf_model
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": args.revision} if args.revision else {}
    cfg.checkpoint.pretrained_checkpoint = str(Path(args.megatron_checkpoint).resolve())

    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = ep
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = args.sequence_length
    if profile == LIGHTNING35_ADVANCED:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
    if profile == LIGHTNING35_ADVANCED and world_size == 1:
        # The official single-H100 cookbook uses the checkpoint's one physical MTP head.
        cfg.model.mtp_num_layers = None

    cfg.dataset.hf_dataset = None
    cfg.dataset.hf_validation_dataset = None
    cfg.dataset.hf_test_dataset = None
    cfg.dataset.hf_output_root = None
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.hf_rewrite = False
    cfg.dataset.dataset_root = str(data_dir)
    cfg.dataset.seq_length = args.sequence_length
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.packed_sequence_size = args.sequence_length
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = 1

    cfg.train.train_iters = train_steps
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.optimizer.lr = args.learning_rate
    cfg.optimizer.min_lr = args.learning_rate / 10
    cfg.scheduler.min_lr = args.learning_rate / 10
    cfg.scheduler.lr_warmup_iters = max(1, train_steps // 10)
    cfg.scheduler.lr_decay_iters = train_steps
    cfg.peft.dim = args.lora_rank
    cfg.peft.alpha = args.lora_rank

    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1
    cfg.checkpoint.save = str(output_dir)
    cfg.checkpoint.load = str(output_dir) if has_checkpoint and args.resume else None
    cfg.checkpoint.save_interval = checkpoint_save_interval
    cfg.checkpoint.async_save = False

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"PEFT config ({profile.name}): {examples} examples, ~{packed_sequences} packed sequences, "
            f"{train_steps} steps, TP{tp}/EP{ep}, seq={args.sequence_length}, "
            f"LoRA r={args.lora_rank}, cookbook={OFFICIAL_COOKBOOK_REVISION}",
            flush=True,
        )
    return cfg


def main() -> None:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch train_peft.py with torchrun, even for one GPU.")
    args = parse_args()
    cfg = build_config(args)
    if get_model_profile(args.model_profile) == LIGHTNING35_ADVANCED:
        from megatron.core.transformer.moe.router import TopKRouter

        compatibility_mode = configure_expert_bias_padding_mask_compatibility(TopKRouter)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Megatron expert-bias padding-mask mode: {compatibility_mode}", flush=True)
    if args.dry_run:
        print("Dry run passed: official PEFT config constructed successfully.")
        return
    finetune(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
