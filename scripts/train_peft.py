#!/usr/bin/env python3
"""LoRA SFT for Nemotron 3.5 Lightning using the official Megatron-Bridge recipe."""

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
from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION


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
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--megatron-checkpoint", default="/workspace/storage/checkpoints/lightning35-megatron")
    parser.add_argument("--data-dir", default="artifacts/data/banking77")
    parser.add_argument("--output-dir", default="/workspace/storage/checkpoints/banking77-lora")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp = 1
    ep = world_size
    data_dir = Path(args.data_dir).resolve()
    examples, packed_sequences = count_packed_sequences(data_dir, args.sequence_length)
    steps_for_epoch = max(1, math.ceil(packed_sequences / args.global_batch_size))
    train_steps = min(args.max_steps, steps_for_epoch)

    cfg = nemotron_3_5_lightning_peft_config("lora")
    cfg.model.hf_model_id = args.hf_model
    cfg.model.hf_model_revision = args.revision or None
    cfg.tokenizer.tokenizer_model = args.hf_model
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": args.revision} if args.revision else {}
    cfg.checkpoint.pretrained_checkpoint = str(Path(args.megatron_checkpoint).resolve())

    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = ep
    cfg.model.seq_length = args.sequence_length
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    if world_size == 1:
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
    cfg.scheduler.lr_warmup_iters = max(1, train_steps // 10)
    cfg.scheduler.lr_decay_iters = train_steps
    cfg.peft.dim = args.lora_rank
    cfg.peft.alpha = args.lora_rank * 2

    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1
    cfg.checkpoint.save = str(Path(args.output_dir).resolve())
    cfg.checkpoint.save_interval = train_steps
    cfg.checkpoint.async_save = False

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"PEFT config: {examples} examples, ~{packed_sequences} packed sequences, "
            f"{train_steps} steps, TP{tp}/EP{ep}, seq={args.sequence_length}, LoRA r={args.lora_rank}",
            flush=True,
        )
    return cfg


def main() -> None:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch train_peft.py with torchrun, even for one GPU.")
    args = parse_args()
    cfg = build_config(args)
    if args.dry_run:
        print("Dry run passed: official PEFT config constructed successfully.")
        return
    finetune(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
