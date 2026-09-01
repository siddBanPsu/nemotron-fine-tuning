#!/usr/bin/env python3
"""True full-parameter SFT using NVIDIA's Nemotron 3.5 Lightning recipe."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION
from nemotron_ft_lab.hardware import inspect_cuda, validate_full_sft_hardware


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--megatron-checkpoint", default="/workspace/storage/checkpoints/lightning35-megatron")
    parser.add_argument("--data-dir", default="artifacts/data/banking77")
    parser.add_argument("--output-dir", default="/workspace/storage/checkpoints/banking77-full")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 8:
        raise RuntimeError(f"Full SFT requires at least 8 ranks; torchrun launched {world_size}.")
    tp = 2 if world_size >= 16 else 1
    ep = 8
    if world_size % (tp * ep) != 0:
        raise RuntimeError(f"WORLD_SIZE={world_size} is not divisible by TP({tp}) x EP({ep}).")

    cfg = nemotron_3_5_lightning_sft_config()
    cfg.model.hf_model_id = args.hf_model
    cfg.model.hf_model_revision = args.revision or None
    cfg.tokenizer.tokenizer_model = args.hf_model
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": args.revision} if args.revision else {}
    cfg.checkpoint.pretrained_checkpoint = str(Path(args.megatron_checkpoint).resolve())

    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = ep
    cfg.model.sequence_parallel = tp > 1
    cfg.model.seq_length = args.sequence_length
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "mlp"]

    cfg.dataset.hf_dataset = None
    cfg.dataset.hf_validation_dataset = None
    cfg.dataset.hf_test_dataset = None
    cfg.dataset.hf_output_root = None
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.hf_rewrite = False
    cfg.dataset.dataset_root = str(Path(args.data_dir).resolve())
    cfg.dataset.seq_length = args.sequence_length
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.packed_sequence_size = args.sequence_length
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = max(1, tp)

    cfg.train.train_iters = args.max_steps
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.optimizer.lr = args.learning_rate
    cfg.optimizer.min_lr = 0.0
    cfg.scheduler.lr_warmup_iters = max(1, args.max_steps // 10)
    cfg.scheduler.lr_decay_iters = args.max_steps
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0

    cfg.checkpoint.save = str(Path(args.output_dir).resolve())
    cfg.checkpoint.save_interval = args.max_steps
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = 1

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"FULL SFT config: {args.max_steps} steps, TP{tp}/EP{ep}, "
            f"DP={world_size // (tp * ep)}, seq={args.sequence_length}; all model parameters trainable.",
            flush=True,
        )
    return cfg


def main() -> None:
    args = parse_args()
    validate_full_sft_hardware(inspect_cuda())
    if "RANK" not in os.environ:
        raise RuntimeError("Launch train_full.py with torchrun on the complete multi-GPU node.")
    cfg = build_config(args)
    if args.dry_run:
        print("Dry run passed: official full-SFT config constructed successfully.")
        return
    finetune(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
