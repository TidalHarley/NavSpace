#!/usr/bin/env python3
"""
SNav Stage-1 vanilla SFT driver for LLaVA-Video-7B-Qwen2.

Self-contained reference trainer built on top of Hugging Face's ``Trainer``
and DeepSpeed. Consumes the *StreamVLN-style* frame dump produced by
``snav_training/data_generation/render_streamvln.py`` with
``--output_mode frames``:

  ``<video_folder>/annotations.json``
  ``<video_folder>/<clip>/rgb/*.jpg``

This trainer is the "reference" Python entry — it is smaller and easier to
read than LLaVA's native ``train_mem.py`` pipeline, but it only covers the
Stage-1 vanilla SFT regime. It does NOT perform any data augmentation,
DAgger, or Stage-2/3 correction flows.

Usage (single-node, 8 GPUs):

    LLAVA_ROOT=/abs/path/to/StreamVLN  \
    deepspeed --num_gpus=8  snav_training/train_snav.py  \
        --model_path /abs/path/to/LLaVA-Video-7B-Qwen2   \
        --video_folders /abs/path/to/snav_data/r2rce,/abs/path/to/snav_data/rxrce,/abs/path/to/snav_data/envdrop  \
        --output_dir /abs/path/to/checkpoints/snav_stage1_vanilla  \
        --deepspeed snav_training/configs/deepspeed_zero2.json
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import Trainer, TrainingArguments

# ── Make sibling modules + external LLaVA importable ─────────────────────
LLAVA_ROOT = os.environ.get("LLAVA_ROOT", "")
if LLAVA_ROOT and LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from llava.mm_utils import get_model_name_from_path  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from dataset_snav import (  # noqa: E402
    SNavVideoCollator,
    SNavVideoDataset,
    VideoQADataset,
    build_mixed_dataset,
)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNav Stage-1 vanilla SFT — LLaVA-Video-7B")
    parser.add_argument("--model_path", required=True,
                        help="Base LLaVA-Video model directory")
    parser.add_argument("--video_folders", required=True,
                        help="Comma-separated list of directories that each contain "
                             "an annotations.json + per-clip rgb/ subdirs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="")

    # Sampling / dataset knobs
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_future_steps", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=32768)

    # Optional general Video-QA companion set
    parser.add_argument("--qa_json_paths", type=str, default="",
                        help="Comma-separated QA annotation JSON paths (LLaVA-Video-178K)")
    parser.add_argument("--qa_video_roots", type=str, default="",
                        help="Comma-separated root dirs for QA videos (1:1 with qa_json_paths)")
    parser.add_argument("--qa_ratio", type=float, default=0.15,
                        help="Fraction of training data that should be QA samples "
                             "(0.15=15%%). Ignored if no QA data is provided.")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=1,
                        help="per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.075)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention backend for the base model")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def main():
    args = build_parser().parse_args()

    if not args.logging_dir:
        args.logging_dir = os.path.join(args.output_dir, "logs")

    video_folders = [p.strip() for p in args.video_folders.split(",") if p.strip()]
    rank = int(os.environ.get("LOCAL_RANK", 0))

    qa_json_list = [p.strip() for p in args.qa_json_paths.split(",") if p.strip()]
    qa_vroot_list = [p.strip() for p in args.qa_video_roots.split(",") if p.strip()]

    if rank == 0:
        print("=" * 60)
        print("SNav Stage-1 Vanilla SFT — LLaVA-Video-7B-Qwen2")
        print("=" * 60)
        print(f"  Model         : {args.model_path}")
        print(f"  Data          : {video_folders}")
        print(f"  QA data       : {qa_json_list or 'none'}")
        print(f"  QA ratio      : {args.qa_ratio:.0%}")
        print(f"  Output        : {args.output_dir}")
        print(f"  Num frames    : {args.num_frames}")
        print(f"  Future steps  : {args.num_future_steps}")
        print(f"  LR            : {args.lr}")
        print(f"  Batch/GPU     : {args.batch_size}")
        print(f"  Grad accum    : {args.grad_accum}")
        print(f"  DeepSpeed     : {args.deepspeed}")
        print(f"  Attn impl     : {args.attn_implementation}")
        print("=" * 60)

    model_name = get_model_name_from_path(args.model_path)
    if rank == 0:
        print(f"[INFO] Loading base model (name={model_name}) ...")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device_map=None,
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
    )

    model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_p, train_p = count_params(model)
    if rank == 0:
        print(f"[INFO] Params total={total_p / 1e6:.1f}M  train={train_p / 1e6:.1f}M")

    snav_dataset = SNavVideoDataset(
        video_folders=video_folders,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        seed=args.seed,
    )

    qa_dataset = None
    if qa_json_list and qa_vroot_list and len(qa_json_list) == len(qa_vroot_list):
        qa_dataset = VideoQADataset(
            qa_json_paths=qa_json_list,
            video_root_dirs=qa_vroot_list,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_frames=args.num_frames,
            seed=args.seed,
        )

    dataset = build_mixed_dataset(
        snav_dataset, qa_dataset, qa_ratio=args.qa_ratio, seed=args.seed
    )
    collator = SNavVideoCollator(tokenizer, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        report_to=["tensorboard"],
        seed=args.seed,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if rank == 0:
        n_gpu = max(1, torch.cuda.device_count())
        eff_batch = args.batch_size * args.grad_accum * n_gpu
        print(f"[INFO] Dataset size       : {len(dataset)}")
        print(f"[INFO] Effective batch    : {eff_batch}")
        print("[INFO] Starting training ...")

    resume_ckpt = None
    if os.path.isdir(args.output_dir):
        ckpts = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if ckpts:
            resume_ckpt = os.path.join(args.output_dir, ckpts[-1])
            if rank == 0:
                print(f"[INFO] Resuming from {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    if rank == 0:
        print(f"[INFO] Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if rank == 0:
        print("[DONE] SNav Stage-1 vanilla SFT complete!")


if __name__ == "__main__":
    main()
