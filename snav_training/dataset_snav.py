"""
SNav Stage-1 dataset for LLaVA-Video-7B-Qwen2.

Consumes the per-step RGB frames rendered by
``snav_training/data_generation/render_streamvln.py`` in ``--output_mode frames``
(i.e. StreamVLN-style layout with ``images/<clip>/rgb/*.jpg`` + an
``annotations.json`` index file).

Highlights of the Stage-1 sampler:
  * Uniform frame sampling (first + current + middle).
  * Default num_future_steps=6 (predict 6 actions per chunk).
  * Optional mix with a general Video-QA dataset (LLaVA-Video-178K format)
    to mitigate catastrophic forgetting.

This file is intentionally dependency-light: outside of LLaVA tokenizer
helpers it only relies on torch / numpy / PIL.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

LLAVA_ROOT = os.environ.get("LLAVA_ROOT", "")
if LLAVA_ROOT and LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

from llava.constants import (  # noqa: E402  (import after sys.path tweak)
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402


# ── Action space (VLN-CE standard) ──────────────────────────────────────
IDX2ACT = {0: "STOP", 1: "↑", 2: "←", 3: "→"}

SYSTEM_PROMPT = (
    "You are an autonomous indoor navigation agent. "
    "You observe the environment through sequential RGB frames and follow "
    "natural language instructions to reach a goal location. "
    "At each decision step, predict the next {n} low-level actions from: "
    "FORWARD (↑) moves 25 cm, TURN LEFT (←) rotates 15°, "
    "TURN RIGHT (→) rotates 15°, STOP ends navigation."
)


def actions_to_text(actions: List[int]) -> str:
    return "".join(IDX2ACT.get(a, "?") for a in actions)


def parse_action_text(text: str) -> List[int]:
    act_map = {"↑": 1, "←": 2, "→": 3, "STOP": 0}
    actions: List[int] = []
    i = 0
    while i < len(text):
        if text[i: i + 4] == "STOP":
            actions.append(0)
            i += 4
        elif text[i] in act_map:
            actions.append(act_map[text[i]])
            i += 1
        else:
            i += 1
    return actions


# ── Helpers ─────────────────────────────────────────────────────────────

def _load_annotations(video_folders: List[str]):
    all_items = []
    for vf in video_folders:
        anno_path = os.path.join(vf, "annotations.json")
        if not os.path.exists(anno_path):
            print(f"[WARN] {anno_path} not found, skipping")
            continue
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
        for item in anno:
            item["_video_folder"] = vf
        all_items.extend(anno)
    return all_items


def _sample_frames_uniform(
    current_step: int,
    total_frames: int,
    num_frames: int = 8,
) -> List[int]:
    """Sample frames: always keep first (idx 0) and current, uniformly pick
    the remaining (num_frames - 2) in between.

    Returns exactly ``num_frames`` indices in ascending order.
    """
    first = 0
    last = min(current_step, total_frames - 1)

    if num_frames <= 1:
        return [last]
    if num_frames == 2 or first == last:
        return [first] + [last] * (num_frames - 1)

    mid_count = num_frames - 2
    if last - first - 1 <= mid_count:
        mid = list(range(first + 1, last))
    else:
        mid = np.linspace(first + 1, last - 1, mid_count, dtype=int).tolist()

    while len(mid) < mid_count:
        mid.append(last)
    indices = [first] + mid + [last]
    return indices


def _build_conv(system_text: str, user_msg: str, assistant_msg: Optional[str]):
    conv = conv_templates["qwen_1_5"].copy()
    conv.system = f"<|im_start|>system\n{system_text}"
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], assistant_msg)
    return conv.get_prompt()


# ═══════════════════════════════════════════════════════════════════════
# SNav Stage-1 action-prediction dataset
# ═══════════════════════════════════════════════════════════════════════

class SNavVideoDataset(Dataset):
    """
    Dataset that emits ``(frames, instruction) → next K actions`` samples.

    Expects ``frames`` mode output from ``render_streamvln.py``:

      ``<vf>/annotations.json``          list of items with
        ``{"video": "<rel_dir>", "instructions": [...], "actions": [...]}``
      ``<vf>/<item.video>/rgb/*.jpg``   per-step RGB frames.
    """

    def __init__(
        self,
        video_folders: List[str],
        tokenizer,
        image_processor,
        num_frames: int = 8,
        num_future_steps: int = 6,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.num_future_steps = num_future_steps
        self.rng = random.Random(seed)

        self.samples: List[Dict] = []
        all_items = _load_annotations(video_folders)

        for item in all_items:
            vf = item["_video_folder"]
            instructions = item.get("instructions", [])
            if not isinstance(instructions, list):
                instructions = [instructions]
            raw_actions = item.get("actions", [])
            if len(raw_actions) < 2:
                continue
            effective_actions = raw_actions[1:] + [0]
            N = len(effective_actions)

            rgb_dir = os.path.join(vf, item["video"], "rgb")
            if not os.path.isdir(rgb_dir):
                continue
            total_frames = len(os.listdir(rgb_dir))
            if total_frames == 0:
                continue

            for ins in instructions:
                if not ins or not ins.strip():
                    continue
                for step in range(0, N, num_future_steps):
                    action_chunk = effective_actions[step: step + num_future_steps]
                    if not action_chunk:
                        break
                    while len(action_chunk) < num_future_steps:
                        action_chunk.append(0)

                    frame_indices = _sample_frames_uniform(
                        current_step=step,
                        total_frames=total_frames,
                        num_frames=num_frames,
                    )

                    self.samples.append({
                        "video_root": vf,
                        "video_rel": item["video"],
                        "instruction": ins.strip(),
                        "frame_indices": frame_indices,
                        "actions": action_chunk,
                    })

        self.rng.shuffle(self.samples)
        print(f"[SNavVideoDataset] {len(self.samples)} samples "
              f"from {len(video_folders)} folder(s) "
              f"(predict {num_future_steps} actions per chunk)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        rgb_dir = os.path.join(sample["video_root"], sample["video_rel"], "rgb")
        frame_files = sorted(os.listdir(rgb_dir))

        frames: List[Image.Image] = []
        for i in sample["frame_indices"]:
            fi = min(i, len(frame_files) - 1)
            img = Image.open(os.path.join(rgb_dir, frame_files[fi])).convert("RGB")
            frames.append(img)

        video = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]

        system_text = SYSTEM_PROMPT.format(n=self.num_future_steps)
        user_msg = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"These frames show your navigation history. "
            f"Instruction: {sample['instruction']}\n"
            f"Predict the next {self.num_future_steps} actions."
        )
        answer = actions_to_text(sample["actions"])

        full_prompt = _build_conv(system_text, user_msg, answer)
        prefix_prompt = _build_conv(system_text, user_msg, None)

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        prefix_ids = tokenizer_image_token(
            prefix_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        labels = input_ids.clone()
        labels[: len(prefix_ids)] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": video,
            "image_size": frames[0].size,
            "modality": "video",
        }


# ═══════════════════════════════════════════════════════════════════════
# Optional general Video-QA companion dataset
# ═══════════════════════════════════════════════════════════════════════

QA_SYSTEM_PROMPT = (
    "You are a helpful video understanding assistant. "
    "Watch the video carefully and answer the question."
)


def _load_video_frames(
    video_path: str,
    num_frames: int,
    image_processor,
) -> Optional[torch.Tensor]:
    """Load a video file and uniformly sample ``num_frames`` as PIL images."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total == 0:
            return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        raw_frames = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f).convert("RGB") for f in raw_frames]
    except ImportError:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames or 300
        if total <= 0:
            total = 300
        target_indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_indices:
                frames.append(frame.to_image().convert("RGB"))
            if len(frames) >= num_frames:
                break
        container.close()
        if len(frames) == 0:
            return None

    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    return video_tensor


def _resolve_video_path(video_root: str, video_rel: str) -> Optional[str]:
    """Resolve video path robustly across different extracted layouts."""
    candidates = [
        os.path.join(video_root, video_rel),
        os.path.join(video_root, os.path.basename(video_rel)),
    ]
    rel_parts = video_rel.split("/")
    if len(rel_parts) > 1:
        candidates.append(os.path.join(video_root, *rel_parts[1:]))
    # Common extracted directory wrappers (LLaVA-Video-178K naming).
    candidates.append(os.path.join(video_root, "ActivityNet-QA", video_rel))
    candidates.append(os.path.join(video_root, "NextQA", video_rel))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


class VideoQADataset(Dataset):
    """LLaVA-Video-178K style Video-QA JSON reader (optional companion)."""

    def __init__(
        self,
        qa_json_paths: List[str],
        video_root_dirs: List[str],
        tokenizer,
        image_processor,
        num_frames: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.rng = random.Random(seed)

        self.samples: List[Dict] = []
        for json_path, vroot in zip(qa_json_paths, video_root_dirs):
            if not os.path.isfile(json_path):
                print(f"[VideoQADataset] WARN: {json_path} not found, skip")
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            loaded = 0
            for item in items:
                convs = item.get("conversations", [])
                if len(convs) < 2:
                    continue
                question_raw = convs[0].get("value", "")
                answer = convs[1].get("value", "")
                if not answer.strip():
                    continue
                question = question_raw.replace("<image>", "").strip()
                if question.startswith("\n"):
                    question = question[1:]

                video_rel = item.get("video", "")
                video_path = _resolve_video_path(vroot, video_rel)
                if video_path is None:
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "question": question.strip(),
                    "answer": answer.strip(),
                })
                loaded += 1
            print(f"[VideoQADataset] Loaded {loaded} from {json_path}")

        self.rng.shuffle(self.samples)
        print(f"[VideoQADataset] Total: {len(self.samples)} QA samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        video = _load_video_frames(
            sample["video_path"], self.num_frames, self.image_processor
        )
        if video is None:
            return self[self.rng.randint(0, len(self) - 1)]

        frame_h, frame_w = video.shape[-2], video.shape[-1]

        system_text = QA_SYSTEM_PROMPT
        user_msg = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"{sample['question']}"
        )
        answer = sample["answer"]

        full_prompt = _build_conv(system_text, user_msg, answer)
        prefix_prompt = _build_conv(system_text, user_msg, None)

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        prefix_ids = tokenizer_image_token(
            prefix_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        labels = input_ids.clone()
        labels[: len(prefix_ids)] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": video,
            "image_size": (frame_w, frame_h),
            "modality": "video",
        }


# ═══════════════════════════════════════════════════════════════════════
# Mixed-dataset builder
# ═══════════════════════════════════════════════════════════════════════

class _RepeatDataset(Dataset):
    """Repeat a smaller dataset to reach a target length (for balancing)."""

    def __init__(self, dataset: Dataset, target_len: int):
        self.dataset = dataset
        self.target_len = target_len

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class _TakeDataset(Dataset):
    """Deterministic subset view with replacement-safe fixed indices."""

    def __init__(self, dataset: Dataset, target_len: int, seed: int = 42):
        self.dataset = dataset
        self.target_len = target_len
        rng = random.Random(seed)
        if target_len <= len(dataset):
            self.indices = rng.sample(range(len(dataset)), target_len)
        else:
            self.indices = [rng.randrange(len(dataset)) for _ in range(target_len)]

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def build_mixed_dataset(
    snav_dataset: SNavVideoDataset,
    qa_dataset: Optional[VideoQADataset],
    qa_ratio: float = 0.15,
    seed: int = 42,
) -> Dataset:
    """Mix SNav navigation data with general QA data.

    qa_ratio: fraction of total that should be QA (0.15 = 15% QA, 85% SNav).
    """
    if qa_dataset is None or len(qa_dataset) == 0:
        print("[build_mixed_dataset] No QA data, returning SNav only")
        return snav_dataset

    n_snav = len(snav_dataset)
    n_qa_target = int(n_snav * qa_ratio / (1.0 - qa_ratio))
    n_qa_target = max(n_qa_target, 1)

    if len(qa_dataset) < n_qa_target:
        qa_ds = _RepeatDataset(qa_dataset, n_qa_target)
    else:
        qa_ds = _TakeDataset(qa_dataset, n_qa_target, seed=seed)

    mixed = ConcatDataset([snav_dataset, qa_ds])
    print(
        f"[build_mixed_dataset] SNav={n_snav} + QA={n_qa_target} "
        f"= {len(mixed)} total ({qa_ratio*100:.0f}% QA)"
    )
    return mixed


# ═══════════════════════════════════════════════════════════════════════
# Collator
# ═══════════════════════════════════════════════════════════════════════

class SNavVideoCollator:
    def __init__(self, tokenizer, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, instances: List[Dict]) -> Dict:
        input_ids = [inst["input_ids"][: self.max_length] for inst in instances]
        labels = [inst["labels"][: self.max_length] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        images = [inst["image"] for inst in instances]
        image_sizes = [inst["image_size"] for inst in instances]
        modalities = [inst["modality"] for inst in instances]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "image_sizes": image_sizes,
            "modalities": modalities,
        }
