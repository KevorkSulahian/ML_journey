Title: Quantizing CLIP to W4A8 with NVIDIA ModelOpt
Tags: AI, Machine Learning, Quantization, Hugging Face, Computer Vision
Canonical artifact: https://huggingface.co/herooooooooo/clip-vit-l-14-laion2b-coco-w4a8-modelopt

# Quantizing CLIP to W4A8 with NVIDIA ModelOpt

This is both a runnable experiment and an article of how to use Nvidia's new model optimization and quantization library.

It walks through post-training quantization of CLIP with NVIDIA ModelOpt, evaluates COCO
image-text retrieval quality, saves the checkpoint, and includes optional
Hugging Face upload and reload paths.

## Results-First Summary

- **Base model:** [`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)
- **Quantized checkpoint and artifacts:** [`herooooooooo/clip-vit-l-14-laion2b-coco-w4a8-modelopt`](https://huggingface.co/herooooooooo/clip-vit-l-14-laion2b-coco-w4a8-modelopt)
- **Quantization:** W4A8 post-training quantization with ModelOpt AWQ-style calibration.
- **W4A8 definition:** INT4 blockwise weights plus FP8-style activation quantization in this ModelOpt configuration.

## 1. Why

I wanted to reproduce a practical vision-language model quantization workflow
locally. NVIDIA's CLIP PTQ article demonstrates FP8 post-training quantization.
Here I adapt that workflow to W4A8 with ModelOpt's AWQ-style calibration.

The useful question is not only whether the quantized model runs. This
experiment measures image-to-text and text-to-image retrieval recall on
MS-COCO so that the quality tradeoff is visible and reproducible.

## 2. Environment and Reproducibility Checks

Ask an AI coding to set you up :)

```python
import importlib.metadata as metadata
import os
import platform
import shutil
import subprocess
import sys

import torch
```

## 3. Imports and Configuration

These variables are intentionally editable. The full experiment uses 1,024
COCO calibration caption-image pairs and all 5,000 COCO validation images.

By default, calibration uses `train2017` and retrieval evaluation uses
`val2017`, so calibration examples are not drawn from the evaluation set. For a
quick smoke test with only `val2017` downloaded, set `CALIB_SPLIT = "val2017"`
and choose non-overlapping start indexes.

```python
import copy
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPAttention

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
COCO_ROOT = Path.home() / "data/coco"
RUN_DIR = Path("./runs/train2017calib_w4a8_1024calib_val2017_5000eval_no_patch")
MODEL_DIR = RUN_DIR / "model"

CALIB_SPLIT = "train2017"
EVAL_SPLIT = "val2017"
CALIB_N = 1024
EVAL_IMAGES = 5000
CALIB_START_INDEX = 0
EVAL_START_IMAGE_INDEX = 0
CALIB_BATCH_SIZE = 1
IMAGE_BATCH_SIZE = 4
TEXT_BATCH_SIZE = 16
NUM_WORKERS = 4
DISABLE_PATCH_EMBEDDING = True

if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires CUDA.")
DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
```

## 4. COCO Dataset Utilities

```python
class CocoCaptionPairs(Dataset):
    """
    COCO image-caption pairs for calibration.

    Each item returns:
      pixel_values
      input_ids
      attention_mask
      image_id
      caption
    """

    def __init__(
        self,
        coco_root: str,
        split: str,
        tokenizer: CLIPTokenizer,
        image_processor: CLIPImageProcessor,
        max_items: Optional[int] = None,
        start_index: int = 0,
    ):
        self.coco_root = os.path.expanduser(coco_root)
        self.split = split
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        ann_path = os.path.join(
            self.coco_root,
            "annotations",
            f"captions_{split}.json",
        )
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        img_dir = os.path.join(self.coco_root, split)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.images_by_id = {img["id"]: img for img in data["images"]}
        anns = data["annotations"]

        if start_index:
            anns = anns[start_index:]

        if max_items is not None:
            anns = anns[:max_items]

        self.anns = anns

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann = self.anns[idx]
        image_id = ann["image_id"]
        caption = ann["caption"]

        file_name = self.images_by_id[image_id]["file_name"]
        img_path = os.path.join(self.coco_root, self.split, file_name)

        image = Image.open(img_path).convert("RGB")

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"][0]

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "image_id": image_id,
            "caption": caption,
        }


def collate_calib(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }
```

```python
@dataclass
class CocoRetrievalSet:
    image_paths: List[str]
    image_ids: List[int]
    captions: List[str]
    caption_to_image_index: List[int]
    image_to_caption_indices: Dict[int, List[int]]


def build_coco_retrieval_set(
    coco_root: str,
    split: str,
    max_images: int,
    start_image_index: int = 0,
) -> CocoRetrievalSet:
    """
    Builds a COCO retrieval eval set.

    For N images:
      - image_paths has N images
      - captions has roughly 5N captions
      - image_to_caption_indices maps each image to all positive captions
      - caption_to_image_index maps each caption to its positive image
    """
    coco_root = os.path.expanduser(coco_root)

    ann_path = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    img_dir = os.path.join(coco_root, split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    with open(ann_path, "r") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann["caption"])

    all_image_ids = list(anns_by_image.keys())
    selected_image_ids = all_image_ids[start_image_index:start_image_index + max_images]

    image_paths = []
    image_ids = []
    captions = []
    caption_to_image_index = []
    image_to_caption_indices = defaultdict(list)

    for img_idx, image_id in enumerate(selected_image_ids):
        file_name = images_by_id[image_id]["file_name"]
        image_paths.append(os.path.join(coco_root, split, file_name))
        image_ids.append(image_id)

        for caption in anns_by_image[image_id]:
            cap_idx = len(captions)
            captions.append(caption)
            caption_to_image_index.append(img_idx)
            image_to_caption_indices[img_idx].append(cap_idx)

    return CocoRetrievalSet(
        image_paths=image_paths,
        image_ids=image_ids,
        captions=captions,
        caption_to_image_index=caption_to_image_index,
        image_to_caption_indices=dict(image_to_caption_indices),
    )
```

## 5. ModelOpt quantization configuration

W4A8 in this ModelOpt setup uses INT4 blockwise weights and FP8-style
activation quantization. AWQ-lite calibration searches smoothing parameters.
The patch embedding can be excluded from quantization.

Configuration order matters because later matching rules can override earlier
rules. The projection and optional patch-embedding exclusions are appended
last deliberately.

```python
def disable_unusable_deepspeed_plugin():
    """
    ModelOpt's optional DeepSpeed callback imports DeepSpeed during PTQ. A
    runtime-only CUDA install is sufficient for this script but can make that
    import fail while DeepSpeed probes for CUDA build tools.
    """
    from modelopt.torch.quantization.plugins.custom import CUSTOM_POST_CONVERSION_PLUGINS
    from modelopt.torch.quantization.plugins.transformers import make_deepspeed_compatible

    try:
        from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict  # noqa: F401
    except ImportError:
        return
    except Exception as exc:
        import accelerate.utils.other as accelerate_other

        CUSTOM_POST_CONVERSION_PLUGINS.discard(make_deepspeed_compatible)
        accelerate_other.is_deepspeed_available = lambda: False
        print()
        print("Warning: disabling unusable optional DeepSpeed integration.")
        print(f"DeepSpeed import failed: {exc}")


def maybe_register_clip_attention_quantizer():
    """
    NVIDIA's article registers CLIPAttention to a quantized attention wrapper.
    Some ModelOpt versions have this plugin path, some may move it.
    If unavailable, we continue with Linear/module quantization only.
    """
    try:
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantAttention

        mtq.QuantModuleRegistry.register({CLIPAttention: "CLIPAttention"})(_QuantAttention)
        print("Registered CLIPAttention -> _QuantAttention")
        return True
    except Exception as exc:
        print("Warning: could not register _QuantAttention plugin.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("Continuing without attention BMM quantizers.")
        return False


def build_w4a8_clip_cfg(has_quant_attention: bool, disable_patch_embedding: bool):
    """
    W4A8 in ModelOpt:
      - INT4 blockwise weights
      - FP8 activations
      - AWQ-lite calibration

    This starts from ModelOpt's built-in W4A8_AWQ_BETA_CFG and appends CLIP-specific
    exclusions and optional attention-path FP8 quantizers.
    """
    if not hasattr(mtq, "W4A8_AWQ_BETA_CFG"):
        raise RuntimeError(
            "Your installed ModelOpt does not expose mtq.W4A8_AWQ_BETA_CFG. "
            "Upgrade nvidia-modelopt or run with --quant fp8."
        )

    cfg = copy.deepcopy(mtq.W4A8_AWQ_BETA_CFG)

    if "quant_cfg" not in cfg:
        raise RuntimeError("Unexpected ModelOpt W4A8 config format: missing quant_cfg")

    if not isinstance(cfg["quant_cfg"], list):
        raise RuntimeError(
            "This script expects W4A8_AWQ_BETA_CFG['quant_cfg'] to be a list. "
            f"Got: {type(cfg['quant_cfg'])}"
        )

    if has_quant_attention:
        cfg["quant_cfg"].extend([
            {
                "quantizer_name": "*[qkv]_bmm_quantizer",
                "cfg": {
                    "num_bits": (4, 3),
                    "axis": None,
                    "trt_high_precision_dtype": "Half",
                },
            },
            {
                "quantizer_name": "*bmm2_output_quantizer",
                "cfg": {
                    "num_bits": (4, 3),
                    "axis": None,
                    "trt_high_precision_dtype": "Half",
                },
            },
        ])

    # Conservative CLIP exclusions.
    # These are appended last because quant_cfg entries are ordered and later
    # entries override earlier entries.
    cfg["quant_cfg"].extend([
        {"quantizer_name": "*visual_projection*", "enable": False},
        {"quantizer_name": "*text_projection*", "enable": False},
    ])

    if disable_patch_embedding:
        cfg["quant_cfg"].append({"quantizer_name": "*patch_embedding*", "enable": False})

    return cfg
```

## 6. Evaluation functions

Transformers 5 returns a structured CLIP output where older versions returned
a tensor. The small unwrapping helper keeps both APIs working. Encoded features
are accumulated on CPU to limit memory use, then explicitly moved to CUDA for
the similarity matrix multiplication.

```python
@torch.no_grad()
def unwrap_clip_features(features):
    """Handle tensor returns from Transformers 4 and model outputs from Transformers 5."""
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


@torch.no_grad()
def encode_images(
    model,
    image_processor,
    image_paths: List[str],
    batch_size: int,
    device: str,
    use_amp: bool = True,
):
    features = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="encode images"):
        paths = image_paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in paths]

        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.get_image_features(pixel_values=pixel_values)
        else:
            feats = model.get_image_features(pixel_values=pixel_values)

        feats = unwrap_clip_features(feats)
        feats = F.normalize(feats, dim=-1)
        features.append(feats.cpu())

    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    captions: List[str],
    batch_size: int,
    device: str,
    use_amp: bool = True,
):
    features = []

    for start in tqdm(range(0, len(captions), batch_size), desc="encode texts"):
        batch_caps = captions[start:start + batch_size]

        tokens = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            feats = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        feats = unwrap_clip_features(feats)
        feats = F.normalize(feats, dim=-1)
        features.append(feats.cpu())

    return torch.cat(features, dim=0)
```

```python
def recall_at_k_image_to_text(
    sim: torch.Tensor,
    image_to_caption_indices: Dict[int, List[int]],
    ks=(1, 5, 10),
):
    """
    sim: [num_images, num_captions]
    """
    metrics = {}
    topk_max = max(ks)
    topk = sim.topk(topk_max, dim=1).indices

    for k in ks:
        hits = 0
        for image_idx in range(sim.shape[0]):
            positives = set(image_to_caption_indices[image_idx])
            predictions = set(topk[image_idx, :k].tolist())
            if positives.intersection(predictions):
                hits += 1

        metrics[f"i2t_R@{k}"] = hits / sim.shape[0]

    return metrics


def recall_at_k_text_to_image(
    sim: torch.Tensor,
    caption_to_image_index: List[int],
    ks=(1, 5, 10),
):
    """
    sim: [num_images, num_captions]
    text-to-image uses sim.T: [num_captions, num_images]
    """
    metrics = {}
    sim_t = sim.T

    positives = torch.tensor(
        caption_to_image_index,
        dtype=torch.long,
        device=sim.device,
    )

    topk_max = max(ks)
    topk = sim_t.topk(topk_max, dim=1).indices

    for k in ks:
        hits = (topk[:, :k] == positives[:, None]).any(dim=1).float().mean().item()
        metrics[f"t2i_R@{k}"] = hits

    return metrics


@torch.no_grad()
def eval_coco_retrieval(
    model,
    tokenizer,
    image_processor,
    coco_root: str,
    split: str,
    max_images: int,
    start_image_index: int,
    image_batch_size: int,
    text_batch_size: int,
    device: str,
):
    retrieval_set = build_coco_retrieval_set(
        coco_root=coco_root,
        split=split,
        max_images=max_images,
        start_image_index=start_image_index,
    )

    image_features = encode_images(
        model=model,
        image_processor=image_processor,
        image_paths=retrieval_set.image_paths,
        batch_size=image_batch_size,
        device=device,
    )

    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        captions=retrieval_set.captions,
        batch_size=text_batch_size,
        device=device,
    )

    image_features = image_features.to(device, non_blocking=True)
    text_features = text_features.to(device, non_blocking=True)

    torch.cuda.synchronize()
    print(
        "Eval similarity matmul on:",
        image_features.device,
        text_features.device,
    )

    sim = image_features @ text_features.T
    torch.cuda.synchronize()

    metrics = {}
    metrics.update(
        recall_at_k_image_to_text(
            sim,
            retrieval_set.image_to_caption_indices,
            ks=(1, 5, 10),
        )
    )
    metrics.update(
        recall_at_k_text_to_image(
            sim,
            retrieval_set.caption_to_image_index,
            ks=(1, 5, 10),
        )
    )

    return metrics
```

## 7. Load the FP16 model

ModelOpt Hugging Face checkpointing is enabled before loading or saving.
This environment also has an optional DeepSpeed package that cannot import
without a full CUDA toolkit, so the helper below disables that unused
integration when necessary.

```python
mto.enable_huggingface_checkpointing()
disable_unusable_deepspeed_plugin()
has_quant_attention = maybe_register_clip_attention_quantizer()

model = CLIPModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
).eval().to(DEVICE)

tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
image_processor = CLIPImageProcessor.from_pretrained(MODEL_ID)

print("Model device:", next(model.parameters()).device)
```

## 8. Baseline FP16 evaluation

The full 5,000-image COCO validation evaluation can take time. For a quick
notebook smoke test, temporarily use a smaller `max_images` value such as
`50`.

```python
# For a quick smoke test, try: max_images=50
baseline_metrics = eval_coco_retrieval(
    model=model,
    tokenizer=tokenizer,
    image_processor=image_processor,
    coco_root=str(COCO_ROOT),
    split=EVAL_SPLIT,
    max_images=EVAL_IMAGES,
    start_image_index=EVAL_START_IMAGE_INDEX,
    image_batch_size=IMAGE_BATCH_SIZE,
    text_batch_size=TEXT_BATCH_SIZE,
    device=DEVICE,
)
pd.Series(baseline_metrics, name="fp16").to_frame()
```

## 9. Calibration loader

The calibration callback exercises both CLIP towers. The explicit assertions
make it clear that calibration inputs and model parameters are on CUDA.

```python
calib_dataset = CocoCaptionPairs(
    coco_root=str(COCO_ROOT),
    split=CALIB_SPLIT,
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_items=CALIB_N,
    start_index=CALIB_START_INDEX,
)

calib_loader = DataLoader(
    calib_dataset,
    batch_size=CALIB_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_calib,
)

@torch.no_grad()
def calibrate(m):
    m.eval()
    for batch in tqdm(calib_loader, desc="W4A8 calibration"):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

        assert pixel_values.device.type == "cuda"
        assert next(m.parameters()).device.type == "cuda"

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = m.get_image_features(pixel_values=pixel_values)
            _ = m.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
```

## 10. Run W4A8 PTQ

The full quantizer summary is useful for debugging but very verbose. This
notebook prints a compact completion message by default and leaves the summary
call commented out.

```python
quant_cfg = build_w4a8_clip_cfg(
    has_quant_attention=has_quant_attention,
    disable_patch_embedding=DISABLE_PATCH_EMBEDDING,
)

started = time.time()
q_model = mtq.quantize(model, quant_cfg, forward_loop=calibrate)
quant_seconds = time.time() - started

print(f"W4A8 quantization completed in {quant_seconds:.1f} seconds.")
# mtq.print_quant_summary(q_model)
```

## 11. Quantized evaluation

The quantized model is evaluated on the same retrieval set so that the delta
is directly comparable with the FP16 baseline.

```python
quant_metrics = eval_coco_retrieval(
    model=q_model,
    tokenizer=tokenizer,
    image_processor=image_processor,
    coco_root=str(COCO_ROOT),
    split=EVAL_SPLIT,
    max_images=EVAL_IMAGES,
    start_image_index=EVAL_START_IMAGE_INDEX,
    image_batch_size=IMAGE_BATCH_SIZE,
    text_batch_size=TEXT_BATCH_SIZE,
    device=DEVICE,
)

delta = {
    metric: quant_metrics[metric] - baseline_metrics[metric]
    for metric in baseline_metrics
}
pd.DataFrame(
    {
        "metric": list(baseline_metrics),
        "fp16": [baseline_metrics[key] for key in baseline_metrics],
        "w4a8": [quant_metrics[key] for key in baseline_metrics],
        "delta": [delta[key] for key in baseline_metrics],
    }
)
```

## 12. Save checkpoint and metrics

`save_pretrained()` writes the Transformers checkpoint. With ModelOpt
checkpointing enabled, it also writes `modelopt_state.pth`, which is required
to restore the quantized architecture correctly.

```python
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

metrics_payload = {
    "model_id": MODEL_ID,
    "quant": "w4a8",
    "calib_split": CALIB_SPLIT,
    "eval_split": EVAL_SPLIT,
    "calib_n": CALIB_N,
    "calib_start_index": CALIB_START_INDEX,
    "eval_images": EVAL_IMAGES,
    "eval_start_image_index": EVAL_START_IMAGE_INDEX,
    "disable_patch_embedding": DISABLE_PATCH_EMBEDDING,
    "has_quant_attention": has_quant_attention,
    "quant_seconds": quant_seconds,
    "baseline_fp16": baseline_metrics,
    "quantized": quant_metrics,
    "delta": delta,
}

(RUN_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
q_model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
image_processor.save_pretrained(MODEL_DIR)

print("Saved metrics:", RUN_DIR / "metrics.json")
print("Saved checkpoint:", MODEL_DIR)
print("ModelOpt sidecar:", MODEL_DIR / "modelopt_state.pth")
```

## 13. Load existing run results from JSON

This cell reads the saved run rather than hardcoding scores. It can be run
independently of the expensive evaluation cells above.

```python
try:
    RESULTS_PATH = RUN_DIR / "metrics.json"
except NameError:
    RESULTS_PATH = Path("./runs/train2017calib_w4a8_1024calib_val2017_5000eval_no_patch/metrics.json")


def metrics_table_from_json(path: Path):
    if not path.exists():
        print(f"Missing {path}. Run the baseline, PTQ, quantized eval, and save cells first.")
        return None

    payload = json.loads(path.read_text())
    fp16 = payload.get("baseline_fp16") or payload.get("fp16") or payload.get("baseline")
    w4a8 = payload.get("quantized") or payload.get("w4a8")
    delta = payload.get("delta")

    if not fp16 or not w4a8:
        raise KeyError("Expected FP16 and quantized metrics in the saved JSON.")
    if delta is None:
        delta = {metric: w4a8[metric] - fp16[metric] for metric in fp16}

    return pd.DataFrame(
        {
            "metric": list(fp16),
            "fp16": [fp16[metric] for metric in fp16],
            "w4a8": [w4a8[metric] for metric in fp16],
            "delta": [delta[metric] for metric in fp16],
        }
    )


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| metric | fp16 | w4a8 | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.fp16:.4f} | {row.w4a8:.4f} | {row.delta:.4f} |"
        )
    return "\n".join(lines)


results_df = metrics_table_from_json(RESULTS_PATH)
if results_df is not None:
    print(dataframe_to_markdown(results_df))
results_df
```

## 14. Final Results

The table above is loaded directly from the saved metrics JSON and is the source
of truth for any numeric claim in the article. The uploaded artifacts are linked
from the [Hugging Face repository](https://huggingface.co/herooooooooo/clip-vit-l-14-laion2b-coco-w4a8-modelopt).

Interpret the deltas as retrieval-quality deltas only. Runtime export,
throughput, p95/p99 latency, and serving memory need separate measurement before
making speedup or production-deployment claims.
