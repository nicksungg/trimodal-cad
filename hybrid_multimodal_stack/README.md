# Hybrid Multimodal Stack — Release Bundle

Self-contained snapshot of the hybrid multimodal CAD model: source code, training
recipe, and the small frozen point-encoder weights. The large weight files
(DINOv2 backbone, ~331 MB; final checkpoint, ~6.2 GB) are excluded from git;
see *Trained weights* in the top-level README for how to materialize them.

For project context, paper, and results see the [top-level README](../README.md).
Copied from `/orcd/data/faez/001/nick/hybrid_multimodal_stack`.

## Architecture

| Component       | Role                        | Model                                                  | Trainable? |
|-----------------|-----------------------------|--------------------------------------------------------|------------|
| Vision encoder  | image → tokens (1×257×768)  | `facebook/dinov2-base` (local copy in `outputs/dinov2-base/`) | frozen |
| Image adapter   | DINOv2 → LLM hidden (1536)  | 2-layer MLP w/ GELU (`mlp2x_gelu`)                     | trainable  |
| Point encoder   | XYZ → tokens (B×N×1536)     | Recode Fourier point encoder (51 → 1536 linear, 8 freqs) | frozen   |
| Point adapter   | Recode → LLM hidden (1536)  | 2-layer MLP w/ GELU                                    | trainable  |
| Decoder LLM     | tokens → CAD code           | `Qwen/Qwen2-1.5B` (hidden_size=1536)                   | frozen (warmup) / trainable (joint) |

Multimodal prefix = `[image_tokens ; point_tokens]` is concatenated before the text
embeddings; labels at prefix positions are masked to `-100`. See
[`src/hybrid_multimodal_stack/model.py`](src/hybrid_multimodal_stack/model.py).

## What is in this folder

```
hybrid_multimodal_stack_release/
├── README.md                              ← this file
├── pyproject.toml, requirements.txt       ← deps + package metadata
├── src/hybrid_multimodal_stack/           ← installable Python package
│   ├── config.py        (HybridConfig dataclass + paths)
│   ├── model.py         (HybridCADStack: forward + generate)
│   ├── encoders.py      (DinoV2Encoder, RecodeFourierPointEncoder)
│   ├── projectors.py    (MLPProjector)
│   ├── data.py          (CADMultimodalDataset + manifest builder)
│   ├── stages.py        (PROJECTOR_WARMUP / JOINT_FINETUNE freeze maps)
│   └── utils.py
├── scripts/
│   ├── bootstrap_model.py                 ← one-shot weight initializer
│   ├── extract_dinov2_base.py             ← downloads facebook/dinov2-base
│   ├── extract_recode_point_encoder.py    ← extracts Recode point encoder weights
│   ├── train_multimodal.py                ← actual training loop
│   ├── train_two_adapters_orcd.sh         ← bash wrapper used in every SLURM run
│   ├── train_two_adapters_pi_faez_100h.sbatch        ← initial 100h training job
│   ├── train_two_adapters_resume_300k.sbatch         ← resume-from-300k job
│   ├── train_two_adapters_continue_latest.sbatch     ← auto-resume from newest ckpt
│   ├── eval_forward_and_iou.py            ← greedy forward + best-IoU eval
│   ├── eval_best_of_8_iou.py              ← best-of-8 sampling eval
│   └── run_eval_forward_and_iou.sbatch
├── configs/extracted_from_recode_notebook.yaml
├── examples/
│   ├── forward_pass.py                    ← single image+pointcloud inference demo
│   └── run_forward_pass.sbatch
├── docs/README_original.md                ← original project README
└── outputs/
    ├── dinov2-base/                       ← frozen vision encoder weights
    ├── recode_point_encoder.pt            ← frozen point encoder weights
    └── final_model/
        ├── checkpoint_step_1230000.pt     ← final trained checkpoint (~6.2 GB)
        ├── final.pt -> checkpoint_step_1230000.pt
        └── train_run.tail.log             ← last 200 lines of training log
```

> **Note on the large weight files (`dinov2-base/model.safetensors` and
> `final_model/checkpoint_step_1230000.pt`)**: these are **hardlinks** to the
> originals under `/orcd/data/faez/001/nick/hybrid_multimodal_stack/outputs/...`
> on the same filesystem. They behave like normal files (cat, hash, load all
> work identically) and have full data persistence — deleting one side does not
> affect the other. This avoids duplicating ~4.7 GB on disk. If you need to move
> this folder to a different filesystem, `cp -L` or `rsync` will materialize
> them into real copies.

## Install

```bash
# from this folder
pip install -e .
```

Tested with the conda env at `/orcd/data/faez/001/nicksung/conda/envs/cadevolve`.
Key versions: `torch>=2.1`, `transformers>=4.45` (older 4.40 fails to recognize
the DINOv2 `BitImageProcessorFast` entry — see "Known gotchas" below).

For evaluation (IoU) you additionally need `cadquery` and `tqdm`.

## How it was trained

### Data

Three roots on the cluster, joined by case_id (folder name):

| Root                                                                     | What                                       |
|--------------------------------------------------------------------------|--------------------------------------------|
| `/orcd/data/faez/001/nick/CADEvolve-C_image_fresh_20260408/`             | `view_{0..7}.png` per case                 |
| `/orcd/data/faez/001/nick/CADEvolve-C_pointcloud/`                       | `<case>.npy` shape `[N,3]` per case        |
| `/orcd/data/faez/001/nick/CADEvolve-C/`                                  | `<case>.py` CADQuery target code per case  |

The training driver walks the image root and emits one JSONL row per
`view_[0-7].png` with the matched pointcloud + CAD code path:

```json
{"image_path": "...view_0.png", "pointcloud_path": "...case.npy", "cad_path": "...case.py"}
```

Manifest building is automatic in
[`scripts/train_multimodal.py`](scripts/train_multimodal.py) (`--rebuild-manifest`),
or you can pass an existing JSONL with `--train-manifest`.

Pointclouds are subsampled at load time with farthest-point-sampling to 256 points
(seeded by the pointcloud path so it is deterministic across epochs).

### Stage

All trained runs used **`projector_warmup`**, i.e.

- frozen: DINOv2 backbone, Recode point encoder, Qwen2-1.5B decoder
- trainable: image MLP projector + point MLP projector

`joint_finetune` (unfreezes the LLM) is available in
[`stages.py`](src/hybrid_multimodal_stack/stages.py) but was not used for the
released checkpoint.

### Optimization

- Optimizer: `AdamW`, lr=`2e-4`, no weight-decay override
- Grad clipping: `max_norm=1.0`
- Loss: standard causal LM cross-entropy, prefix positions masked to `-100`
- Mixed precision: model loaded in fp32; the adapters cast their output to the
  LLM input embedding dtype on the fly ([`model.py:108`](src/hybrid_multimodal_stack/model.py#L108))
- Hardware: 2× H100 via `torch.nn.DataParallel`
- Batch size: 4 per run (effective 8 over 2 GPUs)
- Save cadence: every 10k steps

### Training history → final checkpoint

| Run dir (under original `outputs/`)                      | Steps range          | Notes |
|----------------------------------------------------------|----------------------|-------|
| `train_two_adapters_node2435_12029506`                   | 10k → 300k           | first long run from scratch |
| `train_two_adapters_resume_300k_12204250`                | 310k → **1,234,060** | resumed via `--resume-checkpoint`, ran to ~1.23 M steps |

**Final checkpoint** included here:
`outputs/final_model/checkpoint_step_1230000.pt` (alias `final.pt`).

Reproducing the full training pipeline:

```bash
# 1) (one-time) download the frozen encoders — only needed if you delete
#    outputs/dinov2-base/ or outputs/recode_point_encoder.pt
PYTHONPATH=src python scripts/extract_dinov2_base.py \
    --model facebook/dinov2-base \
    --output-dir outputs/dinov2-base

PYTHONPATH=src python scripts/extract_recode_point_encoder.py \
    --recode-model filapro/cad-recode-v1.5 \
    --output outputs/recode_point_encoder.pt

# 2) initial 100h job on the pi_faez partition (2x H100)
sbatch scripts/train_two_adapters_pi_faez_100h.sbatch

# 3) once a checkpoint exists, resume from the most recent one
sbatch scripts/train_two_adapters_continue_latest.sbatch
```

The wrapper [`scripts/train_two_adapters_orcd.sh`](scripts/train_two_adapters_orcd.sh)
reads env vars (`OUTPUT_DIR`, `RESUME_CHECKPOINT`, `BATCH_SIZE`, `SAVE_EVERY`,
`NUM_WORKERS`, `USE_DATA_PARALLEL`, `TRAIN_MANIFEST`, etc.) and invokes
`train_multimodal.py` with the right CLI.

## How to use the trained model

### Quick single-sample inference

```bash
PYTHONPATH=src python examples/forward_pass.py \
    --image-path /abs/path/to/view_0.png \
    --pointcloud-path /abs/path/to/case.npy \
    --prompt "Write the CADQuery code for this object." \
    --max-new-tokens 256
```

That example currently runs from a **freshly initialized** stack (no trained
weights). To use the released checkpoint, load it after building the model — see
the minimal loader below or use `scripts/eval_forward_and_iou.py`, which already
does this.

### Minimal Python loader

```python
import torch
from hybrid_multimodal_stack.config import HybridConfig
from hybrid_multimodal_stack.model import HybridCADStack

REL = "/orcd/data/faez/001/nick/hybrid_multimodal_stack_release"

config = HybridConfig(
    llm_name_or_path="Qwen/Qwen2-1.5B",
    vision_name_or_path=f"{REL}/outputs/dinov2-base",
    point_encoder_weights=f"{REL}/outputs/recode_point_encoder.pt",
    point_encoder_type="recode_fourier",
)

model = HybridCADStack(config).eval()

ckpt = torch.load(f"{REL}/outputs/final_model/final.pt", map_location="cpu")
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")

model.to("cuda")

# Build a pixel_values tensor and a pointcloud_embeddings tensor
# (see scripts/eval_forward_and_iou.py for the full preprocessing pipeline),
# then either model(input_ids=..., pixel_values=..., pointcloud_embeddings=...)
# for training-style forward, or model.generate(prompt, pixel_values=...,
# pointcloud_embeddings=...) for autoregressive decoding.
```

### Batch evaluation (best-of-N IoU on cadbench)

```bash
sbatch scripts/run_eval_forward_and_iou.sbatch
# or invoke directly:
PYTHONPATH=src python scripts/eval_forward_and_iou.py \
    --image-root /orcd/data/faez/001/nick/cadbench_stls/from_workstation/cadbench_stl_images \
    --pointcloud-root /orcd/data/faez/001/nick/cadbench_stls/cadbench_pointclouds_256 \
    --gt-step-root /orcd/data/faez/001/nick/cadbench_stls/cadbench_steps_256 \
    --out-dir ./outputs/eval_forward_iou
```

`eval_best_of_8_iou.py` does the same with 8 stochastic samples per case and
keeps the best IoU.

## Known gotchas

1. **`Unrecognized image processor in outputs/dinov2-base`** with older
   `transformers`. The `preprocessor_config.json` here uses
   `image_processor_type: BitImageProcessorFast`, which only exists in
   `transformers>=4.45`. Either upgrade, or replace that field with
   `BitImageProcessor` if you must pin to an older version. This is what
   killed the `train_two_adapters_continue_latest_13059748` run.

2. **`mm_hidden_size` must equal `llm.config.hidden_size`** (1536 for Qwen2-1.5B);
   enforced in [`model.py:38`](src/hybrid_multimodal_stack/model.py#L38).

3. **Point cloud shape** must be `[N, 3]` as `.npy`; the dataset enforces this.

4. **Resuming** the optimizer state requires the original checkpoint dict
   layout (`{global_step, epoch, model_state_dict, optimizer_state_dict}`).
   Plain `state_dict` checkpoints still work for weight-only loading.
