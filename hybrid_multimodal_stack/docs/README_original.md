# Hybrid Multimodal Stack

This setup now matches your target architecture:

- Vision encoder: frozen DINOv2 (`facebook/dinov2-base` or local extracted copy)
- Vision adapter: trainable MLP projector
- Point encoder: frozen Recode Fourier point encoder
- Point adapter: trainable MLP projector
- Text side: `Qwen/Qwen2-1.5B` tokenizer + decoder

## What is trainable in each stage

- `projector_warmup`
  - Frozen: vision encoder, point encoder, LLM decoder
  - Trainable: image projector, point projector

- `joint_finetune`
  - Frozen: vision encoder, point encoder
  - Trainable: image projector, point projector, LLM decoder

## Included artifacts

- Recode point-encoder weights: `outputs/recode_point_encoder.pt`
- DINOv2 local weights: `outputs/dinov2-base/`

## Install

```bash
cd /home/nicksung/Desktop/nicksung/multimodel_class/hybrid_multimodal_stack
pip install -e .
```

## Training data format (from cluster)

Create a JSONL manifest where each line has:

```json
{"image_path": "/abs/path/to/image.png", "pointcloud_path": "/abs/path/to/points.npy", "cad_code": "import cadquery as cq\n..."}
```

- `image_path`: RGB image file
- `pointcloud_path`: `.npy` with shape `[N, 3]`
- `cad_code`: target CAD Python code string

The model reads files directly by absolute path, so cluster storage paths work.

## Bootstrap initialized model weights

```bash
python scripts/bootstrap_model.py \
  --llm Qwen/Qwen2-1.5B \
  --vision hybrid_multimodal_stack/outputs/dinov2-base \
  --point-encoder-weights hybrid_multimodal_stack/outputs/recode_point_encoder.pt \
  --point-feature-dim 1536 \
  --stage projector_warmup \
  --save-dir ./outputs/init
```

## Train end-to-end (with frozen encoders + trainable adapters)

```bash
python scripts/train_multimodal.py \
  --train-manifest /abs/path/to/train_manifest.jsonl \
  --output-dir ./outputs/train \
  --llm Qwen/Qwen2-1.5B \
  --vision hybrid_multimodal_stack/outputs/dinov2-base \
  --point-encoder-weights hybrid_multimodal_stack/outputs/recode_point_encoder.pt \
  --stage projector_warmup \
  --epochs 1 \
  --batch-size 2
```

Use `--stage joint_finetune` when you want to unfreeze and train the Qwen decoder as well.

## Train directly from dataset roots (auto-link paths)

You can skip manual manifest creation and pass your dataset roots directly. This builds a JSONL manifest automatically and starts `projector_warmup`, which trains both adapters (image + point).

```bash
python scripts/train_multimodal.py \
  --image-root /orcd/data/faez/001/nick/CADEvolve-C_image_fresh_20260408 \
  --pointcloud-root /orcd/data/faez/001/nick/CADEvolve-C_pointcloud \
  --cad-root /orcd/data/faez/001/nick/CADEvolve-C \
  --output-dir ./outputs/train_two_adapters \
  --manifest-out ./outputs/train_two_adapters/train_manifest.jsonl \
  --rebuild-manifest \
  --stage projector_warmup \
  --epochs 1 \
  --batch-size 2
```

Quick smoke test (recommended first):

```bash
python scripts/train_multimodal.py \
  --image-root /orcd/data/faez/001/nick/CADEvolve-C_image_fresh_20260408 \
  --pointcloud-root /orcd/data/faez/001/nick/CADEvolve-C_pointcloud \
  --cad-root /orcd/data/faez/001/nick/CADEvolve-C \
  --output-dir ./outputs/train_two_adapters_smoke \
  --manifest-out ./outputs/train_two_adapters_smoke/train_manifest.jsonl \
  --rebuild-manifest \
  --max-samples 1000 \
  --stage projector_warmup \
  --epochs 1 \
  --batch-size 2
```

## Main code

- `src/hybrid_multimodal_stack/model.py`: combined forward/generate model
- `src/hybrid_multimodal_stack/encoders.py`: DINOv2 encoder + Recode Fourier point encoder
- `src/hybrid_multimodal_stack/projectors.py`: LLaVA-style projector blocks
- `src/hybrid_multimodal_stack/data.py`: JSONL-based multimodal dataset loader
- `src/hybrid_multimodal_stack/stages.py`: freeze/train stage definitions
- `scripts/bootstrap_model.py`: one-command initializer
- `scripts/train_multimodal.py`: training loop for image + point cloud + CAD code
