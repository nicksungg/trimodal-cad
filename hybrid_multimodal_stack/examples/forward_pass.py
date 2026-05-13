#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hybrid_multimodal_stack.config import HybridConfig
from hybrid_multimodal_stack.model import HybridCADStack


def fps_numpy(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= n_points:
        return np.arange(points.shape[0], dtype=np.int64)

    selected = np.zeros(n_points, dtype=np.int64)
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    farthest = int(rng.integers(0, points.shape[0]))
    for i in range(n_points):
        selected[i] = farthest
        current_point = points[farthest]
        dist = np.sum((points - current_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-sample multimodal forward/generate test")
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path(
            "/orcd/data/faez/001/nick/CADEvolve-C_image_fresh_20260408/"
            "CADEvolve-C/ABC-C/00883866_896437595f9c11edf678b353_trimesh_000+r00+s90+t170+p95+k00_emitted_centered_scaled_binarized/view_0.png"
        ),
    )
    parser.add_argument(
        "--pointcloud-path",
        type=Path,
        default=Path(
            "/orcd/data/faez/001/nick/CADEvolve-C_pointcloud/ABC-C/"
            "00883866_896437595f9c11edf678b353_trimesh_000+r00+s90+t170+p95+k00_emitted_centered_scaled_binarized.npy"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write the CADQuery code for this object.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pointcloud-num-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-text", type=Path, default=None)
    return parser.parse_args()


def load_pointcloud(pointcloud_path: Path, n_points: int, seed: int) -> torch.Tensor:
    points = np.load(pointcloud_path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape [N,3], got {points.shape} from {pointcloud_path}")

    rng = np.random.default_rng(seed)
    indices = fps_numpy(points, n_points=n_points, rng=rng)
    points = points[indices]
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    args = parse_args()

    print("Starting forward pass...")
    print(f"Image path: {args.image_path}")
    print(f"Pointcloud path: {args.pointcloud_path}")

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.pointcloud_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {args.pointcloud_path}")

    print("Loading config...")
    config = HybridConfig(
        llm_name_or_path="Qwen/Qwen2-1.5B",
        vision_name_or_path="facebook/dinov2-base",
        point_encoder_weights="/orcd/data/faez/001/nick/hybrid_multimodal_stack/outputs/recode_point_encoder.pt",
        point_encoder_type="recode_fourier",
    )

    print("Loading model...")
    model = HybridCADStack(config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Processing image...")
    image = Image.open(args.image_path).convert("RGB")
    processor = model.vision_encoder.processor
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)

    print("Processing pointcloud...")
    pointcloud_embeddings = load_pointcloud(args.pointcloud_path, args.pointcloud_num_points, args.seed).to(device)

    print("Generating text...")
    with torch.inference_mode():
        encoded = model.tokenizer(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        text_embeddings = model.llm.get_input_embeddings()(input_ids)
        prefix = model.encode_modalities(
            pixel_values=pixel_values,
            pointcloud_embeddings=pointcloud_embeddings,
        )
        if prefix is not None:
            prefix = prefix.to(device)
            inputs_embeds = torch.cat([prefix, text_embeddings], dim=1)
            prefix_mask = torch.ones((input_ids.size(0), prefix.size(1)), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            eos_token_id = model.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = model.tokenizer.pad_token_id
            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=min(64, args.max_new_tokens),
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            generated = model.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        else:
            generated_ids = model.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=min(64, args.max_new_tokens),
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            generated = model.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    print("IMAGE_PATH:", args.image_path)
    print("POINTCLOUD_PATH:", args.pointcloud_path)
    print("PROMPT:", args.prompt)
    print("GENERATED_TEXT_START")
    print(generated)
    print("GENERATED_TEXT_END")

    if args.output_text is not None:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(generated, encoding="utf-8")
        print(f"saved generated text to: {args.output_text}")


if __name__ == "__main__":
    main()
