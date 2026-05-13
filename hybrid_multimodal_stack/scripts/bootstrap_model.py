import argparse
from pathlib import Path

import torch

from hybrid_multimodal_stack.config import HybridConfig
from hybrid_multimodal_stack.encoders import FrozenPointCloudEncoder
from hybrid_multimodal_stack.model import HybridCADStack
from hybrid_multimodal_stack.stages import JOINT_FINETUNE, PROJECTOR_WARMUP
from hybrid_multimodal_stack.utils import load_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--vision", type=str, default="hybrid_multimodal_stack/outputs/dinov2-base")
    parser.add_argument("--point-encoder-class", type=str, default="")
    parser.add_argument("--point-encoder-weights", type=str, default="")
    parser.add_argument("--point-feature-dim", type=int, default=1024)
    parser.add_argument("--stage", type=str, choices=["projector_warmup", "joint_finetune"], default="projector_warmup")
    parser.add_argument("--save-dir", type=str, default="./outputs/bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()

    config = HybridConfig(
        llm_name_or_path=args.llm,
        vision_name_or_path=args.vision,
        point_encoder_weights=args.point_encoder_weights or "hybrid_multimodal_stack/outputs/recode_point_encoder.pt",
        point_encoder_feature_dim=args.point_feature_dim,
    )

    model = HybridCADStack(config)

    if args.point_encoder_class:
        encoder_cls = load_class(args.point_encoder_class)
        frozen_encoder = FrozenPointCloudEncoder(
            encoder_builder=lambda: encoder_cls(),
            checkpoint_path=args.point_encoder_weights or None,
            strict=False,
        )
        model.set_point_encoder(frozen_encoder, freeze=True)

    if args.stage == "projector_warmup":
        model.apply_stage(PROJECTOR_WARMUP)
    else:
        model.apply_stage(JOINT_FINETUNE)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "hybrid_bootstrap.pt")
    print(f"Saved initialized model weights to {save_dir / 'hybrid_bootstrap.pt'}")


if __name__ == "__main__":
    main()
