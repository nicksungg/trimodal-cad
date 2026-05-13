import argparse
from pathlib import Path

from transformers import AutoImageProcessor, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Download/save DINOv2 base model locally")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--output-dir", type=str, default="./outputs/dinov2-base")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Saved DINOv2 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
