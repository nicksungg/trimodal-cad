import argparse
import glob
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors


def parse_args():
    parser = argparse.ArgumentParser(description="Extract point-encoder weights from CAD-Recode checkpoint")
    parser.add_argument("--recode-model", type=str, default="filapro/cad-recode-v1.5")
    parser.add_argument("--output", type=str, default="./outputs/recode_point_encoder.pt")
    return parser.parse_args()


def extract_point_state_dict(full_state):
    prefixes = ["point_encoder.", "model.point_encoder."]
    extracted = {}
    for key, value in full_state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                break

    if not extracted:
        raise RuntimeError("Could not find point encoder weights in model checkpoint")
    return extracted


def main():
    args = parse_args()

    local_repo_dir = snapshot_download(
        repo_id=args.recode_model,
        local_dir_use_symlinks=False,
    )

    all_tensors = {}
    safetensor_files = sorted(glob.glob(f"{local_repo_dir}/*.safetensors"))
    bin_files = sorted(glob.glob(f"{local_repo_dir}/pytorch_model*.bin"))

    for file_path in safetensor_files:
        shard = load_safetensors(file_path)
        all_tensors.update(shard)

    for file_path in bin_files:
        shard = torch.load(file_path, map_location="cpu")
        all_tensors.update(shard)

    point_state = extract_point_state_dict(all_tensors)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(point_state, output_path)

    print(f"Saved point encoder state_dict to {output_path}")


if __name__ == "__main__":
    main()
