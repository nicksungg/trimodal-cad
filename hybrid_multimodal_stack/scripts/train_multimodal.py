import argparse
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hybrid_multimodal_stack.config import HybridConfig
from hybrid_multimodal_stack.data import CADMultimodalDataset, build_manifest_from_roots
from hybrid_multimodal_stack.model import HybridCADStack
from hybrid_multimodal_stack.stages import JOINT_FINETUNE, PROJECTOR_WARMUP


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal CAD model")
    parser.add_argument("--train-manifest", type=str, default=None, help="JSONL with image_path, pointcloud_path, and cad_code/cad_path")
    parser.add_argument("--image-root", type=str, default=None, help="Root folder containing image case dirs with view_0.png")
    parser.add_argument("--pointcloud-root", type=str, default=None, help="Root folder containing .npy pointcloud files")
    parser.add_argument("--cad-root", type=str, default=None, help="Root folder containing CAD .py files")
    parser.add_argument("--manifest-out", type=str, default=None, help="Path for auto-generated JSONL manifest")
    parser.add_argument("--image-filename", type=str, default="view_*.png")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./outputs/train")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--vision", type=str, default=str(ROOT_DIR / "outputs" / "dinov2-base"))
    parser.add_argument("--point-encoder-weights", type=str, default=str(ROOT_DIR / "outputs" / "recode_point_encoder.pt"))
    parser.add_argument("--stage", type=str, choices=["projector_warmup", "joint_finetune"], default="projector_warmup")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint path to initialize model weights, and optimizer state when available")
    parser.add_argument("--use-data-parallel", action="store_true", help="Use torch.nn.DataParallel when multiple GPUs are available")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm for clipping (0 to disable)")
    return parser.parse_args()


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run_log(run_log_path: Path, message: str) -> None:
    line = f"[{_ts()}] {message}"
    print(line, flush=True)
    _append_log(run_log_path, line)


def _load_checkpoint(checkpoint_path: Path, run_log_path: Path):
    _run_log(run_log_path, f"loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return None, checkpoint
    raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")


@contextmanager
def _stage(stage_dir: Path, run_log_path: Path, name: str):
    stage_log_path = stage_dir / f"{name}.log"
    _append_log(stage_log_path, f"START {_ts()}")
    _run_log(run_log_path, f"[stage:{name}] START")
    try:
        yield stage_log_path
    except Exception:
        tb = traceback.format_exc()
        _append_log(stage_log_path, f"FAILED {_ts()}")
        _append_log(stage_log_path, tb)
        _run_log(run_log_path, f"[stage:{name}] FAILED")
        raise
    else:
        _append_log(stage_log_path, f"SUCCESS {_ts()}")
        _run_log(run_log_path, f"[stage:{name}] SUCCESS")


def make_collate_fn(model: HybridCADStack, max_length: int):
    processor = model.vision_encoder.processor
    tokenizer = model.tokenizer

    def collate(batch):
        images = [x["image"] for x in batch]
        points = [x["points"] for x in batch]
        cad_code = [x["cad_code"] for x in batch]

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]

        tokenized = tokenizer(
            cad_code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100

        max_points = max(p.shape[0] for p in points)
        padded = torch.zeros((len(points), max_points, 3), dtype=torch.float32)
        for i, p in enumerate(points):
            padded[i, : p.shape[0], :] = p

        return {
            "pixel_values": pixel_values,
            "pointcloud_embeddings": padded,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return collate


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "train_run.log"
    stage_dir = output_dir / "stage_logs"
    stage_dir.mkdir(parents=True, exist_ok=True)

    _run_log(run_log_path, "train_multimodal.py start")
    _run_log(run_log_path, f"output_dir={output_dir}")
    _run_log(run_log_path, f"args.stage={args.stage} args.batch_size={args.batch_size} args.num_workers={args.num_workers}")
    _run_log(run_log_path, f"args.grad_clip={args.grad_clip}")

    manifest_path = args.train_manifest
    roots_provided = all([args.image_root, args.pointcloud_root, args.cad_root])

    try:
        with _stage(stage_dir, run_log_path, "01_manifest"):
            if manifest_path is None:
                if not roots_provided:
                    raise ValueError(
                        "Provide --train-manifest, or provide all of --image-root --pointcloud-root --cad-root."
                    )

                auto_manifest_path = Path(args.manifest_out) if args.manifest_out else output_dir / "train_manifest.jsonl"
                manifest_path = str(auto_manifest_path)

                if args.rebuild_manifest or not auto_manifest_path.exists():
                    _run_log(run_log_path, f"building manifest at: {auto_manifest_path}")
                    written = build_manifest_from_roots(
                        image_root=args.image_root,
                        pointcloud_root=args.pointcloud_root,
                        cad_root=args.cad_root,
                        output_manifest_path=str(auto_manifest_path),
                        image_filename=args.image_filename,
                        max_samples=args.max_samples,
                    )
                    _run_log(run_log_path, f"manifest rows written: {written}")
                else:
                    _run_log(run_log_path, f"using existing manifest: {auto_manifest_path}")
            else:
                _run_log(run_log_path, f"using provided manifest: {manifest_path}")

        with _stage(stage_dir, run_log_path, "02_model_init"):
            config = HybridConfig(
                llm_name_or_path=args.llm,
                vision_name_or_path=args.vision,
                point_encoder_weights=args.point_encoder_weights,
                point_encoder_type="recode_fourier",
            )

            model = HybridCADStack(config)
            _run_log(
                run_log_path,
                f"model initialized: llm={config.llm_name_or_path} vision={config.vision_name_or_path} point_encoder={config.point_encoder_type}",
            )

            resume_training_state = None
            if args.resume_checkpoint:
                resume_checkpoint_path = Path(args.resume_checkpoint)
                if not resume_checkpoint_path.exists():
                    raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
                resume_training_state, model_state_dict = _load_checkpoint(resume_checkpoint_path, run_log_path)
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                _run_log(
                    run_log_path,
                    "loaded resume weights "
                    f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}",
                )

        with _stage(stage_dir, run_log_path, "03_apply_stage"):
            if args.stage == "projector_warmup":
                model.apply_stage(PROJECTOR_WARMUP)
                _run_log(run_log_path, "applied stage: projector_warmup")
            else:
                model.apply_stage(JOINT_FINETUNE)
                _run_log(run_log_path, "applied stage: joint_finetune")

        with _stage(stage_dir, run_log_path, "04_device_setup"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            _run_log(run_log_path, f"device={device}")
            _run_log(run_log_path, f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")

            if args.use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                _run_log(run_log_path, f"using DataParallel across {torch.cuda.device_count()} GPUs")
                model = torch.nn.DataParallel(model)

        model.train()

        with _stage(stage_dir, run_log_path, "05_dataset_init"):
            dataset = CADMultimodalDataset(manifest_path)
            _run_log(run_log_path, f"dataset size: {len(dataset)}")

        with _stage(stage_dir, run_log_path, "06_dataloader_init"):
            model_for_collate = model.module if isinstance(model, torch.nn.DataParallel) else model
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=make_collate_fn(model_for_collate, args.max_length),
            )
            _run_log(run_log_path, "dataloader initialized")

        with _stage(stage_dir, run_log_path, "07_optimizer_init"):
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            _run_log(run_log_path, f"trainable_params={sum(p.numel() for p in trainable_params)}")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            _run_log(run_log_path, f"optimizer=AdamW lr={args.learning_rate}")

            if resume_training_state and "optimizer_state_dict" in resume_training_state:
                optimizer.load_state_dict(resume_training_state["optimizer_state_dict"])
                _run_log(run_log_path, "loaded optimizer state from resume checkpoint")

        with _stage(stage_dir, run_log_path, "08_training_loop"):
            global_step = resume_training_state.get("global_step", 0) if resume_training_state else 0
            first_batch_logged = False
            for epoch in range(args.epochs):
                _run_log(run_log_path, f"epoch {epoch} start")
                for batch in loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    if not first_batch_logged:
                        _run_log(
                            run_log_path,
                            "first batch shapes: "
                            f"input_ids={tuple(batch['input_ids'].shape)} "
                            f"attention_mask={tuple(batch['attention_mask'].shape)} "
                            f"labels={tuple(batch['labels'].shape)} "
                            f"pixel_values={tuple(batch['pixel_values'].shape)} "
                            f"pointcloud_embeddings={tuple(batch['pointcloud_embeddings'].shape)}",
                        )
                        first_batch_logged = True

                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        pixel_values=batch["pixel_values"],
                        pointcloud_embeddings=batch["pointcloud_embeddings"],
                    )

                    loss = outputs.loss
                    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                        loss = loss.mean()

                    if torch.isnan(loss) or torch.isinf(loss):
                        _run_log(run_log_path, f"WARNING: NaN/Inf loss at step {global_step + 1}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

                    optimizer.step()

                    global_step += 1
                    if global_step % 10 == 0:
                        _run_log(run_log_path, f"epoch={epoch} step={global_step} loss={loss.item():.4f}")

                    if global_step % args.save_every == 0:
                        save_path = output_dir / f"checkpoint_step_{global_step}.pt"
                        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                        torch.save({
                            "global_step": global_step,
                            "epoch": epoch,
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, save_path)
                        _run_log(run_log_path, f"saved checkpoint: {save_path}")

        with _stage(stage_dir, run_log_path, "09_finalize"):
            final_path = output_dir / "final_model.pt"
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), final_path)
            _run_log(run_log_path, f"saved final model: {final_path}")
            _run_log(run_log_path, "training script finished successfully")
    except Exception as exc:
        failure_log = output_dir / "train_failure.log"
        _append_log(failure_log, f"FAILED {_ts()}")
        _append_log(failure_log, str(exc))
        _append_log(failure_log, traceback.format_exc())
        _run_log(run_log_path, f"training script failed: {exc}")
        raise


if __name__ == "__main__":
    main()
