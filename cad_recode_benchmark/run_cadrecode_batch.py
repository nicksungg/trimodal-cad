#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from multiprocessing import Process, Queue

import cadquery as cq
import numpy as np
import torch
import trimesh
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn


class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        point_cloud=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["point_cloud"] = kwargs["point_cloud"]
        return model_inputs


def load_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-1.5B",
        pad_token="<|im_end|>",
        padding_side="left",
    )
    model_kwargs = {
        "torch_dtype": "auto",
    }
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = CADRecode.from_pretrained(
            "filapro/cad-recode-v1.5",
            **model_kwargs,
        ).eval().to(device)
    except ImportError as exc:
        if model_kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        print("FlashAttention2 unavailable, falling back to default attention.")
        model_kwargs.pop("attn_implementation", None)
        model = CADRecode.from_pretrained(
            "filapro/cad-recode-v1.5",
            **model_kwargs,
        ).eval().to(device)
    return tokenizer, model


def generate_step_from_point_cloud(
    model,
    tokenizer,
    point_cloud: np.ndarray,
    device: str,
    max_new_tokens: int,
) -> str:
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer("<|im_start|>")["input_ids"][0]]
    attention_mask = [-1] * len(point_cloud) + [1]
    with torch.no_grad():
        batch_ids = model.generate(
            input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
            attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
            point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    py_string = tokenizer.batch_decode(batch_ids)[0]
    begin = py_string.find("<|im_start|>") + 12
    end = py_string.find("<|endoftext|>")
    return py_string[begin:end]


def _exec_and_export_worker(py_string: str, out_step_str: str, queue: Queue) -> None:
    namespace = {}
    try:
        exec(py_string, namespace)
        compound = namespace["r"].val()
        out_step = Path(out_step_str)
        out_step.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(compound, str(out_step))
        queue.put((True, "ok"))
    except Exception as exc:
        queue.put((False, str(exc)))


def exec_and_export(py_string: str, out_step: Path, timeout_sec: int) -> tuple[bool, str]:
    queue: Queue = Queue(maxsize=1)
    process = Process(
        target=_exec_and_export_worker,
        args=(py_string, str(out_step), queue),
    )
    process.start()
    process.join(timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"timeout_after_{timeout_sec}s"

    if not queue.empty():
        return queue.get()

    if process.exitcode == 0:
        return False, "no_result_from_worker"
    return False, f"worker_exit_code_{process.exitcode}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAD-Recode on point clouds and export STEP files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("cadbench_stls/cadbench_pointclouds_256"),
        help="Directory containing .npy point clouds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cadbench_stls/cadbench_steps_256"),
        help="Directory where .step files will be written.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of point clouds to process.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Generation length for CAD-Recode.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip point clouds that already have STEP output.")
    parser.add_argument("--exec-timeout-sec", type=int, default=20, help="Timeout per sample for executing generated CAD code.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer, model = load_model(device)

    pc_files = sorted(input_dir.rglob("*.npy"))
    if args.limit > 0:
        pc_files = pc_files[: args.limit]

    if not pc_files:
        print(f"No point-cloud files found under: {input_dir}")
        return

    ok = 0
    failed: list[str] = []
    print(f"Found {len(pc_files)} point clouds")
    print(f"Writing STEP files to: {output_dir}")

    for idx, pc_path in enumerate(pc_files):
        rel = pc_path.relative_to(input_dir).with_suffix(".step")
        out_step = output_dir / rel
        if args.skip_existing and out_step.exists():
            ok += 1
            continue

        try:
            point_cloud = np.load(pc_path)
            if point_cloud.shape != (256, 3):
                raise ValueError(f"Expected shape (256, 3), got {point_cloud.shape}")

            py_string = generate_step_from_point_cloud(
                model=model,
                tokenizer=tokenizer,
                point_cloud=point_cloud,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            py_path = out_step.with_suffix(".py")
            py_path.parent.mkdir(parents=True, exist_ok=True)
            py_path.write_text(py_string, encoding="utf-8")

            ok_export, export_reason = exec_and_export(
                py_string=py_string,
                out_step=out_step,
                timeout_sec=args.exec_timeout_sec,
            )
            if ok_export:
                ok += 1
                if ok <= 3:
                    print(f"OK: {pc_path} -> {out_step}")
            else:
                raise RuntimeError(f"Generated Python did not execute/export: {export_reason}")
        except Exception as exc:
            failed.append(f"{pc_path}: {exc}")

        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(pc_files)}")

    print(f"Completed: {ok}/{len(pc_files)} STEP files generated")
    if failed:
        log_path = output_dir / "cadrecode_errors.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"Errors: {len(failed)}")
        print(f"Error log: {log_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
