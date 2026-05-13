from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HybridConfig, TrainStage
from .encoders import DinoV2Encoder, IdentityPointCloudEncoder, build_recode_frozen_point_encoder
from .projectors import MLPProjector


class HybridCADStack(nn.Module):
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path, use_fast=False)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            torch_dtype=torch.float32,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        self.vision_encoder = DinoV2Encoder(config.vision_name_or_path)
        if config.point_encoder_type == "recode_fourier":
            llm_hidden_size_for_recode = int(self.llm.config.hidden_size)
            self.point_encoder = build_recode_frozen_point_encoder(
                hidden_size=llm_hidden_size_for_recode,
                checkpoint_path=config.point_encoder_weights,
            )
            point_encoder_feature_dim = llm_hidden_size_for_recode
        else:
            self.point_encoder = IdentityPointCloudEncoder(feature_dim=config.point_encoder_feature_dim)
            point_encoder_feature_dim = config.point_encoder_feature_dim

        llm_hidden_size = config.llm_hidden_size or int(self.llm.config.hidden_size)
        mm_hidden_size = config.mm_hidden_size or llm_hidden_size
        if mm_hidden_size != llm_hidden_size:
            raise ValueError(
                f"mm_hidden_size ({mm_hidden_size}) must match llm hidden size ({llm_hidden_size})"
            )
        self.image_projector = MLPProjector(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=mm_hidden_size,
            projector_type=config.image_projector_type,
        )
        self.point_projector = MLPProjector(
            in_dim=point_encoder_feature_dim,
            out_dim=mm_hidden_size,
            projector_type=config.point_projector_type,
        )

    def set_point_encoder(self, encoder_module: nn.Module, freeze: bool = True) -> None:
        self.point_encoder = encoder_module
        if freeze:
            self.point_encoder.requires_grad_(False)

    def apply_stage(self, stage: TrainStage) -> None:
        self.vision_encoder.requires_grad_(not stage.freeze_vision_encoder)
        self.point_encoder.requires_grad_(not stage.freeze_point_encoder)
        self.llm.requires_grad_(not stage.freeze_llm)
        self.image_projector.requires_grad_(stage.train_image_projector)
        self.point_projector.requires_grad_(stage.train_point_projector)

    def encode_modalities(
        self,
        pixel_values: Optional[torch.Tensor],
        pointcloud_embeddings: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        modality_embeddings = []

        if pixel_values is not None:
            image_tokens = self.vision_encoder(pixel_values)
            image_tokens = self.image_projector(image_tokens)
            modality_embeddings.append(image_tokens)

        if pointcloud_embeddings is not None:
            point_tokens = self.point_encoder(pointcloud_embeddings)
            point_tokens = self.point_projector(point_tokens)
            modality_embeddings.append(point_tokens)

        if not modality_embeddings:
            return None

        return torch.cat(modality_embeddings, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pointcloud_embeddings: Optional[torch.Tensor] = None,
    ):
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        prefix = self.encode_modalities(pixel_values=pixel_values, pointcloud_embeddings=pointcloud_embeddings)

        if prefix is None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Align dtype so projector (float32) matches LLM (may be bfloat16)
        prefix = prefix.to(dtype=text_embeddings.dtype)

        batch_size = text_embeddings.size(0)
        prefix_len = prefix.size(1)
        inputs_embeds = torch.cat([prefix, text_embeddings], dim=1)

        if attention_mask is None:
            text_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        else:
            text_mask = attention_mask

        prefix_mask = torch.ones((batch_size, prefix_len), dtype=text_mask.dtype, device=text_mask.device)
        merged_attention_mask = torch.cat([prefix_mask, text_mask], dim=1)

        merged_labels = None
        if labels is not None:
            prefix_labels = torch.full((batch_size, prefix_len), -100, dtype=labels.dtype, device=labels.device)
            merged_labels = torch.cat([prefix_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            labels=merged_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        pixel_values: Optional[torch.Tensor] = None,
        pointcloud_embeddings: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
    ) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        prefix = self.encode_modalities(pixel_values=pixel_values, pointcloud_embeddings=pointcloud_embeddings)

        if prefix is not None:
            prefix = prefix.to(device)
            inputs_embeds = torch.cat([prefix, text_embeddings], dim=1)
            prefix_mask = torch.ones((input_ids.size(0), prefix.size(1)), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            generated = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        else:
            generated = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
