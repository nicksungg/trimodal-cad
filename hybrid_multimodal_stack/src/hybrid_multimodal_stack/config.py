from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainStage:
    name: str
    freeze_vision_encoder: bool
    freeze_point_encoder: bool
    freeze_llm: bool
    train_image_projector: bool
    train_point_projector: bool


@dataclass
class HybridConfig:
    llm_name_or_path: str = "Qwen/Qwen2-1.5B"
    vision_name_or_path: str = str(PROJECT_ROOT / "outputs" / "dinov2-base")
    point_encoder_type: str = "recode_fourier"
    point_encoder_weights: Optional[str] = str(PROJECT_ROOT / "outputs" / "recode_point_encoder.pt")
    point_encoder_feature_dim: int = 1536
    mm_hidden_size: int = 1536
    llm_hidden_size: Optional[int] = None
    image_projector_type: str = "mlp2x_gelu"
    point_projector_type: str = "mlp2x_gelu"
    device: str = "cuda"
