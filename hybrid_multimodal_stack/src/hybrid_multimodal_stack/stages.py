from .config import TrainStage


PROJECTOR_WARMUP = TrainStage(
    name="projector_warmup",
    freeze_vision_encoder=True,
    freeze_point_encoder=True,
    freeze_llm=True,
    train_image_projector=True,
    train_point_projector=True,
)

JOINT_FINETUNE = TrainStage(
    name="joint_finetune",
    freeze_vision_encoder=True,
    freeze_point_encoder=True,
    freeze_llm=False,
    train_image_projector=True,
    train_point_projector=True,
)

LORA_STYLE = TrainStage(
    name="lora_style",
    freeze_vision_encoder=True,
    freeze_point_encoder=True,
    freeze_llm=True,
    train_image_projector=True,
    train_point_projector=True,
)
