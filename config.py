from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    embedding_model: str = "vit_small_patch16_224"
    num_classes: int = 10
    
    # Model
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.0
    
    # Training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    lr_min: float = 1e-6
    
    # Adam
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Wandb
    project: str = "mlp-from-scratch"
    
    # Misc
    seed: int = 42


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    return Config(**kwargs)
