import torch
import torch.nn as nn
from typing import List


class TorchMLP(nn.Module):
    """
    Simple MLP in PyTorch for comparison.
    
    Architecture options:
        - Without BatchNorm: Linear -> ReLU -> [Dropout] -> ...
        - With BatchNorm: Linear -> BatchNorm1d -> ReLU -> [Dropout] -> ...
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
        use_batchnorm: bool = False
    ):
        super().__init__()
        
        self.use_batchnorm = use_batchnorm
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # BatchNorm (before activation)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (after activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer (no activation, no batchnorm)
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
