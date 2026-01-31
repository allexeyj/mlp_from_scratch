import numpy as np
from typing import List, Dict

from .layers import Linear, Dropout, Layer
from .activations import ReLU, GELU


class Sequential:
    """
    Container for sequential layers.
    
    Similar to torch.nn.Sequential.
    """
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Collect all parameters from all layers."""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for name, param in layer_params.items():
                params[f"layer{i}_{name}"] = param
        return params
    
    def get_grads(self) -> Dict[str, np.ndarray]:
        """Collect all gradients from all layers."""
        grads = {}
        for i, layer in enumerate(self.layers):
            layer_grads = layer.get_grads()
            for name, grad in layer_grads.items():
                grads[f"layer{i}_{name}"] = grad
        return grads
    
    def train(self):
        self._training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval(self):
        self._training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)


class MLP:
    """
    Multi-Layer Perceptron.
    
    Architecture: Linear -> ReLU -> [Linear -> ReLU ->] ... -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        activation_cls = ReLU if activation == "relu" else GELU
        
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(Dropout(dropout))
        
        # Output layer (no activation)
        layers.append(Linear(dims[-1], output_dim))
        
        self.model = Sequential(layers)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model.forward(x)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.model.backward(dout)
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return self.model.get_params()
    
    def get_grads(self) -> Dict[str, np.ndarray]:
        return self.model.get_grads()
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def __repr__(self):
        arch = f"{self.input_dim} -> {self.hidden_dims} -> {self.output_dim}"
        return f"MLP({arch})\n{self.model}"
