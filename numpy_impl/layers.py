import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict


class Layer(ABC):
    """Abstract base class for all layers."""
    
    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return self.params
    
    def get_grads(self) -> Dict[str, np.ndarray]:
        return self.grads


class Linear(Layer):
    """
    Fully connected layer.
    
    y = x @ W + b
    
    Uses He initialization for weights.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # He initialization: var = 2/fan_in
        scale = np.sqrt(2.0 / in_features)
        self.params["W"] = np.random.randn(in_features, out_features) * scale
        #N(0, 1) *  np.sqrt(2.0 / in_features) ~ N(0, np.sqrt(2.0 / in_features)) - He распределение
        
        if bias:
            self.params["b"] = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, in_features)
        
        Returns:
            Output of shape (batch_size, out_features)
        """
        self.cache["x"] = x
        
        out = x @ self.params["W"]
        if self.use_bias:
            out = out + self.params["b"]
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            dout: Gradient from next layer, shape (batch_size, out_features)
        
        Returns:
            Gradient for previous layer, shape (batch_size, in_features)
        """
        x = self.cache["x"]
        
        # Gradient w.r.t. weights: dL/dW = x.T @ dout
        self.grads["W"] = x.T @ dout
        
        # Gradient w.r.t. bias: dL/db = sum(dout, axis=0)
        if self.use_bias:
            self.grads["b"] = dout.sum(axis=0)
        
        # Gradient w.r.t. input: dL/dx = dout @ W.T
        dx = dout @ self.params["W"].T
        
        return dx
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features}, bias={self.use_bias})"


class Dropout(Layer):
    """
    Dropout layer for regularization.
    
    During training, randomly zeros elements with probability p.
    During inference, returns input unchanged.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        
        # Create mask and scale
        self.cache["mask"] = (np.random.rand(*x.shape) > self.p) / (1 - self.p) #E[out] = E[x] чтобы распределение активаций на тесте и в трейне совпадало
        return x * self.cache["mask"]
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return dout
        return dout * self.cache["mask"]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def __repr__(self):
        return f"Dropout(p={self.p})"
