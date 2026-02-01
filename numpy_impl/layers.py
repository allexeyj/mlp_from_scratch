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


class BatchNorm1d(Layer):
    """
    Batch Normalization for 1D inputs (fully connected layers).
    
    Forward:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    During training:
        - Uses batch statistics (mean, var computed from current batch)
        - Updates running_mean and running_var with exponential moving average
    
    During eval:
        - Uses running statistics accumulated during training
    
    Parameters:
        gamma: Scale parameter (learnable), shape (num_features,)
        beta: Shift parameter (learnable), shape (num_features,)
    
    Buffers (not learnable):
        running_mean: EMA of batch means
        running_var: EMA of batch variances
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # Learnable parameters
        self.params["gamma"] = np.ones(num_features)
        self.params["beta"] = np.zeros(num_features)
        
        # Running statistics (not learnable, for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, num_features)
        
        Returns:
            Normalized output of shape (batch_size, num_features)
        """
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            
            # Update running statistics with EMA
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean = batch_mean
            var = batch_var
        else:
            # Use accumulated statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        
        # Scale and shift
        out = self.params["gamma"] * x_norm + self.params["beta"]
        
        # Cache for backward
        self.cache["x"] = x
        self.cache["x_norm"] = x_norm
        self.cache["mean"] = mean
        self.cache["std"] = std
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        The gradient through batch normalization is non-trivial because
        mean and variance depend on the entire batch.
        
        Args:
            dout: Gradient from next layer, shape (batch_size, num_features)
        
        Returns:
            Gradient for previous layer, shape (batch_size, num_features)
        """
        x = self.cache["x"]
        x_norm = self.cache["x_norm"]
        std = self.cache["std"]
        gamma = self.params["gamma"]
        
        batch_size = x.shape[0]
        
        # Gradients for learnable parameters
        self.grads["gamma"] = (dout * x_norm).sum(axis=0)
        self.grads["beta"] = dout.sum(axis=0)
        
        # Gradient for input (using the simplified formula)
        # dx = (1 / (N * std)) * gamma * (
        #     N * dout 
        #     - sum(dout) 
        #     - x_norm * sum(dout * x_norm)
        # )
        dx_norm = dout * gamma
        
        dx = (1.0 / (batch_size * std)) * (
            batch_size * dx_norm 
            - dx_norm.sum(axis=0) 
            - x_norm * (dx_norm * x_norm).sum(axis=0)
        )
        
        return dx
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum})"


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
        self.cache["mask"] = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
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
