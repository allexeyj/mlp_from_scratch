import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    ReLU activation: max(0, x)
    
    Gradient: 1 if x > 0, else 0
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        return np.maximum(0, x)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["x"]
        return dout * (x > 0)
    
    def __repr__(self):
        return "ReLU()"


class GELU(Layer):
    """
    Gaussian Error Linear Unit (approximation).
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        
        inner = self.sqrt_2_over_pi * (x + 0.044715 * x**3)
        self.cache["tanh"] = np.tanh(inner)
        
        return 0.5 * x * (1 + self.cache["tanh"])
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["x"]
        tanh_val = self.cache["tanh"]
        
        # Derivative of GELU
        sech2 = 1 - tanh_val**2
        inner_deriv = self.sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
        
        gelu_grad = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * inner_deriv
        
        return dout * gelu_grad
    
    def __repr__(self):
        return "GELU()"


class Softmax(Layer):
    """
    Softmax activation (numerically stable).
    
    softmax(x)_i = exp(x_i) / sum(exp(x_j))
    """
    
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max
        x_stable = x - x.max(axis=self.axis, keepdims=True)
        exp_x = np.exp(x_stable)
        probs = exp_x / exp_x.sum(axis=self.axis, keepdims=True)
        
        self.cache["probs"] = probs
        return probs
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Full Jacobian backward (for general use).
        Note: Usually combined with CrossEntropy for efficiency.
        """
        probs = self.cache["probs"]
        
        # For each sample: dL/dx = probs * (dout - sum(dout * probs))
        sum_dp = (dout * probs).sum(axis=self.axis, keepdims=True)
        return probs * (dout - sum_dp)
    
    def __repr__(self):
        return f"Softmax(axis={self.axis})"
