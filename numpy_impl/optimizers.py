import numpy as np
from typing import Dict, Tuple


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, params: Dict[str, np.ndarray], lr: float):
        self.params = params
        self.lr = lr
    
    def step(self, grads: Dict[str, np.ndarray]):
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset gradients (not needed for numpy, but for API consistency)."""
        pass
    
    def set_lr(self, lr: float):
        self.lr = lr


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    
    v = momentum * v - lr * grad
    param += v
    """
    
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Velocity
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads: Dict[str, np.ndarray]):
        for key in self.params:
            grad = grads[key]
            
            # L2 regularization (adds to gradient)
            if self.weight_decay > 0 and "W" in key:
                grad = grad + self.weight_decay * self.params[key]
            
            # Momentum
            self.v[key] = self.momentum * self.v[key] - self.lr * grad
            self.params[key] += self.v[key]


class AdamW(Optimizer):
    """
    AdamW: Adam with Decoupled Weight Decay.
    
    Key difference from Adam + L2:
    - Adam + L2: grad = grad + λ·param, then adaptive update
    - AdamW: adaptive update, then param -= lr·λ·param
    
    This matters because Adam scales gradients adaptively,
    and we don't want weight decay to be scaled the same way.
    
    Algorithm:
        m = β₁·m + (1-β₁)·g           # First moment (momentum)
        v = β₂·v + (1-β₂)·g²          # Second moment (adaptive lr)
        m̂ = m / (1-β₁ᵗ)               # Bias correction
        v̂ = v / (1-β₂ᵗ)               # Bias correction
        
        # AdamW update (decoupled weight decay)
        param -= lr·(m̂/√(v̂+ε) + λ·param)
    """
    
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Timestep
        self.t = 0
        
        # First moment (mean of gradients)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        
        # Second moment (mean of squared gradients)
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads: Dict[str, np.ndarray]):
        """Perform one optimization step."""
        self.t += 1
        
        for key in self.params:
            g = grads[key]
            
            # ============================================
            # Update biased first moment estimate
            # m = β₁·m + (1-β₁)·g
            # ============================================
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            
            # ============================================
            # Update biased second raw moment estimate
            # v = β₂·v + (1-β₂)·g²
            # ============================================
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            # ============================================
            # Bias correction
            # At t=1: m̂ = m/(1-0.9) = 10m (corrects for init at 0)
            # As t→∞: m̂ → m
            # ============================================
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # ============================================
            # Compute Adam update
            # Adaptive learning rate: lr / √(v̂+ε)
            # Direction: m̂ (smoothed gradient)
            # ============================================
            adam_update = m_hat / (np.sqrt(v_hat) + self.eps)
            
            # ============================================
            # Apply update with DECOUPLED weight decay
            # Only apply weight decay to weights, not biases
            # ============================================
            if self.weight_decay > 0 and "W" in key:
                # param = param - lr * (adam_update + λ * param)
                self.params[key] -= self.lr * (
                    adam_update + self.weight_decay * self.params[key]
                )
            else:
                self.params[key] -= self.lr * adam_update
    
    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            "t": self.t,
            "m": self.m.copy(),
            "v": self.v.copy()
        }
    
    def load_state_dict(self, state: dict):
        """Load optimizer state."""
        self.t = state["t"]
        self.m = state["m"]
        self.v = state["v"]
