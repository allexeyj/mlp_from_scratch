import numpy as np
from typing import Tuple


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification.
    
    Combines Softmax + Negative Log Likelihood for numerical stability.
    
    L = -log(softmax(logits)[y])
      = -logits[y] + log(sum(exp(logits)))
    """
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss.
        
        Args:
            logits: Raw scores, shape (batch_size, num_classes)
            targets: Class indices, shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        batch_size = logits.shape[0]
        
        # Numerical stability
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        
        # log(sum(exp(x))) = log_sum_exp
        log_sum_exp = np.log(np.exp(logits_stable).sum(axis=1))
        
        # -log(softmax(logits)[y]) = -logits[y] + log_sum_exp
        correct_logits = logits_stable[np.arange(batch_size), targets]
        loss = -correct_logits + log_sum_exp
        
        # Cache for backward
        self.cache["logits_stable"] = logits_stable
        self.cache["targets"] = targets
        self.cache["batch_size"] = batch_size
        
        return loss.mean()
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. logits.
        
        dL/d(logits) = softmax(logits) - one_hot(targets)
        
        Returns:
            Gradient, shape (batch_size, num_classes)
        """
        logits_stable = self.cache["logits_stable"]
        targets = self.cache["targets"]
        batch_size = self.cache["batch_size"]
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Gradient: probs - one_hot
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size
        
        return grad
    
    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)


def compute_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    predictions = logits.argmax(axis=1)
    return (predictions == targets).mean()
