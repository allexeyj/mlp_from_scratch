import numpy as np
from .optimizers import Optimizer


class Scheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.current_step = 0
    
    def step(self):
        raise NotImplementedError
    
    def get_lr(self) -> float:
        return self.optimizer.lr


class CosineScheduler(Scheduler):
    """
    Cosine Annealing Learning Rate Scheduler with optional warmup.
    
    During warmup (steps 0 to warmup_steps):
        lr = lr_max * (step / warmup_steps)  # Linear warmup
    
    After warmup (steps warmup_steps to total_steps):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
    
    The cosine schedule provides:
    - Slow start (flat near lr_max)
    - Gradual decrease in the middle
    - Slow end (flat near lr_min)
    
    This often works better than step decay because:
    - Smoother transitions
    - Natural "exploration then exploitation" pattern
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        lr_min: float = 0.0
    ):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.lr_max = optimizer.lr
    
    def step(self) -> float:
        """Update learning rate and return current value."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # ============================================
            # Linear warmup
            # Gradually increase from 0 to lr_max
            # ============================================
            lr = self.lr_max * (self.current_step / self.warmup_steps)
        else:
            # ============================================
            # Cosine annealing
            # progress goes from 0 to 1
            # cos(0) = 1, cos(π) = -1
            # So (1 + cos(π*progress))/2 goes from 1 to 0
            # ============================================
            progress = (self.current_step - self.warmup_steps) / \
                      max(1, self.total_steps - self.warmup_steps)
            
            # Clamp progress to [0, 1]
            progress = min(1.0, progress)
            
            # Cosine schedule
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
                 (1 + np.cos(np.pi * progress))
        
        self.optimizer.set_lr(lr)
        return lr
    
    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step
        }
    
    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]


class StepScheduler(Scheduler):
    """Step decay: multiply lr by gamma every step_size steps."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1
    ):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self) -> float:
        self.current_step += 1
        
        n_decays = self.current_step // self.step_size
        lr = self.base_lr * (self.gamma ** n_decays)
        
        self.optimizer.set_lr(lr)
        return lr
