import numpy as np
from .optimizers import Optimizer


class Scheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        # Make steps 0-based like many schedulers:
        # first call to step() sets current_step = 0
        self.current_step = -1

    def step(self):
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.optimizer.lr


class CosineScheduler(Scheduler):
    """
    Cosine Annealing Learning Rate Scheduler with optional warmup.

    Warmup is linear in *factor space*:
        lr = lr_max * (start_factor + (1 - start_factor) * progress)

    where progress goes from 0 to 1 during warmup.

    After warmup:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        lr_min: float = 0.0,
        warmup_start_factor: float = 0.0,
    ):
        super().__init__(optimizer)

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if not (0.0 <= warmup_start_factor <= 1.0):
            raise ValueError(
                f"warmup_start_factor must be in [0, 1], got {warmup_start_factor}"
            )

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.lr_min = float(lr_min)
        self.warmup_start_factor = float(warmup_start_factor)

        self.lr_max = optimizer.lr

    def step(self) -> float:
        """
        Update learning rate and return current value.
        This scheduler is step-based (call once per optimizer step).
        """
        self.current_step += 1
        s = self.current_step  # 0-based step index

        # If someone calls step() beyond total_steps, clamp to final lr
        if s >= self.total_steps:
            self.optimizer.set_lr(self.lr_min)
            return self.lr_min

        # --------------------
        # Warmup
        # --------------------
        if self.warmup_steps > 0 and s < self.warmup_steps:
            if self.warmup_steps == 1:
                progress = 1.0
            else:
                # s = 0 ... warmup_steps-1 => progress 0 ... 1
                progress = s / (self.warmup_steps - 1)

            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            lr = self.lr_max * factor
            self.optimizer.set_lr(lr)
            return lr

        # --------------------
        # Cosine
        # --------------------
        cosine_steps = self.total_steps - self.warmup_steps
        t = s - self.warmup_steps  # 0-based index in cosine segment

        if cosine_steps <= 1:
            lr = self.lr_min
        else:
            # t = 0 ... cosine_steps-1 => progress 0 ... 1
            progress = t / (cosine_steps - 1)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * progress))

        self.optimizer.set_lr(lr)
        return lr

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]


class StepScheduler(Scheduler):
    """Step decay: multiply lr by gamma every step_size steps."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def step(self) -> float:
        self.current_step += 1
        s = self.current_step

        n_decays = (s + 1) // self.step_size  # decay after step_size updates
        lr = self.base_lr * (self.gamma ** n_decays)

        self.optimizer.set_lr(lr)
        return lr
