# 🧠 MLP from Scratch: AdamW + Cosine Scheduler

Реализация обучения нейросети **с нуля на NumPy** для понимания того, что происходит под капотом PyTorch.

**Цель проекта (образовательная):** понять оптимизацию, численную стабильность, backprop и устройство тренировочного цикла — без усложнений end-to-end CV.

**Задача:** классификация CIFAR-10 на эмбеддингах из замороженного ViT.

```
Frozen ViT → Embeddings → MLP (наша NumPy-реализация) → 10 классов
```

---

## 🎯 Что реализовано вручную (NumPy)

| Компонент | Описание |
|----------|----------|
| `Linear` | Полносвязный слой с He-инициализацией |
| `ReLU`, `GELU` | Функции активации |
| `Dropout` | Регуляризация |
| `CrossEntropyLoss` | Softmax + NLL (numerically stable) |
| `AdamW` | Adam с decoupled weight decay |
| `CosineScheduler` | Cosine annealing с warmup |
| `Backpropagation` | Chain rule через все слои |

---

## 📁 Структура проекта (актуальная)

```
mlp_from_scratch/
├── config.py
├── requirements.txt
├── run_exp_1.py           # NumPy реализация (from scratch)
├── run_exp_2.py           # PyTorch baseline
├── grad_check.py          # Численный градиент-чек (finite differences)
│
├── data/
│   ├── __init__.py
│   └── embeddings.py      # Извлечение эмбеддингов из ViT + кэширование
│
├── numpy_impl/
│   ├── __init__.py
│   ├── layers.py          # Linear, Dropout, base Layer
│   ├── activations.py     # ReLU, GELU, Softmax
│   ├── losses.py          # CrossEntropyLoss (+ compute_accuracy)
│   ├── model.py           # MLP, Sequential
│   ├── optimizers.py      # SGD, AdamW
│   └── schedulers.py      # CosineScheduler, StepScheduler
│
└── torch_impl/
    ├── __init__.py
    ├── model.py           # TorchMLP
    └── trainer.py         # Training loop (+ create_dataloaders)
```

---

## 🚀 Быстрый старт

```bash
pip install -r requirements.txt
wandb login

python run_exp_1.py
python run_exp_2.py
```

---

## 🧪 Градиент-чек (важно для обучения)

Численно проверяет backprop через `MLP + CrossEntropyLoss`:

```bash
python grad_check.py
```

---

## 🧪 Smoke test (forward/backward)

```bash
python - <<'PY'
from numpy_impl import MLP, CrossEntropyLoss
import numpy as np

np.random.seed(0)

model = MLP(input_dim=10, hidden_dims=[32], output_dim=3)
criterion = CrossEntropyLoss()

X = np.random.randn(4, 10)
y = np.array([0, 1, 2, 1])

logits = model.forward(X)
loss = criterion(logits, y)
print(f"Loss: {loss:.4f}")
print("✅ Forward OK")

dout = criterion.backward()
model.backward(dout)
print("✅ Backward OK")
PY
```

---

## ⚙️ Конфигурация

Смотри `config.py`. Важный параметр для согласования warmup между NumPy и PyTorch:

- `warmup_start_factor`: стартовый множитель learning rate на warmup (например `0.01` значит “начинаем с 1% от lr”).

---
```

---

### config.py
```py
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

    # LR schedule
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.01  # start lr = lr * warmup_start_factor
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
```

---

### numpy_impl/schedulers.py
```py
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
