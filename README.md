# README.md

```markdown
# 🧠 MLP from Scratch: AdamW + Cosine Scheduler

Реализация обучения нейросети **с нуля на NumPy** для понимания того, что происходит под капотом PyTorch.

**Задача:** классификация CIFAR-10 на эмбеддингах из замороженного ViT.

```
Frozen ViT → Embeddings → MLP (наша реализация) → 10 классов
```

---

## 🎯 Что реализовано вручную

| Компонент | Описание |
|-----------|----------|
| `Linear` | Полносвязный слой с He-инициализацией |
| `ReLU`, `GELU` | Функции активации |
| `Dropout` | Регуляризация |
| `CrossEntropyLoss` | Softmax + NLL (numerically stable) |
| `AdamW` | Adam с decoupled weight decay |
| `CosineScheduler` | Cosine annealing с warmup |
| `Backpropagation` | Chain rule через все слои |

---

## 📁 Структура проекта

```
mlp_from_scratch/
├── config.py              # Гиперпараметры
├── requirements.txt
├── run_exp_1.py           # 🔧 NumPy реализация
├── run_exp_2.py           # 🔥 PyTorch baseline
│
├── data/
│   ├── __init__.py
│   ├── embeddings.py      # Извлечение эмбеддингов из ViT
│   └── batching.py        # Детерминированные батчи
│
├── numpy_impl/
│   ├── __init__.py
│   ├── layers.py          # Linear, Dropout
│   ├── activations.py     # ReLU, GELU, Softmax
│   ├── losses.py          # CrossEntropyLoss
│   ├── model.py           # MLP, Sequential
│   ├── optimizers.py      # SGD, AdamW
│   └── schedulers.py      # CosineScheduler
│
├── torch_impl/
│   ├── __init__.py
│   ├── model.py           # TorchMLP
│   └── trainer.py         # Training loop
│
└── utils/
    ├── __init__.py
    └── reproducibility.py # Seed management
```

---

## 🚀 Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/mlp-from-scratch.git
cd mlp-from-scratch

# Установить зависимости
pip install -r requirements.txt

# Залогиниться в wandb
wandb login

# Запустить NumPy эксперимент
python run_exp_1.py

# Запустить PyTorch эксперимент
python run_exp_2.py
```

---

## 📊 Ожидаемые результаты

| Эксперимент | Test Accuracy | Время (GPU) |
|-------------|---------------|-------------|
| NumPy (exp1) | ~97.5% | ~2 мин |
| PyTorch (exp2) | ~97.5% | ~1 мин |

Метрики логируются в [Weights & Biases](https://wandb.ai).

---

## 📐 Теория

### AdamW (Adam with Decoupled Weight Decay)

Ключевое отличие от Adam + L2:

```
❌ Adam + L2:    grad = grad + λ·θ,  затем Adam update
✅ AdamW:        Adam update,        затем θ -= lr·λ·θ
```

**Почему это важно?** Adam масштабирует градиенты адаптивно (делит на √v). При L2 регуляризация тоже масштабируется, что нежелательно.

**Формулы:**
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t           # Первый момент (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²          # Второй момент (adaptive lr)
m̂_t = m_t / (1-β₁ᵗ)                     # Bias correction
v̂_t = v_t / (1-β₂ᵗ)                     # Bias correction
θ_t = θ_{t-1} - lr·(m̂_t/√(v̂_t+ε) + λ·θ_{t-1})  # Decoupled!
```

### Cosine Scheduler

```
lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))
```

```
lr_max ─┐
        │╲
        │ ╲
        │  ╲      ← Плавное затухание
        │   ╲
        │    ╲
lr_min ─┴─────╲───
        0    T_max
```

С warmup: первые N шагов lr растёт линейно от 0 до lr_max.

### Chain Rule (Backpropagation)

Каждый слой в `backward()` применяет chain rule:

```
dL/dx = dL/dy · dy/dx
        ↑        ↑
        │        └── локальный градиент (вычисляем)
        └── градиент "сверху" (получаем как dout)
```

**Пример для Linear (y = x @ W + b):**
```python
def backward(self, dout):
    # dout = dL/dy (пришёл сверху)
    
    dW = x.T @ dout      # dL/dW = dL/dy · dy/dW
    db = dout.sum(0)     # dL/db = dL/dy · dy/db  
    dx = dout @ W.T      # dL/dx = dL/dy · dy/dx (передаём дальше)
    
    return dx
```

**Пример для ReLU (y = max(0, x)):**
```python
def backward(self, dout):
    # dy/dx = 1 if x > 0 else 0
    return dout * (x > 0)
```

---

## ⚙️ Конфигурация

Редактируйте `config.py`:

```python
@dataclass
class Config:
    # Model
    hidden_dims: List[int] = [512, 256]
    dropout: float = 0.0
    
    # Training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    
    # Adam
    beta1: float = 0.9
    beta2: float = 0.999
```

---

## 🔬 Эксперименты

### Exp 1: NumPy (from scratch)

Все компоненты реализованы вручную. PyTorch используется **только** для извлечения эмбеддингов.

```python
# Полностью наш код:
model = MLP(input_dim=384, hidden_dims=[512, 256], output_dim=10)
optimizer = AdamW(model.get_params(), lr=3e-4, weight_decay=0.05)
scheduler = CosineScheduler(optimizer, total_steps=1000, warmup_steps=100)

for epoch in range(epochs):
    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        loss = criterion(logits, y_batch)
        dout = criterion.backward()
        model.backward(dout)
        optimizer.step(model.get_grads())
        scheduler.step()
```

### Exp 2: PyTorch (baseline)

Стандартный PyTorch для сравнения:

```python
model = TorchMLP(input_dim=384, hidden_dims=[512, 256], output_dim=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

for epoch in range(epochs):
    for X_batch, y_batch in loader:
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

---

## 📈 Визуализация в W&B

После запуска экспериментов в W&B будут доступны:

- Loss curves (train/test)
- Accuracy curves (train/test)  
- Learning rate schedule
- Сравнение NumPy vs PyTorch

---

## 🧪 Тестирование

```bash
# Проверить что forward/backward работают корректно
python -c "
from numpy_impl import MLP, CrossEntropyLoss
import numpy as np

# Smoke test
model = MLP(input_dim=10, hidden_dims=[32], output_dim=3)
X = np.random.randn(4, 10)
y = np.array([0, 1, 2, 1])

logits = model.forward(X)
loss = CrossEntropyLoss()(logits, y)
print(f'Loss: {loss:.4f}')
print('✅ Forward OK')

dout = CrossEntropyLoss().backward()
model.backward(dout)
print('✅ Backward OK')
"
```

---

## 📚 Ссылки

- [AdamW Paper](https://arxiv.org/abs/1711.05101) — Decoupled Weight Decay Regularization
- [Cosine Annealing Paper](https://arxiv.org/abs/1608.03983) — SGDR: Stochastic Gradient Descent with Warm Restarts
- [ViT Paper](https://arxiv.org/abs/2010.11929) — An Image is Worth 16x16 Words
- [Backpropagation Explained](http://cs231n.stanford.edu/slides/2024/lecture_4.pdf) — CS231n Lecture

---

## 📝 License

MIT

---

<p align="center">
  <b>Сделано для понимания того, что скрывается за <code>loss.backward()</code></b>
</p>
```

---

Готово! Хочешь что-то добавить/изменить — badges, секцию contributing, больше теории?
