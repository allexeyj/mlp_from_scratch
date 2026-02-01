"""
Experiment 1: Training MLP from scratch using NumPy.

All components (layers, optimizer, scheduler, loss) are implemented manually.
Only uses PyTorch/timm for extracting frozen embeddings.
"""

import numpy as np
import wandb

from config import get_config
from data import load_or_extract_embeddings
from numpy_impl import MLP, AdamW, CosineScheduler, CrossEntropyLoss


def create_batches(X, y, batch_size, shuffle=True):
    """Generator for mini-batches."""
    n = len(y)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train_epoch(model, optimizer, scheduler, criterion, X, y, batch_size):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in create_batches(X, y, batch_size, shuffle=True):
        # Forward
        logits = model.forward(X_batch)

        # Loss
        loss = criterion(logits, y_batch)
        total_loss += loss * len(y_batch)

        # Accuracy
        preds = logits.argmax(axis=1)
        total_correct += (preds == y_batch).sum()
        total_samples += len(y_batch)

        # Backward
        dout = criterion.backward()
        model.backward(dout)

        # Optimizer & Scheduler step
        grads = model.get_grads()
        optimizer.step(grads)
        scheduler.step()

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }


def evaluate(model, criterion, X, y, batch_size):
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in create_batches(X, y, batch_size, shuffle=False):
        logits = model.forward(X_batch)

        loss = criterion(logits, y_batch)
        total_loss += loss * len(y_batch)

        preds = logits.argmax(axis=1)
        total_correct += (preds == y_batch).sum()
        total_samples += len(y_batch)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }


def run_experiment_1():
    """Run NumPy-based training experiment."""
    config = get_config()

    # Initialize wandb
    wandb.init(
        project=config.project,
        name="exp1_numpy",
        config=vars(config),
        tags=["numpy", "from-scratch"]
    )

    # Set seed
    np.random.seed(config.seed)

    print("=" * 70)
    print("EXPERIMENT 1: NumPy Implementation")
    print("=" * 70)

    # Load embeddings
    print("\n[1/4] Loading embeddings...")
    data = load_or_extract_embeddings(
        model_name=config.embedding_model,
        data_dir=config.data_dir
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    input_dim = data["embedding_dim"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create model
    print("\n[2/4] Creating model...")
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm
    )
    print(model)

    # Create optimizer
    print("\n[3/4] Creating optimizer and scheduler...")
    params = model.get_params()

    optimizer = AdamW(
        params=params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )

    # Scheduler
    steps_per_epoch = (len(y_train) + config.batch_size - 1) // config.batch_size
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    scheduler = CosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        lr_min=config.lr_min,
        warmup_start_factor=config.warmup_start_factor
    )

    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Loss
    criterion = CrossEntropyLoss()

    # Training
    print("\n[4/4] Training...")
    print("-" * 70)

    best_acc = 0.0

    for epoch in range(config.epochs):
        # Train
        train_metrics = train_epoch(
            model, optimizer, scheduler, criterion,
            X_train, y_train, config.batch_size
        )

        # Evaluate
        test_metrics = evaluate(
            model, criterion,
            X_test, y_test, config.batch_size
        )

        current_lr = scheduler.get_lr()

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "lr": current_lr
        })

        # Print progress
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            marker = " ★"
        else:
            marker = ""

        print(
            f"Epoch {epoch+1:2d}/{config.epochs} | "
            f"Train: {train_metrics['loss']:.4f} / {train_metrics['accuracy']:.4f} | "
            f"Test: {test_metrics['loss']:.4f} / {test_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.2e}{marker}"
        )

    print("-" * 70)
    print(f"Best Test Accuracy: {best_acc:.4f}")

    wandb.log({"best_test_accuracy": best_acc})
    wandb.finish()

    return model, best_acc


if __name__ == "__main__":
    run_experiment_1()
