"""
Experiment 2: Training MLP using PyTorch.

Standard PyTorch training loop with built-in components.
Shows how simple it is compared to implementing from scratch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from config import get_config
from data import load_or_extract_embeddings
from torch_impl import TorchMLP, TorchTrainer, create_dataloaders


def run_experiment_2():
    """Run PyTorch-based training experiment."""
    config = get_config()
    
    # Initialize wandb
    wandb.init(
        project=config.project,
        name="exp2_pytorch",
        config=vars(config),
        tags=["pytorch", "baseline"]
    )
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("EXPERIMENT 2: PyTorch Implementation")
    print("=" * 70)
    print(f"Device: {device}")
    
    # Load embeddings
    print("\n[1/4] Loading embeddings...")
    data = load_or_extract_embeddings(
        model_name=config.embedding_model,
        data_dir=config.data_dir
    )
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    input_dim = data["embedding_dim"]
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=config.batch_size
    )
    
    # Create model
    print("\n[2/4] Creating model...")
    model = TorchMLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
        dropout=config.dropout
    )
    print(model)
    
    # Optimizer
    print("\n[3/4] Creating optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # Scheduler: Warmup + Cosine
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.lr_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Trainer
    trainer = TorchTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Training
    print("\n[4/4] Training...")
    print("-" * 70)
    
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
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
    run_experiment_2()
