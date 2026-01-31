import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
import wandb


class TorchTrainer:
    """
    Standard PyTorch training loop.
    Shows how simple it is with built-in components.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Stats
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += len(y)
        
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples
        }
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += len(y)
        
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples
        }


def create_dataloaders(
    X_train, y_train, X_test, y_test,
    batch_size: int,
    shuffle_train: bool = True
):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
