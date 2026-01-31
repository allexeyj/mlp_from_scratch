import os
import numpy as np
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_transforms(img_size: int = 224):
    """ImageNet normalization transforms."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_cifar10(data_dir: str, transform):
    """Load CIFAR10 datasets."""
    train_ds = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    return train_ds, test_ds


def create_feature_extractor(model_name: str, device: torch.device):
    """Create frozen feature extractor from timm."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Extracting"
) -> tuple:
    """Extract features from a data loader."""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            features = model(images.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)


def normalize_features(X_train: np.ndarray, X_test: np.ndarray):
    """Normalize features using train statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm, mean, std


def extract_embeddings(
    model_name: str = "vit_small_patch16_224",
    data_dir: str = "./data",
    batch_size: int = 256,
    normalize: bool = True
) -> dict:
    """
    Extract embeddings from pretrained model.
    
    Returns:
        dict with X_train, y_train, X_test, y_test, embedding_dim
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms and datasets
    transform = get_transforms()
    train_ds, test_ds = get_cifar10(data_dir, transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Feature extractor
    print(f"Loading {model_name}...")
    model = create_feature_extractor(model_name, device)
    
    # Extract
    X_train, y_train = extract_features(model, train_loader, device, "Train")
    X_test, y_test = extract_features(model, test_loader, device, "Test")
    
    print(f"Extracted: train={X_train.shape}, test={X_test.shape}")
    
    # Normalize
    if normalize:
        X_train, X_test, _, _ = normalize_features(X_train, X_test)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "embedding_dim": X_train.shape[1]
    }


def load_or_extract_embeddings(
    cache_path: str = "./data/embeddings.npz",
    **kwargs
) -> dict:
    """Load cached embeddings or extract if not found."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "embedding_dim": data["X_train"].shape[1]
        }
    
    print("Extracting embeddings...")
    result = extract_embeddings(**kwargs)
    
    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=result["X_train"],
        y_train=result["y_train"],
        X_test=result["X_test"],
        y_test=result["y_test"]
    )
    print(f"Cached embeddings to {cache_path}")
    
    return result
