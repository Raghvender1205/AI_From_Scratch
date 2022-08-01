"""
Training and Testing a PyTorch Model
"""
import enum
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm.auto import tqdm
from typing import Dict, Tuple, List

def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                optimizer: Optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Train a PyTorch Model for a single step
    
    It returns a Tuple of Training Loss and Training Accuracy.
    """
    model.train()

    train_loss, train_acc = 0, 0

    # Iterate through dataLoader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward Pass
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()

        # Backward Pass
        loss.backward()
        optimizer.step() # Update Weights

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: nn.Module, dataloader: DataLoader, 
                loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Testing a PyTorch Model for a single Epoch by turing the model to 'eval' mode 
    and then perform a Forward pass on a test dataset
    """
    model.eval()
    test_loss, test_acc = 0, 0

    # Turn on Inference Context Manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calc. accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_acc, test_loss

def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
            optimizer: Optimizer, loss_fn: nn.Module, epochs: int, device: torch.device) -> Dict[str, List]:
    """
    Train and Test a PyTorch Model
    """
    # Empty results dict
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    model.to(device)

    # Iterate through training and test steps for epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, 
                                            loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results