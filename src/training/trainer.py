"""
Training loop and utilities for Stack-Augmented Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRADIENT_CLIP,
    LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA
)


class Trainer:
    """
    Trainer class for training and evaluating models.
    """
    
    def __init__(self, model, device, model_name="Model"):
        """
        Args:
            model: The model to train
            device: Device to use (cuda/cpu)
            model_name: Name for logging
        """
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_accuracies = []
    
    def train(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
              freeze_encoder=True):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            freeze_encoder: Whether to freeze the encoder
        
        Returns:
            train_losses: List of training losses per epoch
            val_accuracies: List of validation accuracies per epoch
        """
        # Freeze encoder if requested
        if freeze_encoder:
            self.model.freeze_encoder()
            print(f"  [{self.model_name}] Encoder frozen")
        
        trainable_params = self.model.get_trainable_params()
        print(f"  [{self.model_name}] Trainable parameters: {trainable_params:,}")
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if LR_SCHEDULER == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
            )
        
        # Training loop
        self.train_losses = []
        self.val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"  [{self.model_name}] Epoch {epoch+1}/{epochs}",
                       leave=False)
            
            for input_ids, attention_mask, labels in pbar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            self.train_losses.append(avg_loss)
            
            # Validation phase
            val_acc = self.evaluate(val_loader)
            self.val_accuracies.append(val_acc)
            
            # Step scheduler
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"  [{self.model_name}] Epoch {epoch+1:2d}: "
                  f"Loss = {avg_loss:.4f}, Val Acc = {val_acc:.1f}%, "
                  f"LR = {current_lr:.6f}")
        
        return self.train_losses, self.val_accuracies
    
    def evaluate(self, loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            loader: DataLoader for evaluation data
        
        Returns:
            accuracy: Accuracy percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                predictions = logits.argmax(dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def get_history(self):
        """Get training history."""
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_model(self, save_path):
        """
        Save the trained model checkpoint.
        
        Args:
            save_path: Path to save the model (.pt file)
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        torch.save(checkpoint, save_path)
        print(f"  ✓ Model saved: {save_path}")
