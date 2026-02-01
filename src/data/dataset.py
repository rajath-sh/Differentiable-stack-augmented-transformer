"""
PyTorch Dataset classes for Stack-Augmented Transformer training.
"""

import torch
from torch.utils.data import Dataset


class StackDataset(Dataset):
    """
    Dataset for bracket validation and arithmetic expression tasks.
    Handles tokenization with the provided tokenizer.
    """
    
    def __init__(self, sequences, labels, tokenizer, max_length=64):
        """
        Args:
            sequences: List of input strings
            labels: List of integer labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length after tokenization
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of dataset items
    
    Returns:
        input_ids: Tensor of shape (batch_size, seq_len)
        attention_mask: Tensor of shape (batch_size, seq_len)
        labels: Tensor of shape (batch_size,)
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return input_ids, attention_mask, labels
