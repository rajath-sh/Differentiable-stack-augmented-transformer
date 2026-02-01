"""
Baseline Transformer model: DistilBERT encoder + simple classification head.
"""

import torch.nn as nn
from transformers import AutoModel

from ..config import MODEL_NAME, DROPOUT


class PretrainedBaseline(nn.Module):
    """
    Baseline model using a pretrained DistilBERT encoder with a simple
    classification head on top of the [CLS] token representation.
    """
    
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Simple classifier on top of [CLS] token
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(self.hidden_size // 2, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        
        Returns:
            logits: Classification logits, shape (batch_size, 2)
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def freeze_encoder(self):
        """Freeze the encoder parameters (only train classification head)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
