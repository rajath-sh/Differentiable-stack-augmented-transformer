"""
Stack-Augmented Transformer model: DistilBERT encoder + stack module + classifier.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from ..config import MODEL_NAME, STACK_DEPTH, DROPOUT


class StackAugmentedPretrained(nn.Module):
    """
    Stack-Augmented model using a pretrained DistilBERT encoder with an
    additional stack module that helps track hierarchical structure (e.g., 
    bracket depth, nesting level).
    
    The stack module learns to produce a state vector that captures depth
    information, which is concatenated with the encoder output for classification.
    """
    
    def __init__(self, model_name=MODEL_NAME, stack_depth=STACK_DEPTH):
        super().__init__()
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.stack_depth = stack_depth
        
        # STACK MODULE: Learns to track hierarchical depth/structure
        # Takes the [CLS] representation and produces a stack state vector
        self.stack_module = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, stack_depth),
            nn.Sigmoid()  # Stack state as probabilities/activations
        )
        
        # Auxiliary depth predictor (helps with learning to track depth)
        # This is a multi-task learning component that encourages the model
        # to learn depth-aware representations
        self.depth_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Normalized depth prediction
        )
        
        # Final classifier: combines encoder output with stack state
        # This allows the model to use both semantic information from the
        # encoder and structural information from the stack module
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + stack_depth, self.hidden_size // 2),
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
        
        # Get stack state (learns to track structural depth)
        stack_state = self.stack_module(pooled)
        
        # Auxiliary depth prediction (for learning, not used in inference)
        # This encourages the model to learn depth-aware representations
        _ = self.depth_predictor(pooled)
        
        # Combine encoder output with stack state
        combined = torch.cat([pooled, stack_state], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits
    
    def freeze_encoder(self):
        """Freeze the encoder parameters (only train stack module and heads)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_stack_state(self, input_ids, attention_mask):
        """
        Get the stack state for visualization/debugging.
        
        Returns:
            stack_state: Stack state vector, shape (batch_size, stack_depth)
        """
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            stack_state = self.stack_module(pooled)
        return stack_state
