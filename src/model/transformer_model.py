"""PyTorch transformer that scores (state, action) pairs."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 256
    dropout: float = 0.1
    use_policy_head: bool = False  # Optional policy head for AlphaZero-style training
    num_actions: int = 1858  # Number of possible chess moves
    use_material_head: bool = True  # Material evaluation auxiliary task
    material_loss_weight: float = 0.1  # Weight for material prediction loss

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class ActionValueTransformer(nn.Module):
    """Transformer encoder that predicts win probability logits.
    
    Optionally includes:
    - Policy head for move probability prediction (AlphaZero-style)
    - Material head for material balance prediction (helps learn piece values)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Value head (predicts win probability)
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        
        # Optional policy head (predicts move probabilities)
        if config.use_policy_head:
            self.policy_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.num_actions),
            )
        else:
            self.policy_head = None
        
        # Optional material head (predicts material balance)
        # This helps the model learn piece values explicitly
        if config.use_material_head:
            self.material_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, 1),
            )
        else:
            self.material_head = None
        
        # Backward compatibility: keep 'head' as alias to value_head
        self.head = self.value_head

    def forward(
        self, 
        tokens: torch.Tensor, 
        return_policy: bool = False,
        return_material: bool = False
    ) -> torch.Tensor | tuple:
        """Forward pass through the model.
        
        Args:
            tokens: Input token sequence [batch, seq_len]
            return_policy: If True and policy head exists, return policy logits
            return_material: If True and material head exists, return material prediction
        
        Returns:
            - If only value: value_logits [batch]
            - If value + policy: (value_logits, policy_logits)
            - If value + material: (value_logits, material_pred)
            - If all: (value_logits, policy_logits, material_pred)
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        positions = self.pos_embedding[:, : tokens.size(1)]
        hidden = self.embedding(tokens) + positions
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]
        
        # Value output (always computed)
        value_logits = self.value_head(pooled).squeeze(-1)
        
        # Collect optional outputs
        outputs = [value_logits]
        
        # Policy output (if requested and available)
        if return_policy and self.policy_head is not None:
            policy_logits = self.policy_head(pooled)
            outputs.append(policy_logits)
        
        # Material output (if requested and available)
        if return_material and self.material_head is not None:
            material_pred = self.material_head(pooled).squeeze(-1)
            outputs.append(material_pred)
        
        # Return single value or tuple
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

