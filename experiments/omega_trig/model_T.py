"""
Theory Model T_θ: Learning Layer

The neural network T_θ learns to answer questions about Ω-structure.

Architecture:
- Input: (x: AngleCode, i: TrigIndex)
- Output: ŷ_θ(x,i) ∈ (0,1) probability

Key principle: θ are OUTPUTS of the kinetics (theories to analyze),
not the source of structure. The Ω-structure is fixed in trig_kernel.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from trig_kernel import AngleCode, TrigIndex, TrigIndexType


# ============================================================================
# Theory Model
# ============================================================================

class TrigTheoryModel(nn.Module):
    """
    Theory T_θ: learns to evaluate questions on Ω-structure.
    
    Architecture:
    - Angle encoding: sin(θ), cos(θ) → linear projection
    - Index encoding: type embedding + threshold projection
    - MLP: combines encodings → probability
    """
    
    def __init__(self,
                 angle_embed_dim: int = 16,
                 index_embed_dim: int = 8,
                 hidden_dim: int = 64):
        super().__init__()
        
        # Angle encoding (sin, cos) → embedding
        self.angle_proj = nn.Linear(2, angle_embed_dim)
        
        # Index type embedding (4 types)
        self.index_type_embed = nn.Embedding(
            num_embeddings=len(TrigIndexType),
            embedding_dim=index_embed_dim
        )
        
        # Threshold r projection (for SIN_GE, COS_GE)
        self.r_proj = nn.Linear(1, index_embed_dim)
        
        # Combined input dimension
        input_dim = angle_embed_dim + 2 * index_embed_dim
        
        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def encode_angle(self, x: AngleCode) -> torch.Tensor:
        """Encode angle as (sin(θ), cos(θ)) → projection."""
        theta = x.to_radians()
        v = torch.tensor([math.sin(theta), math.cos(theta)], 
                        dtype=torch.float32)
        return self.angle_proj(v)
    
    def encode_angle_batch(self, angles: List[AngleCode]) -> torch.Tensor:
        """Batch encode angles."""
        thetas = [x.to_radians() for x in angles]
        v = torch.tensor([[math.sin(t), math.cos(t)] for t in thetas],
                        dtype=torch.float32)
        return self.angle_proj(v)
    
    def encode_index(self, idx: TrigIndex) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode index as type embedding + threshold embedding."""
        type_id = torch.tensor([idx.kind.value], dtype=torch.long)
        type_emb = self.index_type_embed(type_id)[0]
        
        if idx.kind in (TrigIndexType.SIN_GE, TrigIndexType.COS_GE):
            r_tensor = torch.tensor([[idx.r]], dtype=torch.float32)
            r_emb = self.r_proj(r_tensor)[0]
        else:
            r_emb = torch.zeros_like(type_emb)
        
        return type_emb, r_emb
    
    def encode_index_batch(self, indices: List[TrigIndex]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch encode indices."""
        type_ids = torch.tensor([i.kind.value for i in indices], dtype=torch.long)
        type_embs = self.index_type_embed(type_ids)
        
        r_values = []
        for idx in indices:
            if idx.kind in (TrigIndexType.SIN_GE, TrigIndexType.COS_GE):
                r_values.append([idx.r])
            else:
                r_values.append([0.0])
        
        r_tensor = torch.tensor(r_values, dtype=torch.float32)
        r_embs = self.r_proj(r_tensor)
        
        return type_embs, r_embs
    
    def forward_single(self, x: AngleCode, idx: TrigIndex, 
                        return_embedding: bool = False) -> torch.Tensor:
        """Forward pass for a single (x, idx) pair."""
        a_emb = self.encode_angle(x)
        t_emb, r_emb = self.encode_index(idx)
        
        z = torch.cat([a_emb, t_emb, r_emb], dim=-1)
        
        # Get hidden representation from first layer
        h = self.mlp[0](z)  # Linear
        h = self.mlp[1](h)  # Tanh
        
        # Continue through rest of MLP
        h2 = self.mlp[2](h)  # Linear (hidden -> hidden/2)
        h2 = self.mlp[3](h2)  # Tanh
        logit = self.mlp[4](h2)  # Linear (hidden/2 -> 1)
        prob = torch.sigmoid(logit)
        
        if return_embedding:
            return prob.squeeze(-1), h
        return prob.squeeze(-1)
    
    def forward_batch(self, 
                      angles: List[AngleCode], 
                      indices: List[TrigIndex],
                      return_embedding: bool = False) -> torch.Tensor:
        """Forward pass for a batch of (x, idx) pairs."""
        a_embs = self.encode_angle_batch(angles)  # (B, angle_embed_dim)
        t_embs, r_embs = self.encode_index_batch(indices)  # (B, index_embed_dim) each
        
        z = torch.cat([a_embs, t_embs, r_embs], dim=-1)  # (B, input_dim)
        
        # Get hidden representation from first layer
        h = self.mlp[0](z)  # Linear
        h = self.mlp[1](h)  # Tanh
        
        # Continue through rest of MLP
        h2 = self.mlp[2](h)  # Linear
        h2 = self.mlp[3](h2)  # Tanh
        logits = self.mlp[4](h2)  # Linear
        probs = torch.sigmoid(logits)
        
        if return_embedding:
            return probs.squeeze(-1), h  # (B,), (B, hidden_dim)
        return probs.squeeze(-1)  # (B,)
    
    def forward(self, x, idx=None, return_embedding: bool = False):
        """Flexible forward: single or batch."""
        if isinstance(x, AngleCode) and isinstance(idx, TrigIndex):
            return self.forward_single(x, idx, return_embedding)
        elif isinstance(x, list):
            return self.forward_batch(x, idx, return_embedding)
        else:
            raise ValueError(f"Unsupported input types: {type(x)}, {type(idx)}")


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_predictions(model: TrigTheoryModel, 
                    data: list, 
                    threshold: float = 0.5) -> List[int]:
    """Get binary predictions for a dataset."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for q in data:
            prob = model(q.x, q.idx).item()
            y_hat = 1 if prob >= threshold else 0
            predictions.append(y_hat)
    
    return predictions


# ============================================================================
# Main: Self-test
# ============================================================================

if __name__ == "__main__":
    print("=== Theory Model Tests ===\n")
    
    from trig_kernel import generate_X_trig, generate_I_trig
    
    # Create model
    model = TrigTheoryModel(
        angle_embed_dim=16,
        index_embed_dim=8,
        hidden_dim=64
    )
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test single forward
    X = generate_X_trig()
    I = generate_I_trig()
    
    x = X[45]  # 45 degrees
    idx = I[0]  # SIGN_SIN
    
    prob = model(x, idx)
    print(f"\nSingle forward:")
    print(f"  Input: x={x}, idx={idx}")
    print(f"  Output: prob={prob.item():.4f}")
    
    # Test batch forward
    angles = [X[0], X[45], X[90], X[135]]
    indices = [I[0], I[0], I[0], I[0]]
    
    probs = model(angles, indices)
    print(f"\nBatch forward ({len(angles)} samples):")
    print(f"  Output probs: {probs.tolist()}")
    
    # Verify output range
    assert 0 <= prob.item() <= 1, "Probability out of range!"
    assert all(0 <= p <= 1 for p in probs.tolist()), "Batch probabilities out of range!"
    
    # Test gradient flow
    probs.sum().backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"\nGradient check: total grad magnitude = {grad_sum:.4f}")
    assert grad_sum > 0, "No gradients!"
    
    print("\n=== All model tests passed! ===")
