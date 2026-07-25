"""
forge.backbone -- the from-scratch transformer that Forge fine-tunes.

This is a small NumPy transformer (attention, RoPE, RMSNorm/LayerNorm, SwiGLU,
KV-cache, an Adam optimizer, a char dataset, and a gradient checker). It is the
same architecture developed in the companion Glassbox project, VENDORED here so
that Forge is completely self-contained: you can clone and demo Forge on its own
with no other project on your path. NumPy is the only dependency.

Forge treats this backbone as the frozen base model and trains small LoRA/DoRA
adapters on top of it -- see forge.adapt.
"""

from .tensor_ops import (Matmul, Bias, Softmax, LayerNorm, RMSNorm, GELU,
                         CrossEntropy)
from .rope import RoPE
from .attention import (SelfAttentionHead, MultiHeadAttention,
                        GroupedQueryAttention, causal_mask)
from .model import GPT, Block, FeedForward, SwiGLU
from .optimizer import Adam
from .data import CharDataset, train, generate
from .gradcheck import numerical_grad, rel_error

__all__ = [
    "Matmul", "Bias", "Softmax", "LayerNorm", "RMSNorm", "GELU", "CrossEntropy",
    "RoPE",
    "SelfAttentionHead", "MultiHeadAttention", "GroupedQueryAttention", "causal_mask",
    "GPT", "Block", "FeedForward", "SwiGLU",
    "Adam",
    "CharDataset", "train", "generate",
    "numerical_grad", "rel_error",
]
