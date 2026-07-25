"""
Forge -- LoRA fine-tuning from scratch, adapting a frozen Glassbox transformer,
now with modern extensions: DoRA, QLoRA-style quantization, adapter save/load
and hot-swap, and perplexity evaluation.

Everything (forward AND backward) is hand-written and gradient-checked. Only the
tiny adapters train; the base is provably frozen (and can be int8-quantized to
cut its memory ~4x).

Forge is self-contained -- the transformer it fine-tunes lives in
forge.backbone, so there's nothing else to install (NumPy only).

Quick start:

    from forge.backbone.model import GPT
    from forge import apply_lora, lora_params_and_grads, count_params

    base = GPT(vocab_size=V, d_model=64, n_heads=4, n_layers=2, block_size=32)
    apply_lora(base, rank=4, alpha=8.0)      # freeze base, attach adapters
    # ... train with optimizer.step(lora_params_and_grads(base)) ...

Modern building blocks:
    from forge import DoRALinear, QuantizedLoRALinear
    from forge import save_adapters, load_adapters, perplexity
"""

from .lora import LoRALinear
from .dora import DoRALinear
from .quant import (QuantizedLoRALinear, quantize_int8, dequantize_int8,
                    quantization_error)
from .adapt import (LoRAHead, DoRAHead, apply_lora, lora_params_and_grads, count_params)
from .adapters import (save_adapters, load_adapters, adapter_nbytes, perplexity)

__version__ = "0.3.0"

__all__ = [
    "LoRALinear", "DoRALinear",
    "QuantizedLoRALinear", "quantize_int8", "dequantize_int8", "quantization_error",
    "LoRAHead", "DoRAHead", "apply_lora", "lora_params_and_grads", "count_params",
    "save_adapters", "load_adapters", "adapter_nbytes", "perplexity",
]
