"""Causal decoder with an optional, explicitly experimental residual gate."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .tokenizer import ByteTokenizer


@dataclass
class ModelConfig:
    dim: int = 128
    layers: int = 2
    heads: int = 4
    context: int = 128
    dropout: float = 0.0
    coherence_gate: bool = False
    gradient_checkpointing: bool = False
    vocab_size: int = ByteTokenizer.vocab_size

    def __post_init__(self):
        if min(self.dim, self.layers, self.heads, self.context) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.dim % self.heads or self.vocab_size != ByteTokenizer.vocab_size:
            raise ValueError("dim must divide into heads; vocabulary must match byte tokenizer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


PRESETS = {
    "cpu": ModelConfig(),
    "2gb": ModelConfig(dim=384, layers=6, heads=6, context=512, gradient_checkpointing=True),
    "4gb": ModelConfig(dim=512, layers=8, heads=8, context=1024, gradient_checkpointing=True),
}


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(cfg.dim, cfg.dim * 4), nn.GELU(),
                                nn.Linear(cfg.dim * 4, cfg.dim), nn.Dropout(cfg.dropout))
        # Per-token feature gate: never pools over future tokens. Zero init
        # makes the experimental path initially identical to the baseline.
        # Keep baseline initialization identical under a paired random seed.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Linear(cfg.dim, cfg.dim) if cfg.coherence_gate else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.cfg.heads, d // self.cfg.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.cfg.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        h = self.norm2(x)
        update = self.ff(h)
        if self.gate is not None:
            update = update * (2 * torch.sigmoid(self.gate(h)))
        return x + update


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.position = nn.Embedding(cfg.context, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        # Output projection shares embedding weights, saving parameters/memory.

    def forward(self, input_ids, labels=None):
        t = input_ids.shape[1]
        if t < 1 or t > self.config.context:
            raise ValueError("Input length must be between 1 and configured context")
        x = self.embedding(input_ids) + self.position(torch.arange(t, device=input_ids.device))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if (
                self.config.gradient_checkpointing and self.training and torch.is_grad_enabled()
            ) else block(x)
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None
        if labels is not None:
            # Dataset supplies already-shifted next-token labels. Prompt and
            # right-padding targets are -100 and excluded from the objective.
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                   labels.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, ids, *, max_new_tokens=128, temperature=0.0, top_k=40, seed=42):
        if max_new_tokens < 0 or temperature < 0 or top_k < 1:
            raise ValueError("Invalid generation parameters")
        prior_mode = self.training
        self.eval()
        device = next(self.parameters()).device
        rng = torch.Generator(device=device).manual_seed(seed)
        tokens = list(ids) or [ByteTokenizer.bos_id]
        generated = []
        try:
            for _ in range(max_new_tokens):
                # Sliding window is deliberate; no unbounded KV allocation.
                x = torch.tensor([tokens[-self.config.context:]], device=device)
                logits = self(x)[0][0, -1].float()
                logits[[ByteTokenizer.pad_id, ByteTokenizer.bos_id, ByteTokenizer.sep_id]] = -float("inf")
                if temperature == 0:
                    nxt = int(logits.argmax())
                else:
                    values, indices = torch.topk(logits / temperature, min(top_k, len(logits)))
                    nxt = int(indices[torch.multinomial(values.softmax(-1), 1, generator=rng)])
                if nxt == ByteTokenizer.eos_id:
                    break
                generated.append(nxt)
                tokens.append(nxt)
        finally:
            self.train(prior_mode)
        return generated


def memory_report(cfg: ModelConfig) -> dict:
    # Meta device counts parameters without allocating model tensors.
    with torch.device("meta"):
        model = LanguageModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return {"config": asdict(cfg), "parameters": n,
            "fp32_weights_mib": round(n * 4 / 2**20, 1),
            "fp32_adam_weights_grad_moments_mib": round(n * 16 / 2**20, 1),
            "note": "Excludes activations, attention workspace, CUDA context and allocator. GPU fit is unverified."}
