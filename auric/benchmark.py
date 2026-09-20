"""Bounded synthetic training benchmark for CPU development / GPU arrival."""
from dataclasses import replace
import time

import torch

from .model import PRESETS, LanguageModel, memory_report
from .training import TrainConfig, precision_for, autocast


def benchmark(preset="2gb", *, device="cpu", steps=3, batch_size=1, context=None):
    if steps < 1 or batch_size < 1:
        raise ValueError("Positive steps and batch size required")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = replace(PRESETS[preset])
    if context is not None:
        cfg = replace(cfg, context=context)
    precision = precision_for(TrainConfig(device=device))
    model = LanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    x = torch.randint(4, cfg.vocab_size, (batch_size, cfg.context), device=device)
    y = torch.roll(x, -1, dims=1)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = model(x, y)[1]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    return {**memory_report(cfg), "device": device, "precision": precision, "steps": steps,
            "batch_size": batch_size, "seconds": elapsed,
            "synthetic_tokens_per_second": steps*batch_size*cfg.context/elapsed,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if device == "cuda" else None,
            "note": "Includes cold optimizer initialization. Synthetic throughput is not a corpus training estimate."}
