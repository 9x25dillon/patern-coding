"""CPU-first training, token-weighted validation and resumable checkpoints."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import TokenDataset, file_hash
from .model import LanguageModel, ModelConfig
from .tokenizer import ByteTokenizer


@dataclass
class TrainConfig:
    steps: int = 100
    batch_size: int = 1
    accumulation: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    eval_every: int = 25
    eval_batches: int = 16
    seed: int = 42
    threads: int = 2
    device: str = "cpu"
    precision: str = "auto"

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.accumulation, self.eval_every,
               self.eval_batches, self.threads) < 1:
            raise ValueError("Training counts must be positive")
        if self.learning_rate <= 0 or self.warmup_steps < 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.device not in ("cpu", "cuda") or self.precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError("Unsupported device or precision")


def precision_for(cfg):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu until a supported GPU is installed")
    if cfg.device == "cpu":
        if cfg.precision not in ("auto", "fp32"):
            raise ValueError("CPU development uses fp32")
        return "fp32"
    p = cfg.precision
    if p == "auto":
        p = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if p == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; select fp16")
    return p


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def collate(pairs, device):
    x, y = (torch.stack(v) for v in zip(*pairs))
    # Right padding has no supervision and cannot affect past tokens. Trim it
    # before GPU transfer/attention, while retaining all instruction-prefix tokens.
    active = (y != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if not len(active):
        raise ValueError("Batch contains no supervised targets")
    end = int(active[-1])+1
    return x[:, :end].to(device), y[:, :end].to(device)


def evaluate(model, dataset, *, batch_size=1, max_batches=16, precision="fp32"):
    prior = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss, tokens, batches = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for start in range(0, min(len(dataset), batch_size*max_batches), batch_size):
                pairs = [dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]
                x, y = collate(pairs, device)
                n = int((y != -100).sum())
                with autocast(device.type, precision):
                    loss = model(x, y)[1]
                total_loss += float(loss) * n
                tokens += n
                batches += 1
    finally:
        model.train(prior)
    mean = total_loss / tokens
    return {"loss": mean, "perplexity": math.exp(min(mean, 80)),
            "bits_per_byte_token": mean/math.log(2), "target_tokens": tokens,
            "batches": batches, "includes_eos": True}


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != 1 or payload.get("tokenizer") != ByteTokenizer.version:
        raise ValueError("Unsupported AURIC checkpoint/tokenizer")
    model = LanguageModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device), payload


def train(dataset_path, output, model_config, cfg, *, resume=None, stop_after=None, init_from=None):
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    precision = precision_for(cfg)
    if resume and init_from:
        raise ValueError("Choose resume or init_from, not both")
    data = TokenDataset(dataset_path, "train")
    val = TokenDataset(dataset_path, "val")
    if data.manifest["context"] != model_config.context:
        raise ValueError("Prepared dataset context must equal model context")
    signature = file_hash(Path(dataset_path)/"manifest.json")
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("Run directory already exists; use --resume or choose a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    model = LanguageModel(model_config).to(cfg.device)
    inherited_groups = set()
    if init_from:
        previous, previous_state = load_model(init_from)
        if asdict(previous.config) != asdict(model_config):
            raise ValueError("Initial checkpoint model configuration must match the new run")
        model.load_state_dict(previous.state_dict())
        inherited_groups.update(previous_state.get("training_groups", []))
        del previous
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    start_step, best, supervised_tokens = 0, float("inf"), 0
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload.get("format") != 1 or payload.get("tokenizer") != ByteTokenizer.version:
            raise ValueError("Unsupported checkpoint")
        if payload["data_signature"] != signature or payload["model_config"] != asdict(model_config):
            raise ValueError("Resume requires identical model and prepared dataset")
        comparable = lambda d: {k: v for k, v in d.items() if k not in ("device", "threads", "precision")}
        if comparable(payload["train_config"]) != comparable(asdict(cfg)):
            raise ValueError("Resume requires the original training configuration and total step budget")
        model.load_state_dict(payload["model"])
        inherited_groups.update(payload.get("training_groups", []))
        optimizer.load_state_dict(payload["optimizer"])
        if payload["precision"] == precision:
            scaler.load_state_dict(payload["scaler"])
        rng.set_state(payload["sample_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if cfg.device == "cuda" and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start_step, best = payload["step"], payload["best_val_loss"]
        supervised_tokens = payload["supervised_tokens"]
    if start_step >= cfg.steps:
        raise ValueError("Checkpoint already reached the configured step budget")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = {"model_config": asdict(model_config), "train_config": asdict(cfg),
              "parameters": sum(p.numel() for p in model.parameters()), "precision": precision,
              "data_signature": signature, "torch_version": str(torch.__version__)}
    report["init_from"] = str(init_from) if init_from else None
    (output/"run.json").write_text(json.dumps(report, indent=2))
    initial = evaluate(model, val, batch_size=cfg.batch_size, max_batches=cfg.eval_batches, precision=precision)
    began = time.monotonic()
    end_step = min(cfg.steps, start_step+stop_after) if stop_after else cfg.steps
    history = []
    for step in range(start_step, end_step):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Draw complete accumulation groups. No dropped final epoch remainder.
        indices = torch.randint(len(data), (cfg.accumulation, cfg.batch_size), generator=rng)
        microbatches = [[data[int(i)] for i in row] for row in indices]
        target_count = sum(int((y != -100).sum()) for pairs in microbatches for _, y in pairs)
        lr_factor = min(1.0, (step+1)/max(1, cfg.warmup_steps))
        progress = max(0, step-cfg.warmup_steps)/max(1, cfg.steps-cfg.warmup_steps)
        lr = cfg.learning_rate * lr_factor * (0.1 + 0.9 * 0.5 * (1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.0
        for pairs in microbatches:
            x, y = collate(pairs, cfg.device)
            n = int((y != -100).sum())
            with autocast(cfg.device, precision):
                loss = model(x, y)[1]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; last checkpoint remains available")
            total_loss += float(loss.detach()) * n/target_count
            scaler.scale(loss * n/target_count).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if precision != "fp16" and not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradients; last checkpoint remains available")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= old_scale
        if updated:
            supervised_tokens += target_count
        event = {"step": step+1, "train_loss": total_loss, "lr": lr,
                 "gradient_norm": float(norm), "optimizer_updated": updated,
                 "supervised_tokens": supervised_tokens}
        should_save = (step+1) % cfg.eval_every == 0 or step+1 == end_step
        if should_save:
            metrics = evaluate(model, val, batch_size=cfg.batch_size, max_batches=cfg.eval_batches, precision=precision)
            event["validation"] = metrics
            improved = metrics["loss"] < best
            best = min(best, metrics["loss"])
            state = {"format": 1, "tokenizer": ByteTokenizer.version,
                     "model_config": asdict(model_config), "train_config": asdict(cfg),
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "precision": precision,
                     "step": step+1, "best_val_loss": best, "data_signature": signature,
                     "training_groups": sorted(inherited_groups | {r["group"] for r in data.manifest["records"] if r["split"] == "train"}),
                     "supervised_tokens": supervised_tokens, "sample_rng": rng.get_state(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else []}
            atomic_save(state, output/"last.pt")
            if improved:
                atomic_save(state, output/"best.pt")
            print(json.dumps(event), flush=True)
        history.append(event)
        with (output/"metrics.jsonl").open("a") as log:
            log.write(json.dumps(event)+"\n")
    summary = {**report, "initial_validation": initial, "final_validation": history[-1]["validation"],
               "best_val_loss": best, "step": end_step, "complete": end_step == cfg.steps,
               "elapsed_seconds": time.monotonic()-began, "supervised_tokens": supervised_tokens,
               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if cfg.device == "cuda" else None}
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary
