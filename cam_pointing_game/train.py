"""Fine-tune all layers on the official VOC 2007 train and validation splits."""

import math
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from .models import SpatialClassifier, build_model, model_input_spec, target_layer_name
from .timm_adapter import TimmClassifier
from .voc import make_loader, multilabel_targets, read_annotations


def epoch_loss(
    model: nn.Module, batches: DataLoader, targets: torch.Tensor, device: torch.device,
    optimizer: Optimizer | None = None, *, bfloat16: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_images = 0
    memory_format = (torch.channels_last if isinstance(model, (SpatialClassifier, TimmClassifier))
                     and device.type == "cuda"
                     else torch.preserve_format)
    with torch.set_grad_enabled(training):
        for images, indices in batches:
            images = images.to(device, dtype=torch.float32, non_blocking=True, memory_format=memory_format)
            labels = targets[indices].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bfloat16):
                loss = criterion(model(images), labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            num_images += len(images)
    return total_loss / num_images


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    root: Path, architecture: str, output: Path, device: torch.device, *, epochs: int,
    batch_size: int, workers: int, seed: int, target_layer: str | None = None,
) -> None:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_annotations = read_annotations(root, "train")
    val_annotations = read_annotations(root, "val")
    train_targets = multilabel_targets(train_annotations)
    val_targets = multilabel_targets(val_annotations)
    model = build_model(architecture, pretrained=True, target_layer=target_layer).to(device)
    input_spec = model_input_spec(model)
    train_batches = make_loader(
        root, train_annotations, batch_size, workers, device, shuffle=True, generator=generator,
        input_spec=input_spec,
    )
    val_batches = make_loader(root, val_annotations, batch_size, workers, device, input_spec=input_spec)

    bfloat16 = False
    if isinstance(model, (SpatialClassifier, TimmClassifier)):
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
            with torch.cuda.device(device):
                bfloat16 = torch.cuda.is_bf16_supported(including_emulation=False)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=(60, 120, 160), gamma=0.2)
    print(f"{architecture} | {device} | {optimizer.__class__.__name__} | "
          f"{'BF16 mixed precision' if bfloat16 else 'FP32'}", flush=True)
    best_loss = math.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss = epoch_loss(model, train_batches, train_targets, device, optimizer, bfloat16=bfloat16)
        scheduler.step()
        val_loss = epoch_loss(model, val_batches, val_targets, device)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite training or validation loss at epoch {epoch}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            state = {
                "architecture": architecture, "state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "seed": seed,
                "target_layer": target_layer_name(architecture, model), "input": input_spec.to_dict(),
            }
            save_checkpoint(output, state)
        print(f"Epoch {epoch:3d}/{epochs} | train BCE {train_loss:.6f} | "
              f"val BCE {val_loss:.6f}", flush=True)
    print(f"Best validation BCE {best_loss:.6f} at epoch {best_epoch} | {output}")
