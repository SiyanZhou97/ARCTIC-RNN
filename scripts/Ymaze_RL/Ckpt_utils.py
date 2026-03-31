import torch
from pathlib import Path

def _get_state_dict(model):
    """Handle models wrapped in DataParallel (even if not used)."""
    return getattr(model, "module", model).state_dict()

def save_ckpt(path, model, optimizer=None):
    """
    Save a training checkpoint.
    Includes model weights, optimizer state, and optional metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": _get_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "torch_version": torch.__version__,
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, optimizer=None,weights_only=False,strict=True):
    """
    Load a checkpoint.
    """
    device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=weights_only)

    missing, unexpected = getattr(model, "module", model).load_state_dict(
        ckpt["model"], strict=strict
    )

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if missing or unexpected:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)