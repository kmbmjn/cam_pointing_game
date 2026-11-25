"""Use a timm image classifier and its pretrained data configuration for CAMs."""

from threading import get_ident

import torch
from torch import nn

from .voc import InputSpec


def _load_timm():
    try:
        import timm
    except ModuleNotFoundError as error:
        if error.name != "timm":
            raise
        raise RuntimeError("This backbone requires pip install '.[timm]'") from error
    return timm


def create_timm_model(name: str, *, pretrained: bool, num_classes: int) -> nn.Module:
    return _load_timm().create_model(name, pretrained=pretrained, num_classes=num_classes)


class TimmClassifier(nn.Module):
    """Normalize raw RGB pixels and expose a spatial target in a timm classifier."""

    def __init__(self, backbone: nn.Module, *, target_layer: str | None = None):
        super().__init__()
        self.backbone = backbone
        data = _load_timm().data
        patch_embed = getattr(backbone, "patch_embed", None)
        resolve_config = getattr(data, "resolve_input_data_config", None)
        config = data.resolve_model_data_config(backbone) if resolve_config is None else resolve_config(backbone)
        channels, height, width = config["input_size"]
        if resolve_config is None and getattr(patch_embed, "img_size", None) is not None:
            height, width = patch_embed.img_size
        if channels != 3 or len(config["mean"]) != 3 or len(config["std"]) != 3:
            raise ValueError("This adapter requires a three-channel RGB classifier")
        self.input_spec = InputSpec(
            (height, width), config["interpolation"], tuple(config["mean"]), tuple(config["std"]),
        )
        device = next(backbone.parameters()).device
        for key in ("mean", "std"):
            values = torch.tensor(config[key], dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self.register_buffer(key, values, persistent=False)

        info = getattr(backbone, "feature_info", None) or []
        if hasattr(info, "get_dicts"):
            info = info.get_dicts()
        self.patch_grid = getattr(patch_embed, "grid_size", None)
        self.prefix_tokens = getattr(backbone, "num_prefix_tokens", 0)
        self.target_layer = target_layer or self._default_layer(info)
        try:
            backbone.get_submodule(self.target_layer)
        except AttributeError as error:
            raise ValueError(f"No layer {self.target_layer!r} in the timm classifier") from error
        block_parent, _, block_index = self.target_layer.partition(".")
        self._reroll_block = None
        if (isinstance(getattr(backbone, "reroll", None), nn.Module)
                and block_parent == "blocks" and block_index.isdecimal()):
            self._reroll_block = int(block_index)
        blocks = getattr(backbone, "blocks", None)
        if (self.patch_grid is not None and getattr(backbone, "global_pool", None) == "token"
                and isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks)
                and self.target_layer in (f"blocks.{len(blocks) - 1}", "norm")):
            raise ValueError(
                f"The class-token head does not read patches at {self.target_layer!r}. "
                f"Use {self._default_layer(info)!r} or an earlier spatial target."
            )
        self.feature_channels = next(
            (entry["num_chs"] for entry in info if entry["module"] == self.target_layer), None,
        )
        self.channels_last = getattr(backbone, "output_fmt", None) == "NHWC"
        self.training = backbone.training

    def _default_layer(self, info: list[dict]) -> str:
        backbone = self.backbone
        blocks = getattr(backbone, "blocks", None)
        if self.patch_grid is not None and isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks):
            if getattr(backbone, "global_pool", None) in ("token", "token_avg"):
                # Patch tokens need the final attention block to reach the class token.
                return f"blocks.{len(blocks) - 2}" if len(blocks) >= 2 else "blocks.0.norm1"
            return f"blocks.{len(blocks) - 1}"
        if info:
            at_most_stride_16 = [entry for entry in info if entry["reduction"] <= 16]
            return (at_most_stride_16 or info[:1])[-1]["module"]
        raise ValueError("Choose a spatial CAM target with --target-layer or target_layer=")

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Expected an RGB tensor with shape (batch, 3, height, width)")
        return self.backbone((pixels / 255 - self.mean) / self.std)

    def forward_with_target(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.backbone, "num_classes", None) == 0:
            raise ValueError("CAMs require a timm classification head with at least one output")
        captured = []
        caller = get_ident()

        def capture(_module, _inputs, output):
            if get_ident() == caller:
                captured.append(output)

        with self.backbone.get_submodule(self.target_layer).register_forward_hook(capture):
            logits = self(pixels)
        if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
            raise ValueError(f"CAM target {self.target_layer!r} must return a tensor once per forward")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("The timm classifier must return a tensor of shape (batch, classes)")
        return logits, captured[0]

    def spatial(
        self, features: torch.Tensor, *, input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Convert BCHW, BHWC, or a patch-token sequence to BCHW."""
        if features.ndim == 4:
            channels_last = self.channels_last
            if self.feature_channels not in (None, features.shape[1], features.shape[-1]):
                raise ValueError(f"Unexpected channel count at CAM target {self.target_layer!r}")
            if self.feature_channels is not None and features.shape[1] != features.shape[-1]:
                channels_last = features.shape[-1] == self.feature_channels
            return features.permute(0, 3, 1, 2) if channels_last else features
        if features.ndim == 3 and self._reroll_block is not None:
            # Hiera stores intermediate tokens in pooling-window order.
            return self.backbone.reroll(features, self._reroll_block).permute(0, 3, 1, 2)
        if features.ndim == 3 and self.patch_grid is not None:
            grid = self.patch_grid
            if input_size is not None:
                grid_for_size = getattr(self.backbone.patch_embed, "dynamic_feat_size", None)
                if grid_for_size is not None:
                    grid = grid_for_size(input_size)
            height, width = grid
            patches = features[:, self.prefix_tokens:]
            if patches.shape[1] == height * width:
                return patches.transpose(1, 2).unflatten(2, (height, width))
        raise ValueError(f"CAM target {self.target_layer!r} needs a spatial map or a full patch grid")
