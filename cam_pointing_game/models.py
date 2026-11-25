"""Build VOC classifiers with their input preprocessing and CAM targets."""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import torch
from torch import nn
from torchvision.models import get_model, get_model_weights
from torchvision.models.resnet import ResNet
from torchvision.ops.misc import Permute

from .timm_adapter import TimmClassifier, create_timm_model
from .voc import TABLE4_INPUT, VOC_CLASSES, InputSpec


@dataclass(frozen=True)
class Table4Checkpoint:
    sigma: float
    sha256: str


# The reported ERF belongs to these weights and the Table 4 preprocessing.
TABLE4_CHECKPOINTS = {
    "resnet18": Table4Checkpoint(
        32.9960, "1356b663fe4531e1f8e76b5cfb4294daed2f245276949cc7ef6c29c546fba407",
    ),
    "resnet50": Table4Checkpoint(
        45.6683, "e749bfa59d69c2d44f157f24e6b238f690b1dae1713f2090da1b15ed14efa237",
    ),
    "resnet101": Table4Checkpoint(
        66.0161, "cc347223a8d4529212c1f4600faaeb636918cf6fb47f782091acd2f4b99f5ec2",
    ),
    "wide_resnet101_2": Table4Checkpoint(
        48.1002, "341872b2e9bff4f028a60530be20c03a9bdc0ab78cf4a26a8ac44de46cbb5911",
    ),
    "resnet152": Table4Checkpoint(
        73.9119, "9cf29b9f73d39c1c7c07211b2ca86e2bfd2ce56682a8d9e38c216aa089dfad0c",
    ),
    "resnext101_32x8d": Table4Checkpoint(
        55.6377, "213d039bfc5840e068e5631cffe433360ae0cff5d4c9af804a28772639f03914",
    ),
}
TABLE4_SIGMAS = {name: checkpoint.sigma for name, checkpoint in TABLE4_CHECKPOINTS.items()}
EXTENSION_LAYERS = {
    "convnext_tiny": "features.5",
    "mobilenetv4_conv_medium": "blocks.2",
    "dinov3_convnext_tiny": "stages.2",
    "mambaout_small": "stages.2",
    "mambaout_base": "stages.2",
    "convnextv2_tiny": "stages.2",
    "siglip2_vit_b16": "norm",
}
MODEL_NAMES = (*TABLE4_SIGMAS, *EXTENSION_LAYERS)
TIMM_EXTENSION_NAMES = EXTENSION_LAYERS.keys() - {"convnext_tiny"}
MOBILENET_WEIGHTS = "mobilenetv4_conv_medium.e500_r224_in1k"
DINOV3_WEIGHTS = "convnext_tiny.dinov3_lvd1689m"
MAMBAOUT_WEIGHTS = "mambaout_small.in1k"
MAMBAOUT_BASE_WEIGHTS = "mambaout_base.in1k"
CONVNEXTV2_WEIGHTS = "convnextv2_tiny.fcmae_ft_in22k_in1k"
SIGLIP2_WEIGHTS = "vit_base_patch16_siglip_224.v2_webli"


class SpatialClassifier(nn.Module):
    """Split a classifier at its CAM target and normalize the VOC RGB pixels."""

    def __init__(
        self, encoder: nn.Module, head: nn.Module,
        mean: tuple[float, ...] | list[float], std: tuple[float, ...] | list[float],
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.input_spec = InputSpec(mean=tuple(mean), std=tuple(std))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder((pixels / 255 - self.mean) / self.std)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pixels))


class SiglipPatchEmbedding(nn.Module):
    """Embed the fixed 224 × 224 SigLIP inputs, with no prefix tokens."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.pos_drop(self.patch_embed(pixels) + self.pos_embed)


def _build_timm_model(weights: str, pretrained: bool) -> nn.Module:
    return create_timm_model(weights, pretrained=pretrained, num_classes=len(VOC_CLASSES))


def build_model(
    architecture: str, *, pretrained: bool = False, target_layer: str | None = None,
) -> nn.Module:
    if architecture.startswith("timm:"):
        backbone = _build_timm_model(architecture.removeprefix("timm:"), pretrained)
        return TimmClassifier(backbone, target_layer=target_layer)
    if target_layer is not None:
        raise ValueError("Custom CAM targets require a timm: model")
    num_classes = len(VOC_CLASSES)
    if architecture in TABLE4_SIGMAS:
        weights = get_model_weights(architecture).IMAGENET1K_V1 if pretrained else None
        model = get_model(architecture, weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "convnext_tiny":
        weights = get_model_weights(architecture).IMAGENET1K_V1
        model = get_model(architecture, weights=weights if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        preprocessing = weights.transforms()
        encoder = model.features[:6]
        head = nn.Sequential(model.features[6:], model.avgpool, model.classifier)
        return SpatialClassifier(encoder, head, preprocessing.mean, preprocessing.std)
    if architecture == "mobilenetv4_conv_medium":
        model = _build_timm_model(MOBILENET_WEIGHTS, pretrained)
        encoder = nn.Sequential(model.conv_stem, model.bn1, model.blocks[:3])
        head = nn.Sequential(
            model.blocks[3:], model.global_pool, model.conv_head, model.norm_head,
            model.act2, model.flatten, nn.Dropout(model.drop_rate), model.classifier,
        )
        return SpatialClassifier(encoder, head, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    if architecture in ("dinov3_convnext_tiny", "convnextv2_tiny"):
        weights = DINOV3_WEIGHTS if architecture == "dinov3_convnext_tiny" else CONVNEXTV2_WEIGHTS
        model = _build_timm_model(weights, pretrained)
        encoder = nn.Sequential(model.stem, model.stages[:3])
        head = nn.Sequential(model.stages[3], model.norm_pre, model.head)
        return SpatialClassifier(encoder, head, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    if architecture in ("mambaout_small", "mambaout_base"):
        weights = MAMBAOUT_WEIGHTS if architecture == "mambaout_small" else MAMBAOUT_BASE_WEIGHTS
        model = _build_timm_model(weights, pretrained)
        # MambaOut uses NHWC internally. The CAM methods expect NCHW features.
        encoder = nn.Sequential(model.stem, model.stages[:3], Permute([0, 3, 1, 2]))
        head = nn.Sequential(Permute([0, 2, 3, 1]), model.stages[3], model.head)
        return SpatialClassifier(encoder, head, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    if architecture == "siglip2_vit_b16":
        model = _build_timm_model(SIGLIP2_WEIGHTS, pretrained)
        # Attention pooling uses the final patch tokens directly. Each token can
        # contribute to the class logit through the pooling layer.
        encoder = nn.Sequential(
            SiglipPatchEmbedding(model), model.patch_drop, model.norm_pre,
            model.blocks, model.norm, Permute([0, 2, 1]),
            nn.Unflatten(2, model.patch_embed.grid_size),
        )
        head = nn.Sequential(
            nn.Flatten(2), Permute([0, 2, 1]), model.attn_pool,
            model.fc_norm, model.head_drop, model.head,
        )
        return SpatialClassifier(encoder, head, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    raise ValueError(f"Unsupported architecture {architecture!r}")


def load_model(
    architecture: str, path: Path, device: torch.device, *, target_layer: str | None = None,
) -> nn.Module:
    path = Path(path)
    digest = sha256()
    with path.open("rb") as file:
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        # Read the same inode if training replaces the checkpoint while it is being loaded.
        file.seek(0)
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if checkpoint.get("architecture", architecture) != architecture:
        raise ValueError(f"{path} was trained for {checkpoint['architecture']}, not {architecture}")
    if target_layer is None and architecture.startswith("timm:"):
        target_layer = checkpoint.get("target_layer")
    model = build_model(architecture, target_layer=target_layer)
    if "input" in checkpoint and checkpoint["input"] != model_input_spec(model).to_dict():
        raise ValueError(f"Current input preprocessing differs from the settings saved in {path}")
    weights = checkpoint.get("state_dict", checkpoint)
    if isinstance(model, TimmClassifier) and model.backbone.state_dict().keys() == weights.keys():
        # External fine-tuning can save the timm classifier before wrapping it.
        model.backbone.load_state_dict(weights)
    else:
        model.load_state_dict(weights)
    model.checkpoint_sha256 = digest.hexdigest()
    model.checkpoint_training = {}
    for key in ("epoch", "val_loss", "seed"):
        value = checkpoint.get(key)
        if isinstance(value, (int, float)) and isfinite(value):
            model.checkpoint_training[key] = value
    return model.to(device).eval()


def layer3_features(model: ResNet, images: torch.Tensor) -> torch.Tensor:
    stem = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    return model.layer3(model.layer2(model.layer1(stem)))


def classify_layer3(model: ResNet, features: torch.Tensor) -> torch.Tensor:
    return model.fc(model.avgpool(model.layer4(features)).flatten(1))


def target_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if isinstance(model, TimmClassifier):
        _, target = model.forward_with_target(images)
        return model.spatial(target, input_size=tuple(images.shape[-2:]))
    if isinstance(model, SpatialClassifier):
        return model.features(images)
    return layer3_features(model, images)


def classify_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SpatialClassifier):
        return model.head(features)
    return classify_layer3(model, features)


def model_input_spec(model: nn.Module) -> InputSpec:
    return getattr(model, "input_spec", TABLE4_INPUT)


def table4_sigma(architecture: str, checkpoint_sha256: str) -> float | None:
    checkpoint = TABLE4_CHECKPOINTS.get(architecture)
    if checkpoint and checkpoint_sha256 == checkpoint.sha256:
        return checkpoint.sigma
    return None


def target_layer_name(architecture: str, model: nn.Module) -> str:
    if isinstance(model, TimmClassifier):
        return model.target_layer
    return EXTENSION_LAYERS.get(architecture, "layer3")
