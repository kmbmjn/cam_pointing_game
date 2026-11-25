"""Command-line entry points for evaluation, training, validation, and ERF fitting."""

import argparse
import json
import math
import platform
import sys
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import cv2
import torch
import torchvision
from torch import nn

from .cams import CAM_ABLATIONS, GRADCAMPP_FORMULAS
from .evaluate import METHODS, evaluate
from .metric import DIFFICULT_SETS, RADIUS
from .models import (
    EXTENSION_LAYERS,
    MODEL_NAMES,
    TABLE4_SIGMAS,
    TIMM_EXTENSION_NAMES,
    load_model,
    model_input_spec,
    table4_sigma,
    target_layer_name,
)
from .voc import (
    BOX_SOURCES,
    CLASS_INDEX,
    SCORING_SIZE,
    InputSpec,
    make_loader,
    read_annotation,
    read_annotations,
)

SPLITS = ("train", "val", "test")
EXTENSION_EPOCHS = {"siglip2_vit_b16": 12, "mambaout_base": 8}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def add_data_arguments(parser: argparse.ArgumentParser, batch_size: int) -> None:
    parser.add_argument("--voc-root", type=Path, required=True, help="Path to VOCdevkit/VOC2007")
    parser.add_argument("--batch-size", type=positive_int, default=batch_size)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (0 runs in-process)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")


def model_name(name: str) -> str:
    if name in MODEL_NAMES or (name.startswith("timm:") and name.removeprefix("timm:")):
        return name
    raise argparse.ArgumentTypeError("use a Table 4 or extension name, or timm:<model.pretrained_tag>")


def add_model_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--model", type=model_name, required=required,
                        help="Table 4/extension model or timm:<model.pretrained_tag>")
    parser.add_argument("--checkpoint", type=Path, required=required, help="VOC model weights")
    parser.add_argument("--target-layer", help="Dotted CAM layer name for a timm: model")


def add_split_arguments(parser: argparse.ArgumentParser, default: str | None = "test") -> None:
    parser.add_argument("--split", choices=SPLITS, default=default)
    parser.add_argument("--limit", type=positive_int, help="Process only the first N images")


def add_cam_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--box-source", choices=BOX_SOURCES, default=BOX_SOURCES[0],
                        help="Use the Table 4 XML traversal or direct object boxes")
    parser.add_argument("--gradcampp-formula", choices=GRADCAMPP_FORMULAS,
                        help="Default archive for Table 4/extension names, equation19 for timm: models")
    gaussian = parser.add_mutually_exclusive_group()
    gaussian.add_argument("--sigma", type=positive_float, help="Extended-CAM sigma in input pixels")
    gaussian.add_argument("--erf-fit", type=Path, help="Read sigma from fit-erf JSON for this model")
    gaussian.add_argument("--fit-erf", action="store_true", help="Fit sigma before computing CAMs")
    parser.add_argument("--erf-split", choices=SPLITS, help="Calibration split for an in-command fit")
    parser.add_argument("--erf-limit", type=positive_int, help="Limit the in-command calibration split")


def make_progress(num_images: int) -> Callable[[int], None]:
    next_report = 512

    def report(processed: int) -> None:
        nonlocal next_report
        if processed >= next_report or processed == num_images:
            print(f"Processed {processed}/{num_images} images", file=sys.stderr, flush=True)
            next_report = processed + 512

    return report


def write_json(path: Path, result: dict) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_erf_fit(
    path: Path, model: str, *, layer: str | None = None, input_size: tuple[int, int] = SCORING_SIZE,
    spec: InputSpec | None = None, digest: str | None = None,
) -> dict:
    fit = json.loads(path.read_text(encoding="utf-8"))
    layer = layer or EXTENSION_LAYERS.get(model, "layer3")
    if not isinstance(fit, dict) or fit.get("model") != model or fit.get("target_layer") != layer:
        raise ValueError(f"{path} must contain a fit for {model} at {layer}")
    if fit.get("input_size", list(SCORING_SIZE)) != list(input_size):
        raise ValueError(f"{path} must contain a fit at input size {input_size}")
    if spec is not None and "input" in fit and fit["input"] != spec.to_dict():
        raise ValueError(f"{path} was fitted with different input preprocessing")
    if digest is not None and "checkpoint_sha256" in fit and fit["checkpoint_sha256"] != digest:
        raise ValueError(f"{path} was fitted with a different checkpoint (SHA-256 mismatch)")
    sigma = fit.get("sigma")
    if type(sigma) not in (int, float) or not 0 < sigma <= sys.float_info.max:
        raise ValueError(f"{path} must contain a finite positive sigma")
    return fit


def read_erf_sigma(
    path: Path, model: str, *, layer: str | None = None, input_size: tuple[int, int] = SCORING_SIZE,
) -> float:
    return float(read_erf_fit(path, model, layer=layer, input_size=input_size)["sigma"])


def model_runtime(device: torch.device, batch_size: int, architecture: str) -> dict:
    cuda = device.type == "cuda"
    runtime = {
        "device": str(device), "batch_size": batch_size,
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__), "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "opencv": cv2.__version__,
    }
    runtime.update({package: version(package) for package in ("numpy", "pillow", "scipy", "shapely")})
    if architecture.startswith("timm:") or architecture in TIMM_EXTENSION_NAMES:
        runtime["timm"] = version("timm")
    return runtime


def load_classifier(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return load_model(args.model, args.checkpoint, device, target_layer=args.target_layer)


def default_erf_split(architecture: str, checkpoint_sha256: str) -> str:
    return "test" if table4_sigma(architecture, checkpoint_sha256) is not None else "val"


def fit_erf(
    args: argparse.Namespace, model: nn.Module, device: torch.device,
    split: str, limit: int | None, batch_size: int,
) -> dict:
    from .erf import measure_erf

    annotations = read_annotations(args.voc_root, split, limit)
    spec = model_input_spec(model)
    batches = make_loader(args.voc_root, annotations, batch_size, args.workers, device, input_spec=spec)
    print(f"Fitting ERF on VOC 2007 {split} ({len(annotations)} images)", file=sys.stderr, flush=True)
    fit = measure_erf(model, batches, input_size=spec.size, progress=make_progress(len(annotations)))
    return {
        "model": args.model, "checkpoint": args.checkpoint.name, "checkpoint_sha256": model.checkpoint_sha256,
        "target_layer": target_layer_name(args.model, model),
        "input_size": list(spec.size), "input": spec.to_dict(), "split": split, "images": len(annotations),
        "runtime": model_runtime(device, batch_size, args.model), **fit,
    }


def resolve_erf(
    args: argparse.Namespace, model: nn.Module, device: torch.device,
) -> tuple[float, dict | None]:
    if args.sigma is not None:
        return args.sigma, None
    if args.erf_fit is not None:
        spec = model_input_spec(model)
        fit = read_erf_fit(
            args.erf_fit, args.model, layer=target_layer_name(args.model, model),
            input_size=spec.size, spec=spec, digest=model.checkpoint_sha256,
        )
        return float(fit["sigma"]), fit
    published_sigma = table4_sigma(args.model, model.checkpoint_sha256)
    if published_sigma is not None and not args.fit_erf:
        return published_sigma, None
    split = args.erf_split or default_erf_split(args.model, model.checkpoint_sha256)
    fit = fit_erf(args, model, device, split, args.erf_limit, min(args.batch_size, 8))
    return fit["sigma"], fit


def evaluate_command(args: argparse.Namespace, device: torch.device) -> None:
    methods = METHODS if args.method == "all" else (args.method,)
    if args.ablation:
        methods = (*methods, *CAM_ABLATIONS)
    annotations = read_annotations(args.voc_root, args.split, args.limit, box_source=args.box_source)
    model = None
    batches = None
    sigma = None
    fit = None
    if args.method != "center":
        model = load_classifier(args, device)
        batches = make_loader(
            args.voc_root, annotations, args.batch_size, args.workers, device, input_spec=model_input_spec(model),
        )
        if "extendedcam" in methods:
            sigma, fit = resolve_erf(args, model, device)

    scores = evaluate(
        annotations, methods, difficult_set=args.difficult_set, model=model,
        batches=batches, sigma=sigma, gradcampp_formula=args.gradcampp_formula,
        progress=make_progress(len(annotations)),
    )
    print(f"VOC 2007 {args.split} | {len(annotations)} images | Boxes {args.box_source} | "
          f"Difficult rule {args.difficult_set}")
    if model is not None:
        print(f"{args.model} | {device} | {args.checkpoint.name}")
    if "gradcampp" in methods:
        print(f"Grad-CAM++ formula {args.gradcampp_formula}")
    if sigma is not None:
        print(f"Extended-CAM sigma {sigma:.4f} input pixels")
    method_width = max(13, max(map(len, scores)))
    print(f"{'Method':<{method_width}} {'All (%)':>10} {'Difficult (%)':>15}")
    for method, result in scores.items():
        values = [result[subset]["score_percent"] for subset in ("all", "difficult")]
        all_text, difficult_text = (f"{score:.4f}" if score is not None else "—" for score in values)
        print(f"{method:<{method_width}} {all_text:>10} {difficult_text:>15}")
    if args.output:
        report = {
            "split": args.split, "images": len(annotations), "box_source": args.box_source,
            "difficult_set": args.difficult_set,
            "radius_pixels": RADIUS, "scoring_size": list(SCORING_SIZE),
            "model": args.model if model is not None else None,
            "checkpoint": args.checkpoint.name if model is not None else None,
            "checkpoint_sha256": model.checkpoint_sha256 if model is not None else None,
            "target_layer": target_layer_name(args.model, model) if model is not None else None,
            "input": model_input_spec(model).to_dict() if model is not None else None,
            "sigma": sigma,
            "erf_fit": fit,
            "gradcampp_formula": args.gradcampp_formula if "gradcampp" in methods else None,
            "runtime": model_runtime(device, args.batch_size, args.model) if model is not None else None,
            "methods": scores,
        }
        if model is not None and model.checkpoint_training:
            report["checkpoint_training"] = model.checkpoint_training
        write_json(args.output, report)


def train_command(args: argparse.Namespace, device: torch.device) -> None:
    from .train import train

    default_epochs = 200 if args.model in TABLE4_SIGMAS else EXTENSION_EPOCHS.get(args.model, 20)
    train(
        args.voc_root, args.model, args.output, device,
        epochs=args.epochs or default_epochs,
        batch_size=args.batch_size, workers=args.workers, seed=args.seed, target_layer=args.target_layer,
    )


def validate_command(args: argparse.Namespace, device: torch.device) -> None:
    from .train import epoch_loss
    from .voc import multilabel_targets

    annotations = read_annotations(args.voc_root, args.split, args.limit)
    model = load_classifier(args, device)
    batches = make_loader(
        args.voc_root, annotations, args.batch_size, args.workers, device, input_spec=model_input_spec(model),
    )
    loss = epoch_loss(model, batches, multilabel_targets(annotations), device)
    result = {
        "model": args.model, "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": model.checkpoint_sha256,
        "split": args.split, "images": len(annotations), "bce": loss,
        "runtime": model_runtime(device, args.batch_size, args.model),
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        write_json(args.output, result)


def erf_command(args: argparse.Namespace, device: torch.device) -> None:
    model = load_classifier(args, device)
    split = args.split or default_erf_split(args.model, model.checkpoint_sha256)
    result = fit_erf(
        args, model, device, split, args.limit, args.batch_size,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        write_json(args.output, result)


def visualize_command(args: argparse.Namespace, device: torch.device) -> None:
    from .visualize import render_pointing_game

    image_id = args.image_id
    annotation = read_annotation(args.voc_root / "Annotations" / f"{image_id}.xml", box_source=args.box_source)
    category = CLASS_INDEX[args.category]
    if category not in annotation.categories:
        raise ValueError(f"{args.category} is absent from {image_id}")
    model = load_classifier(args, device)
    sigma, _ = resolve_erf(args, model, device)
    figure = render_pointing_game(
        model, args.voc_root / "JPEGImages" / f"{image_id}.jpg", annotation, category, sigma,
        gradcampp_formula=args.gradcampp_formula,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    print(f"Saved {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extended-CAM pointing game on PASCAL VOC 2007")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluation = subparsers.add_parser("evaluate", help="Measure class-averaged pointing accuracy")
    add_data_arguments(evaluation, batch_size=16)
    add_model_arguments(evaluation, required=False)
    add_split_arguments(evaluation)
    evaluation.add_argument("--method", choices=("all", *METHODS), default="all")
    evaluation.add_argument("--ablation", action="store_true",
                            help="Evaluate both bilinear and Gaussian upsampling for each CAM weighting")
    add_cam_arguments(evaluation)
    evaluation.add_argument("--difficult-set", choices=DIFFICULT_SETS, default="all-boxes",
                            help="Area rule for Difficult (all-boxes reproduces Table 4)")
    evaluation.add_argument("--output", type=Path, help="Write scores and per-class counts to JSON")
    evaluation.set_defaults(run=evaluate_command)

    training = subparsers.add_parser("train", help="Fine-tune an ImageNet classifier on VOC")
    add_data_arguments(training, batch_size=32)
    training.add_argument("--model", type=model_name, required=True,
                          help="Table 4/extension model or timm:<model.pretrained_tag>")
    training.add_argument("--target-layer", help="Dotted CAM layer name for a timm: model")
    training.add_argument("--output", type=Path, required=True, help="Save the best-validation checkpoint")
    training.add_argument("--epochs", type=positive_int,
                          help="Defaults to 200 for Table 4, 12 for SigLIP 2, 8 for MambaOut-Base, or 20 otherwise")
    training.add_argument("--seed", type=int, default=0)
    training.set_defaults(run=train_command)

    validation = subparsers.add_parser("validate", help="Measure multi-label classification BCE")
    add_data_arguments(validation, batch_size=32)
    add_model_arguments(validation, required=True)
    add_split_arguments(validation, default="val")
    validation.add_argument("--output", type=Path, help="Write classification BCE to JSON")
    validation.set_defaults(run=validate_command)

    erf = subparsers.add_parser("fit-erf", help="Fit a Gaussian to the input gradient of the center cell")
    add_data_arguments(erf, batch_size=8)
    add_model_arguments(erf, required=True)
    add_split_arguments(erf, default=None)
    erf.add_argument("--output", type=Path, help="Write the fitted sigmas and R-squared to JSON")
    erf.set_defaults(run=erf_command)

    visualization = subparsers.add_parser("visualize", help="Render a VOC image/category cue and its hits")
    add_data_arguments(visualization, batch_size=8)
    add_model_arguments(visualization, required=True)
    add_cam_arguments(visualization)
    visualization.add_argument("--image-id", required=True, help="VOC image stem, such as 000001")
    visualization.add_argument("--category", choices=CLASS_INDEX, required=True)
    visualization.add_argument("--output", type=Path, required=True, help="Output image path")
    visualization.set_defaults(run=visualize_command)

    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.target_layer is not None and not (args.model and args.model.startswith("timm:")):
        parser.error("--target-layer requires a timm: model")
    if args.command in ("evaluate", "visualize") and args.gradcampp_formula is None:
        args.gradcampp_formula = "equation19" if args.model and args.model.startswith("timm:") else "archive"
    if args.command == "evaluate":
        if args.ablation and args.method != "all":
            evaluation.error("--ablation requires --method all")
        if args.method != "center" and not (args.model and args.checkpoint):
            evaluation.error("model-based methods require --model and --checkpoint")
        if args.method not in ("all", "extendedcam") and any(
            (args.erf_fit, args.sigma, args.fit_erf, args.erf_split, args.erf_limit)
        ):
            evaluation.error("ERF options require --method all or extendedcam")
    if args.command in ("evaluate", "visualize") and (args.erf_split or args.erf_limit):
        if args.erf_fit is not None or args.sigma is not None:
            parser.error("--erf-split and --erf-limit apply when fitting sigma")
        args.fit_erf = True

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        # Keep the benchmark in FP32 on GPUs with TensorFloat-32 support.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.command in ("evaluate", "visualize"):
            torch.backends.cudnn.deterministic = True
    try:
        args.run(args, device)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
