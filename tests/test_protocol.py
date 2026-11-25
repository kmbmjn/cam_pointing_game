import json
import tempfile
import unittest
from argparse import Namespace
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch

from cam_pointing_game.cli import read_erf_fit, read_erf_sigma, resolve_erf
from cam_pointing_game.evaluate import evaluate
from cam_pointing_game.metric import (
    CENTER,
    PointingGame,
    is_difficult,
    is_hit,
    project_point,
    reflect_point,
)
from cam_pointing_game.models import TABLE4_CHECKPOINTS, load_model
from cam_pointing_game.timm_adapter import TimmClassifier
from cam_pointing_game.voc import (
    CLASS_INDEX,
    Annotation,
    Box,
    InputSpec,
    read_annotation,
)


class ProtocolTests(unittest.TestCase):
    def test_published_sigma_applies_only_to_the_original_checkpoint(self):
        manifest = Path(__file__).resolve().parents[1] / "checkpoints.sha256"
        digests = {Path(filename).stem: digest for digest, filename in
                   map(str.split, manifest.read_text(encoding="ascii").splitlines())}
        self.assertEqual(digests, {name: checkpoint.sha256 for name, checkpoint in TABLE4_CHECKPOINTS.items()})

        args = Namespace(model="resnet18", sigma=None, erf_fit=None, fit_erf=False,
                         erf_split=None, erf_limit=None, batch_size=16)
        device = torch.device("cpu")
        model = torch.nn.Linear(1, 1)
        original = TABLE4_CHECKPOINTS["resnet18"]
        with patch("cam_pointing_game.cli.fit_erf", return_value={"sigma": 42.0}) as fit:
            model.checkpoint_sha256 = original.sha256
            self.assertEqual(resolve_erf(args, model, device), (original.sigma, None))
            fit.assert_not_called()
            model.checkpoint_sha256 = sha256(b"newly trained resnet18").hexdigest()
            self.assertEqual(resolve_erf(args, model, device), (42.0, {"sigma": 42.0}))
            fit.assert_called_once_with(args, model, device, "val", None, 8)

    def test_evaluation_reads_only_a_valid_erf_fit_for_its_model_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "erf.json"
            fit = {"model": "resnet18", "target_layer": "layer3", "sigma": 33}
            path.write_text(json.dumps(fit), encoding="utf-8")
            self.assertEqual(read_erf_sigma(path, "resnet18"), 33.0)
            with self.assertRaisesRegex(ValueError, "input size"):
                read_erf_sigma(path, "resnet18", input_size=(256, 320))
            with self.assertRaisesRegex(ValueError, "must contain a fit for resnet50"):
                read_erf_sigma(path, "resnet50")
            path.write_text(json.dumps({**fit, "sigma": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite positive sigma"):
                read_erf_sigma(path, "resnet18")

    def test_saved_erf_fit_checks_checkpoint_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            digest = sha256(b"voc weights").hexdigest()
            spec = InputSpec(size=(256, 384), interpolation="bicubic", mean=(0.5, 0.5, 0.5))
            path = Path(directory) / "erf.json"
            fit = {
                "model": "timm:example", "target_layer": "blocks.2", "sigma": 24.0,
                "input_size": list(spec.size), "input": spec.to_dict(), "checkpoint_sha256": digest,
                "split": "val", "images": 2510,
            }
            path.write_text(json.dumps(fit), encoding="utf-8")
            settings = {"layer": "blocks.2", "input_size": spec.size}
            self.assertEqual(read_erf_fit(path, "timm:example", **settings, spec=spec, digest=digest), fit)
            with self.assertRaisesRegex(ValueError, "different input preprocessing"):
                read_erf_fit(path, "timm:example", **settings, spec=InputSpec(size=spec.size), digest=digest)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                read_erf_fit(path, "timm:example", **settings, spec=spec, digest=sha256(b"other weights").hexdigest())

    def test_checkpoint_restores_saved_timm_target_and_checks_input(self):
        model = torch.nn.Linear(1, 1)
        model.input_spec = InputSpec(size=(256, 384))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voc.pth"
            torch.save({"state_dict": model.state_dict(), "architecture": "timm:example",
                        "target_layer": "blocks.2", "input": model.input_spec.to_dict(),
                        "epoch": 4, "val_loss": 0.1, "seed": 0}, path)
            with patch("cam_pointing_game.models.build_model", return_value=model) as builder:
                loaded = load_model("timm:example", path, torch.device("cpu"))
                builder.assert_called_once_with("timm:example", target_layer="blocks.2")
                self.assertEqual(loaded.checkpoint_sha256, sha256(path.read_bytes()).hexdigest())
                self.assertEqual(loaded.checkpoint_training, {"epoch": 4, "val_loss": 0.1, "seed": 0})
                model.input_spec = InputSpec(size=(384, 256))
                with self.assertRaisesRegex(ValueError, "settings saved"):
                    load_model("timm:example", path, torch.device("cpu"))

    @unittest.skipUnless(find_spec("timm"), "The adapter requires the timm extra")
    def test_loads_an_unwrapped_timm_state_dict(self):
        from timm.models.vision_transformer import VisionTransformer

        model_kwargs = {"img_size": 32, "embed_dim": 24, "depth": 2, "num_heads": 3, "num_classes": 20}
        original = VisionTransformer(**model_kwargs).eval()
        wrapper = TimmClassifier(VisionTransformer(**model_kwargs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voc.pth"
            torch.save(original.state_dict(), path)
            with patch("cam_pointing_game.models.build_model", return_value=wrapper):
                loaded = load_model("timm:example", path, torch.device("cpu"))
        inputs = torch.rand((1, 3, 32, 32)) * 255
        normalized = (inputs / 255 - loaded.mean) / loaded.std
        with torch.no_grad():
            torch.testing.assert_close(loaded(inputs), original(normalized))

    def test_odd_padding_and_part_box_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.xml"
            path.write_text("""
            <annotation>
              <size><width>7</width><height>4</height></size>
              <object>
                <name>person</name><difficult>1</difficult>
                <part><name>head</name><bndbox>
                  <xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>
                </bndbox></part>
                <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>7</xmax><ymax>4</ymax></bndbox>
              </object>
            </annotation>
            """, encoding="utf-8")
            historical = read_annotation(path)
            direct = read_annotation(path, box_source="object")
        person = CLASS_INDEX["person"]
        self.assertEqual(historical.padding, (0, 1))
        self.assertEqual(historical.mirror_margin, 32)
        self.assertEqual(historical.boxes, (Box(person, 32, 74, 64, 112),))
        self.assertEqual(direct.boxes, (Box(person, 32, 74, 224, 186),))

    def test_reflection_on_padded_axis(self):
        portrait = Annotation("portrait", 300, 400, ())
        landscape = Annotation("landscape", 400, 300, ())
        self.assertEqual(portrait.mirror_margin, 28)
        self.assertEqual(reflect_point((21, 10), portrait), (21, 46))
        self.assertEqual(reflect_point((210, 21), landscape), (182, 21))
        self.assertEqual(reflect_point((28, 21), landscape), (28, 21))
        self.assertEqual(reflect_point(CENTER, portrait), CENTER)

    def test_native_pixel_centers_project_to_the_scoring_grid(self):
        self.assertEqual(project_point((4, 210), (224, 224)), (4, 210))
        self.assertEqual(project_point((255, 383), (256, 384)), (223, 223))
        self.assertEqual(project_point((128, 192), (256, 384)), CENTER)

    def test_circle_checks_euclidean_overlap(self):
        self.assertTrue(is_hit((40, 40), (Box(0, 55, 38, 70, 45),)))
        self.assertFalse(is_hit((40, 40), (Box(0, 52, 52, 70, 70),)))

    def test_difficult_threshold_and_distractors(self):
        small = Box(0, 0, 0, 10, 10)
        just_under = Box(1, 0, 0, 111, 112)
        sample = Annotation("two-classes", 224, 224, (small, just_under))
        self.assertTrue(is_difficult(sample, 0, "all-boxes"))
        boundary = Annotation("quarter", 224, 224, (small, Box(1, 0, 0, 112, 112)))
        self.assertFalse(is_difficult(boundary, 1, "target-boxes"))
        self.assertFalse(is_difficult(boundary, 0, "all-boxes"))
        self.assertTrue(is_difficult(boundary, 0, "target-boxes"))
        same_class = Annotation("one-class", 224, 224, (small, small))
        self.assertFalse(is_difficult(same_class, 0, "all-boxes"))

    def test_macro_average_and_target_class(self):
        annotation = Annotation("multi", 224, 224, (
            Box(0, 110, 110, 120, 120), Box(0, 90, 90, 100, 100),
            Box(1, 0, 0, 10, 10),
        ))
        game = PointingGame()
        game.update(annotation, 0, CENTER)
        game.update(annotation, 1, CENTER)
        game.update(annotation, 1, (0, 0))
        result = game.results()
        self.assertEqual(result["all"]["score_percent"], 75)
        self.assertEqual(result["difficult"]["score_percent"], 75)
        self.assertEqual(result["all"]["hits"], 2)
        self.assertIsNone(result["all"]["per_class"]["bird"]["score_percent"])

    def test_evaluation_rejects_incomplete_or_duplicate_image_batches(self):
        annotations = [Annotation(name, 224, 224, ()) for name in ("first", "second")]
        model = torch.nn.Linear(1, 1).eval()
        first = (torch.empty((1, 3, 224, 224)), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "Received 1 of 2 evaluation images"):
            evaluate(annotations, ("gradcam",), model=model, batches=[first])
        with self.assertRaisesRegex(ValueError, "Image first was evaluated twice"):
            evaluate(annotations, ("gradcam",), model=model, batches=[first, first])

    def test_ablation_uses_the_same_grids_with_swapped_interpolation(self):
        annotation = Annotation("sample", 224, 224, (
            Box(0, 52, 52, 58, 58), Box(0, 164, 52, 170, 58),
        ))
        grids = {
            "gradcam": torch.tensor([[[1.0, 0], [0, 0]]]),
            "gradcampp": torch.tensor([[[0, 0], [1.0, 0]]]),
            "extendedcam": torch.tensor([[[0, 1.0], [0, 0]]]),
        }
        model = torch.nn.Linear(1, 1).eval()
        batches = [(torch.zeros((1, 3, 224, 224)), torch.tensor([0]))]
        methods = ("gradcam", "gradcampp", "extendedcam",
                   "gradcam-gaussian", "gradcampp-gaussian", "extendedcam-bilinear")
        with patch("cam_pointing_game.evaluate.class_grid_cams", return_value=grids) as compute:
            result = evaluate([annotation], methods, model=model, batches=batches, sigma=8.0)
        self.assertEqual(compute.call_args.args[-1], ("gradcam", "gradcampp", "extendedcam"))
        self.assertEqual([result[method]["all"]["hits"] for method in methods], [0, 0, 1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
