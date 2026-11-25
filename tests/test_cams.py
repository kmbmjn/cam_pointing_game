import unittest
from copy import deepcopy
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torchvision.models import convnext_tiny

from cam_pointing_game import TimmClassifier, extended_cam, grad_cam, grad_cam_pp
from cam_pointing_game.cams import class_grid_cams, grid_cam, maxima_points, upsample
from cam_pointing_game.erf import fit_gaussian, measure_erf
from cam_pointing_game.models import (
    CONVNEXTV2_WEIGHTS,
    DINOV3_WEIGHTS,
    MAMBAOUT_BASE_WEIGHTS,
    MAMBAOUT_WEIGHTS,
    MOBILENET_WEIGHTS,
    SIGLIP2_WEIGHTS,
    SpatialClassifier,
    build_model,
    classify_features,
    layer3_features,
    target_features,
)


class CAMTests(unittest.TestCase):
    def test_single_image_keeps_each_class_cue_separate(self):
        model = nn.Module()
        model.conv1 = model.bn1 = model.relu = model.maxpool = nn.Identity()
        model.layer1 = model.layer2 = model.layer3 = model.layer4 = nn.Identity()
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.fc.weight.copy_(torch.eye(2))
        model.eval()

        image = torch.tensor([[[[1.0, 3.0]], [[4.0, 2.0]]]])
        with patch("cam_pointing_game.cams.classify_features", wraps=classify_features) as head:
            cams = class_grid_cams(
                model, image, torch.tensor([0, 0]), torch.tensor([1, 0]), ("extendedcam",),
            )
        self.assertEqual([call.args[1].shape[0] for call in head.call_args_list], [1, 1])
        expected = image[0, [1, 0]] / 2
        torch.testing.assert_close(cams["extendedcam"], expected)

    def test_extended_cam_keeps_spatial_gradients(self):
        activations = torch.tensor([[[[2.0, 3.0]], [[4.0, 5.0]]]])
        gradients = torch.tensor([[[[1.0, -4.0]], [[2.0, 1.0]]]])
        torch.testing.assert_close(grid_cam("extendedcam", activations, gradients),
                                   torch.tensor([[[10.0, 0.0]]]))
        torch.testing.assert_close(grid_cam("gradcam", activations, gradients),
                                   torch.tensor([[[3.0, 3.0]]]))

    def test_gradcampp_stays_finite_when_epsilon_cancels(self):
        gradient = -2.0 ** -17
        epsilon = torch.tensor(1e-7).item()
        activation = (2 * gradient ** 2 + epsilon) / abs(gradient ** 3)
        gradients = torch.tensor([[[[gradient]], [[0.1]]]])
        activations = torch.tensor([[[[activation]], [[1.0]]]])
        denominator = 2 * gradients.square() + activations * gradients.pow(3)
        self.assertEqual((denominator + epsilon)[0, 0].item(), 0)
        cam = grid_cam("gradcampp", activations, gradients)
        torch.testing.assert_close(cam, torch.tensor([[[0.04761882]]]))

    def test_gradcampp_sums_activation_times_cubed_gradient(self):
        activations = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
        gradients = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
        expected_weight = 1 / (19 + 1e-7) + 8 / (25 + 1e-7)
        torch.testing.assert_close(grid_cam("gradcampp", activations, gradients),
                                   expected_weight * activations[:, 0])

    def test_equation19_uses_the_local_cubed_gradient(self):
        activations = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
        gradients = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
        expected_weight = 1 / 5 + 8 / 32
        actual = grid_cam("gradcampp", activations, gradients, gradcampp_formula="equation19")
        torch.testing.assert_close(actual, expected_weight * activations[:, 0])

    def test_equation19_preserves_weights_for_small_gradients(self):
        activations = torch.tensor([[[[3.0, 1.0]], [[2.0, -4.0]]]], dtype=torch.float64)
        gradients = torch.tensor([[[[1e-8, 0.0]], [[1.0, 0.0]]]], dtype=torch.float64)
        expected = (1e-8 / (2 + 4e-8)) * activations[:, 0]
        cam = grid_cam("gradcampp", activations, gradients, gradcampp_formula="equation19")
        torch.testing.assert_close(cam, expected, atol=0, rtol=1e-12)

    def test_convnext_split_matches_torchvision(self):
        reference = convnext_tiny(weights=None)
        with patch("cam_pointing_game.models.get_model", return_value=reference):
            model = build_model("convnext_tiny")
        self.assert_model_split(model, reference, reference.features[5])

    @unittest.skipUnless(find_spec("timm"), "MobileNetV4 requires the mobilenet extra")
    def test_mobilenet_split_matches_timm(self):
        from timm import create_model

        reference = create_model(MOBILENET_WEIGHTS, pretrained=False, num_classes=20)
        with patch("timm.create_model", return_value=reference):
            model = build_model("mobilenetv4_conv_medium")
        self.assert_model_split(model, reference, reference.blocks[2])

    @unittest.skipUnless(find_spec("timm"), "DINOv3 requires the timm extra")
    def test_dinov3_convnext_split_matches_timm(self):
        from timm import create_model

        reference = create_model(DINOV3_WEIGHTS, pretrained=False, num_classes=20)
        with patch("timm.create_model", return_value=reference):
            model = build_model("dinov3_convnext_tiny")
        self.assert_model_split(model, reference, reference.stages[2])

    @unittest.skipUnless(find_spec("timm"), "MambaOut requires the timm extra")
    def test_mambaout_split_matches_timm(self):
        from timm import create_model

        for name, weights in (("mambaout_small", MAMBAOUT_WEIGHTS), ("mambaout_base", MAMBAOUT_BASE_WEIGHTS)):
            with self.subTest(model=name):
                reference = create_model(weights, pretrained=False, num_classes=20)
                with patch("timm.create_model", return_value=reference):
                    model = build_model(name)
                self.assert_model_split(model, reference, reference.stages[2], channels_last=True)

    @unittest.skipUnless(find_spec("timm"), "ConvNeXt V2 requires the timm extra")
    def test_convnextv2_split_matches_timm(self):
        from timm import create_model

        reference = create_model(CONVNEXTV2_WEIGHTS, pretrained=False, num_classes=20)
        with patch("timm.create_model", return_value=reference):
            model = build_model("convnextv2_tiny")
        self.assert_model_split(model, reference, reference.stages[2])

    @unittest.skipUnless(find_spec("timm"), "SigLIP 2 requires the timm extra")
    def test_siglip2_patch_grid_matches_timm(self):
        from timm import create_model

        reference = create_model(SIGLIP2_WEIGHTS, pretrained=False, num_classes=20)
        with patch("timm.create_model", return_value=reference):
            model = build_model("siglip2_vit_b16")
        self.assert_model_split(model, reference, reference.norm)

    def assert_model_split(self, model, reference, layer, *, channels_last=False):
        pixels = torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(3)) * 255
        config = getattr(reference, "pretrained_cfg", {})
        mean = torch.tensor(config.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std = torch.tensor(config.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        features = []
        model.eval()
        reference.eval()
        with layer.register_forward_hook(lambda _, __, output: features.append(output)):
            logits = reference((pixels / 255 - mean) / std)

        expected_features = features[0]
        gradient, = torch.autograd.grad(logits[0, 2], expected_features)
        if channels_last:
            expected_features = expected_features.permute(0, 3, 1, 2)
            gradient = gradient.permute(0, 3, 1, 2)
        elif expected_features.ndim == 3:
            expected_features = expected_features.transpose(1, 2).unflatten(2, (14, 14))
            gradient = gradient.transpose(1, 2).unflatten(2, (14, 14))
        self.assertEqual(expected_features.shape[-2:], (14, 14))
        expected_cam = grid_cam("extendedcam", expected_features, gradient)
        with torch.no_grad():
            torch.testing.assert_close(target_features(model, pixels), expected_features)
            torch.testing.assert_close(model(pixels), logits)
        cams = class_grid_cams(model, pixels, torch.tensor([0]), torch.tensor([2]), ("extendedcam",))
        torch.testing.assert_close(cams["extendedcam"], expected_cam)

    def test_gaussian_matches_zero_inserted_reference_including_last_cell(self):
        grid = np.random.default_rng(0).random((14, 14), dtype=np.float32)
        grid[-1, -1] = 2
        dense = np.zeros((224, 224), dtype=np.float64)
        dense[7::16, 7::16] = grid / grid.max()
        reference = gaussian_filter(dense, sigma=32.996, mode="constant")
        actual = upsample("extendedcam", torch.from_numpy(grid[None]), sigma=32.996)
        np.testing.assert_allclose(actual.numpy()[0], reference, rtol=1e-13, atol=1e-15)

    def test_rectangular_input_uses_separate_gaussian_bases(self):
        grid = torch.arange(1.0, 7.0).reshape(1, 2, 3)
        dense = np.zeros((32, 60), dtype=np.float64)
        dense[7::16, 9::20] = (grid[0] / grid.max()).numpy()
        reference = gaussian_filter(dense, sigma=4.0, mode="constant")
        actual = upsample("extendedcam", grid, sigma=4.0, output_size=(32, 60))
        np.testing.assert_allclose(actual[0], reference, rtol=1e-13, atol=1e-15)

    def test_one_line_extended_cam_fits_sigma_at_input_size(self):
        encoder = nn.Conv2d(3, 1, 17, padding=8, stride=4, bias=False)
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2, bias=False))
        model = SpatialClassifier(encoder, head, (0, 0, 0), (1, 1, 1)).eval()
        axis = torch.arange(-8, 9).square()
        with torch.no_grad():
            encoder.weight.copy_(torch.exp(-(axis[:, None] + axis) / 8).expand_as(encoder.weight))
            head[-1].weight.fill_(1)
        with torch.inference_mode():
            images = torch.rand((2, 3, 32, 48), generator=torch.Generator().manual_seed(7)) * 255
            maps, fit = extended_cam(model, images, [0, 1], calibration=images)
        self.assertEqual(maps.shape, (2, 32, 48))
        self.assertAlmostEqual(fit["sigma"], 2.0, places=2)
        for cam in (grad_cam, grad_cam_pp):
            self.assertEqual(cam(model, images, 0).shape, maps.shape)
            expected = cam(model, images, 0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                torch.testing.assert_close(cam(model, images, 0), expected)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast_maps, autocast_fit = extended_cam(model, images, [0, 1])
        torch.testing.assert_close(autocast_maps, maps)
        self.assertEqual(autocast_fit, fit)
        with self.assertRaisesRegex(ValueError, "height and width"):
            extended_cam(model, images, 0, calibration=images[..., :16, :])

    @unittest.skipUnless(find_spec("timm"), "The adapter requires the timm extra")
    def test_timm_vit_keeps_cue_order_and_removes_prefix_tokens(self):
        from timm.models.vision_transformer import VisionTransformer

        backbone = VisionTransformer(
            img_size=(32, 48), embed_dim=24, depth=2, num_heads=3, num_classes=3, reg_tokens=2,
        ).eval()
        backbone.pretrained_cfg = {
            "input_size": (3, 32, 48), "mean": (0.5, 0.4, 0.3), "std": (0.6, 0.5, 0.4),
            "interpolation": "bicubic",
        }
        model = TimmClassifier(backbone)
        self.assertEqual(model.input_spec.size, (32, 48))
        self.assertEqual(model.input_spec.interpolation, "bicubic")
        self.assertEqual(model.target_layer, "blocks.0")
        pixels = torch.rand((2, 3, 32, 48), generator=torch.Generator().manual_seed(4)) * 255
        rows = torch.tensor([1, 0, 1, 0])
        categories = torch.tensor([0, 2, 1, 0])
        actual = class_grid_cams(model, pixels, rows, categories, ("extendedcam",))["extendedcam"]

        logits, raw = model.forward_with_target(pixels)
        mean = torch.tensor((0.5, 0.4, 0.3)).view(1, 3, 1, 1)
        std = torch.tensor((0.6, 0.5, 0.4)).view(1, 3, 1, 1)
        with torch.no_grad():
            torch.testing.assert_close(logits, backbone((pixels / 255 - mean) / std))
        patch_features = raw[:, 3:].transpose(1, 2).unflatten(2, (2, 3))
        expected = []
        for row, category in zip(rows, categories, strict=True):
            gradient, = torch.autograd.grad(logits[row, category], raw, retain_graph=True)
            patch_gradients = gradient[:, 3:].transpose(1, 2).unflatten(2, (2, 3))
            expected.append(grid_cam("extendedcam", patch_features, patch_gradients)[row])
        torch.testing.assert_close(actual, torch.stack(expected))

        with torch.inference_mode():
            raw_pixels = pixels.clone()
            targets = torch.tensor([0, 1])
            for cam in (grad_cam, grad_cam_pp):
                torch.testing.assert_close(cam(backbone, raw_pixels, targets), cam(model, pixels, [0, 1]))
            maps, fit = extended_cam(backbone, raw_pixels, targets, sigma=3.0)
        torch.testing.assert_close(maps, extended_cam(model, pixels, [0, 1], sigma=3.0)[0])
        self.assertEqual(fit["sigma"], 3.0)
        with self.assertRaisesRegex(ValueError, "class-token head"):
            grad_cam(backbone, pixels, 0, target_layer="norm")
        backbone.reset_classifier(0)
        with self.assertRaisesRegex(ValueError, "require a timm classification head"):
            grad_cam(backbone, pixels, 0)

    @unittest.skipUnless(find_spec("timm"), "The adapter requires the timm extra")
    def test_timm_dynamic_vit_uses_the_supplied_patch_grid(self):
        from timm.models.vision_transformer import VisionTransformer

        backbone = VisionTransformer(
            img_size=32, dynamic_img_size=True, embed_dim=24, depth=2, num_heads=3, num_classes=3,
        ).eval()
        model = TimmClassifier(backbone)
        pixels = torch.rand((1, 3, 32, 48)) * 255
        with torch.no_grad():
            self.assertEqual(target_features(model, pixels).shape, (1, 24, 2, 3))
        self.assertEqual(grad_cam(model, pixels, 0).shape, (1, 32, 48))
        with self.assertRaisesRegex(ValueError, "RGB"):
            model(pixels[:, :1])

    @unittest.skipUnless(find_spec("timm"), "The adapter requires the timm extra")
    def test_timm_swin_uses_native_size_and_channels_last_stage(self):
        from timm import create_model

        backbone = create_model(
            "swinv2_tiny_window8_256.ms_in1k", pretrained=False, img_size=64, num_classes=3,
        ).eval()
        model = TimmClassifier(backbone)
        self.assertEqual(model.input_spec.size, (64, 64))
        self.assertEqual(model.target_layer, "layers.2")
        pixels = torch.rand((1, 3, 64, 64)) * 255
        with torch.no_grad():
            _, raw = model.forward_with_target(pixels)
        self.assertEqual(raw.shape, (1, 4, 4, 384))
        torch.testing.assert_close(model.spatial(raw), raw.permute(0, 3, 1, 2))
        self.assertEqual(grad_cam(model, pixels, 0).shape, (1, 64, 64))

    @unittest.skipUnless(find_spec("timm"), "The adapter requires the timm extra")
    def test_timm_hiera_uses_spatial_order_for_activations_and_gradients(self):
        from timm.models.hiera import Hiera

        backbone = Hiera(img_size=(64, 96), embed_dim=8, num_heads=1, stages=(1, 1, 2, 1), num_classes=3).eval()
        backbone.pretrained_cfg = {"input_size": (3, 64, 96)}
        model = TimmClassifier(backbone)
        self.assertEqual(model.target_layer, "blocks.3")
        pixels = torch.rand((1, 3, 64, 96)) * 255
        inputs = (pixels / 255 - model.mean) / model.std
        logits, tokens = model.forward_with_target(pixels)
        features, = backbone.forward_intermediates(inputs, indices=[2], intermediates_only=True)
        self.assertEqual(features.shape[-2:], (4, 6))
        torch.testing.assert_close(model.spatial(tokens), features)
        gradients, = torch.autograd.grad(logits[0, 1], tokens)
        spatial_gradients = backbone.reroll(gradients, 3).permute(0, 3, 1, 2)
        expected = grid_cam("extendedcam", features, spatial_gradients)
        cams = class_grid_cams(model, pixels, torch.tensor([0]), torch.tensor([1]), ("extendedcam",))
        torch.testing.assert_close(cams["extendedcam"], expected)

    def test_row_major_ties_and_zero_cam(self):
        maps = torch.tensor([[[1.0, 3.0], [3.0, 3.0]], [[0.0, 0.0], [0.0, 0.0]]])
        self.assertEqual(maxima_points(maps), [(0, 1), (0, 0)])
        flat = upsample("extendedcam", torch.zeros((1, 14, 14)), sigma=30.0)
        self.assertEqual(maxima_points(flat), [(0, 0)])

    def test_effective_receptive_field_fit(self):
        y, x = np.mgrid[:224, :224]
        heatmap = np.exp(-0.5 * (((x - 105) / 19) ** 2 + ((y - 123) / 27) ** 2))
        fit = fit_gaussian(heatmap)
        self.assertAlmostEqual(fit["sigma_x"], 19, places=6)
        self.assertAlmostEqual(fit["sigma_y"], 27, places=6)
        self.assertAlmostEqual(fit["sigma"], 23, places=6)

    def test_erf_uses_batchnorm_statistics_from_each_image(self):
        model = nn.Module()
        model.conv1 = nn.Identity()
        model.bn1 = nn.BatchNorm2d(3)
        model.relu = nn.ReLU()
        model.maxpool = nn.Identity()
        model.layer1 = nn.Identity()
        model.layer2 = nn.Identity()
        model.layer3 = nn.AvgPool2d(16)
        model.eval()
        original_stats = model.bn1.running_mean.clone()
        inputs = torch.rand((2, 3, 224, 224), generator=torch.Generator().manual_seed(9))
        inputs[1] += 5
        single_image_model = deepcopy(model).train()
        expected = torch.zeros((224, 224), dtype=torch.float64)
        for image in inputs.split(1):
            image = image.detach().requires_grad_(True)
            response = layer3_features(single_image_model, image)[0, :, 7, 7].sum()
            gradient, = torch.autograd.grad(response, image)
            expected += gradient.sum(1).relu()[0].double()

        with patch("cam_pointing_game.erf.fit_gaussian", side_effect=lambda heatmap: heatmap):
            accumulated = measure_erf(model, [(inputs, torch.arange(len(inputs)))])
        np.testing.assert_allclose(accumulated, expected.numpy(), rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(model.bn1.running_mean, original_stats)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
