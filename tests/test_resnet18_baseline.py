from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from buffalo_weight.resnet_baseline_adapter import (
    RESNET18_BASELINE_RECIPE,
    ResNet18BaselineAdapter,
)
from tests.fake_compute import fake_available_cuda
from buffalo_weight.resnet_baseline_evaluation import (
    ResNetBaselineEvaluator,
    ResNetBaselinePredictor,
    ResNetSample,
)
from buffalo_weight.resnet_mask import (
    ResNet18MaskNetwork,
    augment_binary_mask,
    load_letterboxed_mask,
)


class TinyResNet18(nn.Module):
    """Expose ResNet phase boundaries; for example, tests inspect layer4 separately."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 1)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.fc = nn.Linear(2, 1000)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.layer4(self.layer3(self.layer2(self.layer1(self.bn1(self.conv1(values))))))
        return self.fc(features.mean(dim=(2, 3)))


class TinyResNet18Builder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> nn.Module:
        """Build one seeded tiny backbone; for example, CUDA probes avoid a full ResNet."""
        self.calls += 1
        return TinyResNet18()


class RecordingPredictor:
    def predict(self, samples: tuple[ResNetSample, ...]) -> np.ndarray:
        """Return a visible fold-dependent prediction; for example, preserve row ordering."""
        return np.asarray([sample.weight_kg + sample.fold / 10 for sample in samples])


class RecordingResNetAdapter:
    def __init__(self) -> None:
        self.selection_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.refit_calls: list[tuple[tuple[str, ...], int]] = []

    def select_epoch_count(
        self, training: tuple[ResNetSample, ...], validation: tuple[ResNetSample, ...]
    ) -> int:
        """Record isolated inner partitions; for example, choose seven partial-fit epochs."""
        self.selection_calls.append((sample_ids(training), sample_ids(validation)))
        return 7

    def fit_epochs(
        self, training: tuple[ResNetSample, ...], partial_epochs: int
    ) -> ResNetBaselinePredictor:
        """Record full outer refit; for example, prove the selected count is reused."""
        self.refit_calls.append((sample_ids(training), partial_epochs))
        return RecordingPredictor()


class ResNet18BaselineTest(unittest.TestCase):
    def test_adapter_contract_and_checkpoint_use_injected_training_device(self) -> None:
        adapter, builder = tiny_training_adapter(CpuTrainingRuntime())

        assert_adapter_contract_and_checkpoint(self, adapter)

        self.assertEqual(builder.calls, 3)

    def test_two_phase_training_uses_injected_training_device(self) -> None:
        adapter, builder = tiny_training_adapter(CpuTrainingRuntime())

        assert_two_phase_training(self, adapter)

        self.assertEqual(builder.calls, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA")
    def test_cuda_adapter_updates_reloads_and_reproduces_tiny_network(self) -> None:
        adapter, builder = tiny_training_adapter(CallableCudaAvailability())

        assert_adapter_contract_and_checkpoint(self, adapter)

        self.assertEqual(builder.calls, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA")
    def test_refit_builds_fresh_weights_and_repeats_two_phase_training(self) -> None:
        adapter, builder = tiny_training_adapter(CallableCudaAvailability())

        assert_two_phase_training(self, adapter)

        self.assertEqual(builder.calls, 2)

    def test_frozen_recipe_matches_approved_protocol(self) -> None:
        recipe = RESNET18_BASELINE_RECIPE

        self.assertEqual(recipe.image_size, 224)
        self.assertEqual(recipe.warmup_epochs, 20)
        self.assertEqual(recipe.warmup_learning_rate, 0.001)
        self.assertEqual(recipe.layer4_learning_rate, 0.0001)
        self.assertEqual(recipe.head_learning_rate, 0.0005)
        self.assertEqual(recipe.batch_size, 16)
        self.assertEqual(recipe.weight_decay, 0.0001)
        self.assertEqual(recipe.max_partial_epochs, 150)
        self.assertEqual(recipe.patience, 25)
        self.assertEqual(recipe.minimum_improvement_kg, 0.1)
        self.assertEqual(recipe.gradient_clip, 5.0)
        self.assertEqual(recipe.inner_seed, 43)
        self.assertEqual(recipe.training_seed, 44)
        self.assertEqual(recipe.horizontal_flip_probability, 0.5)
        self.assertEqual(recipe.translation_fraction, 0.05)

    def test_loader_letterboxes_binary_mask_as_one_spatial_channel(self) -> None:
        pixels = np.zeros((4, 8), dtype=np.uint8)
        pixels[1:3, 2:6] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(pixels).save(path)

            loaded = load_letterboxed_mask(path, 224)

        self.assertEqual(tuple(loaded.shape), (1, 224, 224))
        self.assertEqual(set(torch.unique(loaded).tolist()), {0.0, 1.0})
        foreground_rows = torch.where(loaded[0].any(dim=1))[0]
        self.assertGreater(int(foreground_rows.min()), 0)
        self.assertLess(int(foreground_rows.max()), 223)

    def test_network_repeats_normalizes_and_freezes_only_approved_layers(self) -> None:
        backbone = TinyResNet18()
        network = ResNet18MaskNetwork(backbone)

        self.assert_approved_trainability(network, backbone)
        self.assert_imagenet_normalization(network)

    def assert_approved_trainability(
        self, network: ResNet18MaskNetwork, backbone: TinyResNet18
    ) -> None:
        network.prepare_head_warmup()
        self.assertEqual(trainable_names(network), {"backbone.fc.weight", "backbone.fc.bias"})
        network.prepare_partial_fit()
        self.assertTrue(all(
            name.startswith("backbone.layer4") or name.startswith("backbone.fc")
            for name in trainable_names(network)
        ))
        network.train()
        self.assertFalse(backbone.bn1.training)
        self.assertFalse(backbone.layer3[1].training)
        self.assertTrue(backbone.layer4[1].training)

    def assert_imagenet_normalization(self, network: ResNet18MaskNetwork) -> None:
        normalized = network.normalize_inputs(torch.zeros((1, 1, 2, 2)))
        self.assertEqual(tuple(normalized.shape), (1, 3, 2, 2))
        expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(normalized[0, :, 0, 0], expected)

    def test_translation_augmentation_never_cuts_foreground(self) -> None:
        mask = torch.zeros((1, 224, 224))
        mask[:, 10:210, 4:220] = 1
        generator = torch.Generator().manual_seed(44)

        augmented = augment_binary_mask(mask, generator, 1.0, 0.05)

        self.assertEqual(float(augmented.sum()), float(mask.sum()))
        self.assertEqual(set(torch.unique(augmented).tolist()), {0.0, 1.0})

    def test_external_folds_are_isolated_and_refit_uses_all_permitted_rows(self) -> None:
        adapter = RecordingResNetAdapter()
        samples = tiny_evaluation_samples()

        predictions = ResNetBaselineEvaluator(adapter).evaluate(samples)

        self.assertEqual(len(predictions), 50)
        self.assertEqual({prediction.file_name for prediction in predictions}, sample_ids(samples))
        self.assertEqual(len(adapter.selection_calls), 5)
        self.assertEqual(len(adapter.refit_calls), 5)
        for fold, ((selection, stopping), (refit, epochs)) in enumerate(
            zip(adapter.selection_calls, adapter.refit_calls), start=1
        ):
            reserved = {sample.file_name for sample in samples if sample.fold == fold}
            permitted = sample_ids(samples) - reserved
            self.assertFalse((set(selection) | set(stopping)) & reserved)
            self.assertEqual(set(selection) | set(stopping), permitted)
            self.assertFalse(set(selection) & set(stopping))
            self.assertEqual(set(refit), permitted)
            self.assertEqual(epochs, 7)


def trainable_names(module: nn.Module) -> set[str]:
    return {name for name, parameter in module.named_parameters() if parameter.requires_grad}


def sample_ids(samples: tuple[ResNetSample, ...]) -> set[str]:
    return {sample.file_name for sample in samples}


class CallableCudaAvailability:
    def cuda_available(self) -> bool:
        """Expose real-test CUDA availability through the project-owned seam."""
        return fake_available_cuda()

    def training_device(self) -> torch.device:
        """Select real CUDA; for example, conditional integration tests use the GPU."""
        return torch.device("cuda")


class CpuTrainingRuntime:
    def cuda_available(self) -> bool:
        """Pass the production guard; for example, adapter logic can run on CI CPUs."""
        return True

    def training_device(self) -> torch.device:
        """Select CPU; for example, tests exercise training without weakening production."""
        return torch.device("cpu")


def tiny_training_adapter(
    runtime: CallableCudaAvailability | CpuTrainingRuntime,
) -> tuple[ResNet18BaselineAdapter, TinyResNet18Builder]:
    recipe = replace(
        RESNET18_BASELINE_RECIPE, image_size=16, warmup_epochs=1,
        max_partial_epochs=1, patience=1, batch_size=2,
    )
    builder = TinyResNet18Builder()
    return ResNet18BaselineAdapter(recipe, builder, runtime), builder


def assert_adapter_contract_and_checkpoint(
    test_case: unittest.TestCase, adapter: ResNet18BaselineAdapter
) -> None:
    first = adapter.contract_probe()
    second = adapter.contract_probe()
    test_case.assertTrue(first.has_gradients)
    test_case.assertTrue(first.parameters_updated)
    test_case.assertEqual(first.predictions, second.predictions)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "resnet.pt"
        adapter.save_model(first.model, checkpoint)
        restored = adapter.load_model(checkpoint)
        test_case.assertEqual(adapter.predict_probe(first.model), adapter.predict_probe(restored))


def assert_two_phase_training(
    test_case: unittest.TestCase, adapter: ResNet18BaselineAdapter
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        samples = tiny_mask_samples(Path(directory))
        epochs = adapter.select_epoch_count(samples[:4], samples[4:])
        predictor = adapter.fit_epochs(samples[:4], epochs)
        predictions = predictor.predict(samples[4:])
    test_case.assertEqual(epochs, 1)
    test_case.assertTrue(np.isfinite(predictions).all())


def tiny_mask_samples(root: Path) -> tuple[ResNetSample, ...]:
    samples = []
    for index in range(6):
        pixels = np.zeros((8 + index, 12), dtype=np.uint8)
        pixels[1:-1, 2:10] = 255
        path = root / f"tiny-{index}.png"
        Image.fromarray(pixels).save(path)
        samples.append(ResNetSample(path.name, path, f"B{index + 1}", 1, 90.0 + index * 5))
    return tuple(samples)


def tiny_evaluation_samples() -> tuple[ResNetSample, ...]:
    return tuple(
        ResNetSample(
            f"mask-{index:03d}.png", Path(f"mask-{index:03d}.png"),
            f"B{index % 10 + 1}", index % 5 + 1, float(80 + index),
        )
        for index in range(50)
    )


if __name__ == "__main__":
    unittest.main()
