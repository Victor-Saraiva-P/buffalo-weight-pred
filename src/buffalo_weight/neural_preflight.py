"""Model-aware CUDA preflight shared by mixed training commands."""

from buffalo_weight.environment_contract import RuntimeProbe
from buffalo_weight.models import CNN_MASK_MODEL, PRETRAINED_MASK_EMBEDDING_MODEL, ModelConfig
from buffalo_weight.system_setup import require_official_neural_runtime


NEURAL_TRAINING_MODELS = frozenset({CNN_MASK_MODEL, PRETRAINED_MASK_EMBEDDING_MODEL})


def require_model_configs_cuda(
    model_configs: list[ModelConfig], runtime_probe: RuntimeProbe | None = None
) -> None:
    """Gate mixed commands; for example, classical-only configurations stay CPU-safe."""
    if not any(config.model in NEURAL_TRAINING_MODELS for config in model_configs):
        return
    require_official_neural_runtime(False, runtime_probe)
