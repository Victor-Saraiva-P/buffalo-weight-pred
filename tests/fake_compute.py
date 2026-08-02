"""Named compute fakes shared by neural execution tests."""


def fake_available_cuda() -> bool:
    """Represent an available CUDA runtime without external inspection."""

    return True


def fake_unavailable_cuda() -> bool:
    """Represent a missing CUDA runtime without external inspection."""

    return False
