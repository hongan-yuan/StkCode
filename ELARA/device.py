from __future__ import annotations


def mps_is_available(torch_module) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and mps.is_built() and mps.is_available())


def resolve_torch_device_name(requested: str, torch_module) -> str:
    """Resolve auto and validate explicitly requested accelerator devices."""
    requested = str(requested).lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if mps_is_available(torch_module):
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return requested
    if requested == "mps":
        if not mps_is_available(torch_module):
            raise RuntimeError(
                "MPS was requested, but this PyTorch build or machine does not provide MPS"
            )
        return requested
    if requested == "cpu":
        return requested
    raise ValueError(f"unsupported device: {requested}; use auto, cuda, mps, or cpu")
