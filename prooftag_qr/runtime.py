from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def runtime_info() -> dict:
    packages = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    result = {"packages": packages, "cuda_available": False}
    try:
        import torch
    except ImportError:
        return result

    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_runtime"] = torch.version.cuda
    if result["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device": torch.cuda.get_device_name(0),
                "device_memory_bytes": properties.total_memory,
            }
        )
    return result
