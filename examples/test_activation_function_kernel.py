import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from kernels import has_kernel
from kernels.function import apply_function_kernel, register_function_kernel


def reference_silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if not has_kernel("kernels-community/activation"):
        print("Kernel not available for this environment, skipping.")
        return

    module_name = "activation_target_module"

    register_function_kernel(
        func_name="silu_and_mul",
        target_module=module_name,
        repo_id="kernels-community/activation",
        device=device.type,
    )

    applied = apply_function_kernel(target_module=module_name, device=device.type)
    print(f"patched: {applied}")

    x = torch.randn(128, 1024, device=device, dtype=dtype)
    out = torch.empty(128, 512, device=device, dtype=dtype)

    module_path = Path(__file__).resolve().parent / "activation_target_module.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    target_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target_module)

    y_kernel = target_module.silu_and_mul(out, x)
    y_ref = reference_silu_and_mul(torch.empty_like(out), x)

    print(f"output shape: {y_kernel.shape}")
    print(f"outputs close: {torch.allclose(y_kernel, y_ref, atol=1e-3, rtol=1e-3)}")

if __name__ == "__main__":
    main()
