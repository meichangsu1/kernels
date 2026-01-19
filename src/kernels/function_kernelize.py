from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

from kernels._versions import select_revision_or_version
from kernels.layer.func import FuncRepositoryProtocol
from kernels.utils import get_kernel


@dataclass(frozen=True)
class FunctionKernelSpec:
    func_name: str
    target_module: str
    device: Optional[str]
    func_impl: Optional[Callable]
    repo: Optional[FuncRepositoryProtocol]
    repo_id: Optional[str]
    revision: Optional[str]
    version: Optional[str]


_FUNCTION_REGISTRY: List[FunctionKernelSpec] = []


def register_function_kernel(
    *,
    func_name: str,
    target_module: str,
    device: Optional[str] = None,
    func_impl: Optional[Callable] = None,
    repo: Optional[FuncRepositoryProtocol] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    sources = [func_impl is not None, repo is not None, repo_id is not None]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of func_impl, repo, or repo_id.")
    if revision is not None and version is not None:
        raise ValueError("Either revision or version must be specified, not both.")

    _FUNCTION_REGISTRY.append(
        FunctionKernelSpec(
            func_name=func_name,
            target_module=target_module,
            device=device,
            func_impl=func_impl,
            repo=repo,
            repo_id=repo_id,
            revision=revision,
            version=version,
        )
    )


def kernelize_functions(
    *,
    target_module: Optional[str] = None,
    device: Optional[str] = None,
    strict: bool = False,
) -> List[str]:
    applied = []
    for spec in _FUNCTION_REGISTRY:
        if target_module is not None and spec.target_module != target_module:
            continue
        if device is not None and spec.device is not None and spec.device != device:
            continue

        try:
            module = importlib.import_module(spec.target_module)
        except Exception as exc:
            if strict:
                raise
            warnings.warn(
                f"Failed to import target module {spec.target_module}: {exc}"
            )
            continue

        if not hasattr(module, spec.func_name):
            msg = (
                f"Target module {spec.target_module} has no function "
                f"{spec.func_name}."
            )
            if strict:
                raise AttributeError(msg)
            warnings.warn(msg)
            continue

        if spec.func_impl is not None:
            impl = spec.func_impl
        elif spec.repo is not None:
            module_cls = spec.repo.load()
            module_instance = module_cls()

            def impl(*args, **kwargs):
                return module_instance(*args, **kwargs)
        else:
            assert spec.repo_id is not None
            revision = select_revision_or_version(
                spec.repo_id, spec.revision, spec.version
            )
            kernel = get_kernel(spec.repo_id, revision=revision)
            impl = getattr(kernel, spec.func_name, None)
            if impl is None:
                msg = (
                    f"Kernel repo {spec.repo_id} does not export "
                    f"{spec.func_name}."
                )
                if strict:
                    raise AttributeError(msg)
                warnings.warn(msg)
                continue

        setattr(module, spec.func_name, impl)
        applied.append(f"{spec.target_module}.{spec.func_name}")

    return applied
