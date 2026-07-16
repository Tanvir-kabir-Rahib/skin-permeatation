from __future__ import annotations

import importlib


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency required by a pipeline stage is unavailable."""


def require_module(module_name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = install_hint or f"Install the missing dependency '{module_name}'."
        raise MissingDependencyError(hint) from exc
