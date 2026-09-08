from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_client(client_factory: Callable[..., Any], args: Any) -> Any:
    """Construct the SDK client without leaking backend details into commands."""

    kwargs: dict[str, Any] = {
        "auth_file": getattr(args, "auth", "auth_data.json"),
        "timeout": getattr(args, "timeout", 90),
    }
    backend = getattr(args, "backend", None)
    if isinstance(backend, str) and backend.strip():
        kwargs["browser_authority_backend"] = backend.strip()
    return client_factory(**kwargs)


__all__ = ["build_client"]
