from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as distribution_version

from .sdk_client import GpttyClient

try:
    __version__ = distribution_version("gptty-web")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["GpttyClient", "__version__"]
