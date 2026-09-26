from __future__ import annotations

from pathlib import Path
from typing import Any


class MediaInputError(ValueError):
    """Raised when CLI media input is invalid before reaching the SDK."""


def collect_media_inputs(args: Any) -> list[str] | None:
    ordered = getattr(args, "_media_inputs", None)
    if ordered is not None:
        raw_items = list(ordered)
    else:
        raw_items = [
            *(("image", value) for value in (getattr(args, "image", None) or [])),
            *(("file", value) for value in (getattr(args, "file", None) or [])),
        ]

    media: list[str] = []
    for kind, raw_item in raw_items:
        normalized_kind = "file" if str(kind).strip().lower() == "file" else "image"
        option = "--file" if normalized_kind == "file" else "--image"
        item = str(raw_item).strip()
        if not item:
            raise MediaInputError(
                f"{option} requires a non-empty path, URL, or data URI"
            )
        media.append(normalize_media_input(item, kind=normalized_kind))
    return media or None


def normalize_media_input(item: str, *, kind: str = "image") -> str:
    normalized_kind = "file" if str(kind).strip().lower() == "file" else "image"
    if _is_remote_url(item) or _is_data_uri(item):
        return item

    path = Path(item).expanduser()
    if not path.exists():
        noun = "image file" if normalized_kind == "image" else "file"
        raise MediaInputError(f"{noun} does not exist: {item}")
    if not path.is_file():
        raise MediaInputError(f"{normalized_kind} path is not a file: {item}")
    return str(path)


def _is_remote_url(item: str) -> bool:
    lowered = item.lower()
    return lowered.startswith(("http://", "https://"))


def _is_data_uri(item: str) -> bool:
    return item.lower().startswith("data:")
