from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Protocol

_CANONICAL_WAIT_MIN_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_MODEL_PROFILE = "DEEP"


class ChatGPTWebClientProtocol(Protocol):
    def send(self, prompt: str, **options: Any) -> Any: ...

    def send_to_conversation(
        self,
        url_or_id: Any,
        prompt: str,
        **options: Any,
    ) -> Any: ...

    def attach_conversation(self, url_or_id: Any, **options: Any) -> Any: ...

    def get_messages(self, url_or_id: Any, **options: Any) -> Any: ...

    def get_required_action(self, url_or_id: Any, **options: Any) -> Any: ...

    def get_status(self, url_or_id: Any, **options: Any) -> Any: ...

    def wait_until_completed(self, url_or_id: Any, **options: Any) -> Any: ...

    def list_conversations(self) -> Any: ...

    def list_models(self) -> Any: ...

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any: ...


class _ProductRuntimeClient:
    """Compatibility adapter from gptty's CLI-shaped SDK surface to CWA 0.3."""

    def __init__(
        self,
        *,
        auth_file: str | Path,
        timeout: int,
        runtime: Any | None = None,
    ) -> None:
        self.auth_file = Path(auth_file)
        self.timeout = int(timeout)
        self.runtime = runtime or self._build_runtime()

    def _build_runtime(self) -> Any:
        from chatgpt_web_adapter import assemble_product_runtime

        return assemble_product_runtime(
            transport="browser-owned",
            auth_file=self.auth_file,
            client_timeout=self.timeout,
        )

    def send(self, prompt: str, **options: Any) -> Any:
        runtime_options = _runtime_send_options(options)
        timeout = float(runtime_options.pop("timeout", self.timeout))
        return self.runtime.send(prompt, timeout=timeout, **runtime_options)

    def send_to_conversation(
        self,
        url_or_id: Any,
        prompt: str,
        **options: Any,
    ) -> Any:
        runtime_options = _runtime_send_options(options)
        timeout = float(runtime_options.pop("timeout", self.timeout))
        return self.runtime.send(
            prompt,
            conversation=url_or_id,
            timeout=timeout,
            **runtime_options,
        )

    def attach_conversation(self, url_or_id: Any, **options: Any) -> Any:
        return self.runtime.attach_conversation(url_or_id, **options)

    def get_messages(self, url_or_id: Any, **options: Any) -> Any:
        return self.runtime.get_messages(url_or_id, **options)

    def get_required_action(self, url_or_id: Any, **options: Any) -> Any:
        canonical = getattr(self.runtime, "canonical", None)
        helper = getattr(canonical, "get_required_action", None)
        if not callable(helper):
            return None
        return helper(url_or_id, **options)

    def get_status(self, url_or_id: Any, **options: Any) -> Any:
        return self.runtime.get_status(url_or_id, **options)

    def list_conversations(self) -> Any:
        return self.runtime.list_conversations()

    def list_models(self) -> Any:
        return self.runtime.list_models()

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        return self.runtime.conversation_snapshot(url_or_id, **options)

    def wait_until_completed(self, url_or_id: Any, **options: Any) -> Any:
        timeout = float(options.pop("timeout", self.timeout))
        poll_interval = max(
            _CANONICAL_WAIT_MIN_POLL_INTERVAL_SECONDS,
            float(options.pop("poll_interval", _CANONICAL_WAIT_MIN_POLL_INTERVAL_SECONDS)),
        )
        if options:
            unexpected = ", ".join(sorted(options))
            raise TypeError(f"unsupported wait options: {unexpected}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        deadline = time.monotonic() + timeout
        while True:
            status = self.runtime.get_status(url_or_id)
            if getattr(status, "status", None) == "completed":
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"conversation did not complete within {timeout:g}s")
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


class GpttyClient:
    """Thin boundary between gptty commands and chatgpt-web-adapter.

    This class must stay CLI-shaped, not backend-shaped. The SDK owns web-session
    transport, payload construction, conversation parsing, and status detection.
    """

    def __init__(
        self,
        auth_file: str | Path = "auth_data.json",
        timeout: int = 90,
        *,
        sdk_client: ChatGPTWebClientProtocol | None = None,
    ) -> None:
        self.auth_file = Path(auth_file)
        self.timeout = int(timeout)
        self._client = sdk_client or self._build_sdk_client()
        self._media_default_model: str | None = None

    def _build_sdk_client(self) -> ChatGPTWebClientProtocol:
        return _ProductRuntimeClient(auth_file=self.auth_file, timeout=self.timeout)

    def send(self, prompt: str, **options: Any) -> Any:
        return self._client.send(prompt, **self._prepare_send_options(options))

    def send_to_conversation(
        self,
        url_or_id: Any,
        prompt: str,
        **options: Any,
    ) -> Any:
        return self._client.send_to_conversation(
            url_or_id,
            prompt,
            **self._prepare_send_options(options),
        )

    def _prepare_send_options(self, options: dict[str, Any]) -> dict[str, Any]:
        media = options.get("media")
        has_explicit_model = bool(options.get("model") or options.get("model_profile"))
        media_default_model: str | None = None
        if media and not has_explicit_model:
            if self._media_default_model is None:
                self._media_default_model = _resolve_latest_frontier_model(self._client.list_models())
            media_default_model = self._media_default_model
        return _sdk_send_options(options, media_default_model=media_default_model)

    def attach_conversation(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.attach_conversation(url_or_id, **options)

    def get_messages(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.get_messages(url_or_id, **options)

    def get_required_action(self, url_or_id: Any, **options: Any) -> Any:
        helper = getattr(self._client, "get_required_action", None)
        if not callable(helper):
            return None
        return helper(url_or_id, **options)

    def get_status(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.get_status(url_or_id, **options)

    def list_conversations(self) -> Any:
        return self._client.list_conversations()

    def list_models(self) -> Any:
        return self._client.list_models()

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.conversation_snapshot(url_or_id, **options)

    def wait_until_completed(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.wait_until_completed(url_or_id, **options)


def _sdk_send_options(
    options: dict[str, Any],
    *,
    media_default_model: str | None = None,
) -> dict[str, Any]:
    sdk_options = dict(options)
    sdk_options.pop("stream", None)
    if not sdk_options.get("model") and not sdk_options.get("model_profile"):
        if sdk_options.get("media"):
            if not media_default_model:
                raise RuntimeError("no compatible default image model is available")
            sdk_options["model"] = media_default_model
        else:
            sdk_options["model_profile"] = DEFAULT_MODEL_PROFILE
    return sdk_options


def _resolve_latest_frontier_model(models: Any) -> str:
    try:
        items = list(models)
    except TypeError as exc:
        raise RuntimeError("ChatGPT model catalog is unavailable") from exc

    candidates: list[tuple[str, tuple[int, tuple[int, int], int, int, int]]] = []
    for item in items:
        slug = _model_field(item, "slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        if _model_field(item, "enabled") is False or _model_field(item, "is_disabled") is True:
            continue
        if _model_field(item, "is_work_mode_model") is True or slug == "research":
            continue

        title = _model_field(item, "title")
        title_text = title if isinstance(title, str) else ""
        lowered = f"{slug} {title_text}".lower()
        is_mini = "mini" in lowered
        is_thinking = "thinking" in lowered or slug.endswith("-thinking")
        is_instant = "instant" in lowered
        max_tokens = _model_field(item, "max_tokens")
        token_score = int(max_tokens) if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool) else 0
        rank = (
            0 if is_mini else 1,
            _model_version(slug, title_text),
            1 if is_thinking else 0,
            0 if is_instant else 1,
            token_score,
        )
        candidates.append((slug, rank))

    if not candidates:
        raise RuntimeError("no compatible default image model is available")
    return max(candidates, key=lambda candidate: candidate[1])[0]


def _model_version(slug: str, title: str) -> tuple[int, int]:
    for value in (slug, title.lower()):
        match = re.search(r"gpt[- ]?(\d+)[.-](\d+)", value)
        if match:
            return int(match.group(1)), int(match.group(2))
        match = re.search(r"gpt[- ]?(\d+)", value)
        if match:
            return int(match.group(1)), 0
    return 0, 0


def _model_field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _runtime_send_options(options: dict[str, Any]) -> dict[str, Any]:
    runtime_options = dict(options)
    runtime_options.pop("stream", None)
    return runtime_options
