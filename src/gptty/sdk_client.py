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

    def list_recent_conversations(self, *, limit: int = 100) -> Any: ...

    def list_models(self) -> Any: ...

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any: ...

    def conversation_follow_snapshot(self, url_or_id: Any, **options: Any) -> Any: ...

    def conversation_follow_stream(self, url_or_id: Any, **options: Any) -> Any: ...

    def stop_generation(self, url_or_id: Any = None, **options: Any) -> Any: ...

    def send_temporary(self, prompt: str, **options: Any) -> Any: ...

    def end_temporary_chat(self) -> bool: ...

    def temporary_lifecycle_snapshot(self) -> dict[str, Any]: ...


class _ProductRuntimeClient:
    """Compatibility adapter from gptty's CLI-shaped SDK surface to CWA 0.3."""

    def __init__(
        self,
        *,
        auth_file: str | Path,
        timeout: int,
        browser_authority_backend: str | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.auth_file = Path(auth_file)
        self.timeout = int(timeout)
        self.browser_authority_backend = browser_authority_backend
        self.runtime = runtime or self._build_runtime()

    def _build_runtime(self) -> Any:
        from chatgpt_web_adapter import assemble_product_runtime

        runtime_options: dict[str, Any] = {
            "transport": "browser-owned",
            "auth_file": self.auth_file,
            "client_timeout": self.timeout,
        }
        from chatgpt_web_adapter.browser_authority_backend import (
            resolve_browser_authority_backend,
        )

        resolved_backend = resolve_browser_authority_backend(
            self.browser_authority_backend
        )
        if resolved_backend == "wkwebview":
            from chatgpt_web_adapter.wkwebview_provider import WKWebViewTurnProvider

            # Keep WK transport ownership in CWA. In particular, Temporary Chat
            # must use CWA's lifecycle lease, protected-write proxy, and finality
            # fence instead of a second gptty-specific implementation.
            runtime_options["provider"] = WKWebViewTurnProvider()
            runtime_options["browser_authority_policy"] = "TURN_SCOPED"
        else:
            runtime_options["browser_authority_backend"] = resolved_backend
        return assemble_product_runtime(**runtime_options)

    def send(self, prompt: str, **options: Any) -> Any:
        return self._send_normal(prompt, conversation=None, options=options)

    def _send_normal(
        self,
        prompt: str,
        *,
        conversation: Any,
        options: dict[str, Any],
    ) -> Any:
        runtime_options = _runtime_send_options(options)
        timeout = float(runtime_options.pop("timeout", self.timeout))
        submit = getattr(self.runtime, "submit", None)
        await_final = getattr(self.runtime, "await_final", None)
        explicit_model = runtime_options.get("model")
        live_observation_requested = callable(runtime_options.get("on_event")) or callable(
            runtime_options.get("on_token")
        )
        if (
            callable(submit)
            and callable(await_final)
            and not explicit_model
            and not live_observation_requested
        ):
            submit_kwargs = dict(runtime_options)
            if conversation is not None:
                submit_kwargs["conversation"] = conversation
            submission = submit(prompt, timeout=timeout, **submit_kwargs)
            return await_final(submission)
        send_kwargs = dict(runtime_options)
        if conversation is not None:
            send_kwargs["conversation"] = conversation
        return self.runtime.send(prompt, timeout=timeout, **send_kwargs)

    def send_temporary(self, prompt: str, **options: Any) -> Any:
        runtime_options = _runtime_send_options(options)
        timeout = float(runtime_options.pop("timeout", self.timeout))
        return self.runtime.send(
            prompt,
            timeout=timeout,
            conversation_mode="temporary",
            **runtime_options,
        )

    def send_to_conversation(
        self,
        url_or_id: Any,
        prompt: str,
        **options: Any,
    ) -> Any:
        return self._send_normal(prompt, conversation=url_or_id, options=options)

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

    def list_recent_conversations(self, *, limit: int = 100) -> Any:
        helper = getattr(self.runtime, "list_recent_conversations", None)
        if callable(helper):
            return helper(limit=limit)
        return list(self.runtime.list_conversations())[:limit]

    def list_models(self) -> Any:
        return self.runtime.list_models()

    def media_default_model_profile(self) -> str | None:
        write_transport = getattr(self.runtime, "write_transport", None)
        governance = getattr(write_transport, "governance", None)
        if not callable(governance):
            return None
        metadata = governance()
        if not isinstance(metadata, dict):
            return None
        if metadata.get("media_semantic_default_model_profile_supported") is True:
            return DEFAULT_MODEL_PROFILE
        return None

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        return self.runtime.conversation_snapshot(url_or_id, **options)

    def conversation_follow_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        helper = getattr(self.runtime, "conversation_follow_snapshot", None)
        if callable(helper):
            return helper(url_or_id, **options)
        return self.runtime.conversation_snapshot(url_or_id)

    def conversation_follow_stream(self, url_or_id: Any, **options: Any) -> Any:
        helper = getattr(self.runtime, "conversation_follow_stream", None)
        if not callable(helper):
            raise RuntimeError("live topic follow is unavailable")
        return helper(url_or_id, **options)

    def stop_generation(self, url_or_id: Any = None, **options: Any) -> Any:
        return self.runtime.stop_generation(url_or_id, **options)

    def end_temporary_chat(self) -> bool:
        return bool(self.runtime.end_temporary_chat())

    def temporary_lifecycle_snapshot(self) -> dict[str, Any]:
        return dict(self.runtime.temporary_lifecycle_snapshot())

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
        browser_authority_backend: str | None = None,
        sdk_client: ChatGPTWebClientProtocol | None = None,
    ) -> None:
        self.auth_file = Path(auth_file)
        self.timeout = int(timeout)
        self.browser_authority_backend = browser_authority_backend
        from chatgpt_web_adapter.browser_authority_backend import (
            resolve_browser_authority_backend,
        )

        self.effective_browser_authority_backend = (
            resolve_browser_authority_backend(browser_authority_backend)
        )
        self._client = sdk_client or self._build_sdk_client()
        self._media_default_model: str | None = None

    def _build_sdk_client(self) -> ChatGPTWebClientProtocol:
        return _ProductRuntimeClient(
            auth_file=self.auth_file,
            timeout=self.timeout,
            browser_authority_backend=self.browser_authority_backend,
        )

    def send(self, prompt: str, **options: Any) -> Any:
        return self._client.send(prompt, **self._prepare_send_options(options))

    def send_temporary(self, prompt: str, **options: Any) -> Any:
        return self._client.send_temporary(prompt, **self._prepare_send_options(options))

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
        prepared = dict(options)
        media = prepared.get("media")
        has_explicit_model = bool(prepared.get("model") or prepared.get("model_profile"))
        media_default_model: str | None = None
        if media and not has_explicit_model:
            semantic_default = getattr(self._client, "media_default_model_profile", None)
            profile = semantic_default() if callable(semantic_default) else None
            if isinstance(profile, str) and profile.strip():
                prepared["model_profile"] = profile.strip()
            else:
                if self._media_default_model is None:
                    self._media_default_model = _resolve_latest_frontier_model(self._client.list_models())
                media_default_model = self._media_default_model
        return _sdk_send_options(prepared, media_default_model=media_default_model)

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

    def list_recent_conversations(self, *, limit: int = 100) -> Any:
        helper = getattr(self._client, "list_recent_conversations", None)
        if callable(helper):
            return helper(limit=limit)
        return list(self._client.list_conversations())[:limit]

    def list_models(self) -> Any:
        return self._client.list_models()

    def conversation_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        return self._client.conversation_snapshot(url_or_id, **options)

    def conversation_follow_snapshot(self, url_or_id: Any, **options: Any) -> Any:
        helper = getattr(self._client, "conversation_follow_snapshot", None)
        if callable(helper):
            return helper(url_or_id, **options)
        return self._client.conversation_snapshot(url_or_id)

    def conversation_follow_stream(self, url_or_id: Any, **options: Any) -> Any:
        helper = getattr(self._client, "conversation_follow_stream", None)
        if not callable(helper):
            raise RuntimeError("live topic follow is unavailable")
        return helper(url_or_id, **options)

    def stop_generation(self, url_or_id: Any = None, **options: Any) -> Any:
        return self._client.stop_generation(url_or_id, **options)

    def end_temporary_chat(self) -> bool:
        return bool(self._client.end_temporary_chat())

    def temporary_lifecycle_snapshot(self) -> dict[str, Any]:
        return dict(self._client.temporary_lifecycle_snapshot())

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
