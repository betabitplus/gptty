from __future__ import annotations

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

    def _build_sdk_client(self) -> ChatGPTWebClientProtocol:
        return _ProductRuntimeClient(auth_file=self.auth_file, timeout=self.timeout)

    def send(self, prompt: str, **options: Any) -> Any:
        return self._client.send(prompt, **_sdk_send_options(options))

    def send_to_conversation(
        self,
        url_or_id: Any,
        prompt: str,
        **options: Any,
    ) -> Any:
        return self._client.send_to_conversation(
            url_or_id,
            prompt,
            **_sdk_send_options(options),
        )

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


def _sdk_send_options(options: dict[str, Any]) -> dict[str, Any]:
    sdk_options = dict(options)
    sdk_options.pop("stream", None)
    if not sdk_options.get("model") and not sdk_options.get("model_profile"):
        sdk_options["model_profile"] = DEFAULT_MODEL_PROFILE
    return sdk_options


def _runtime_send_options(options: dict[str, Any]) -> dict[str, Any]:
    runtime_options = dict(options)
    runtime_options.pop("stream", None)
    return runtime_options
