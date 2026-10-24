from __future__ import annotations

import argparse
import hashlib
import json
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gptty.sdk_client import GpttyClient

GPTTY = Path("/Users/stas/.local/bin/gptty")
AUTH = Path.home() / ".local/share/gptty/profiles/chatgpt-web/auth_data.json"
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")
DEFAULT_TIMEOUT = 60.0


def clean(text: str) -> str:
    return ANSI.sub("", text).replace("\r", "").replace("\x08", "")


def marker(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


def make_red_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    width = height = 32
    raw = b"".join(b"\x00" + bytes([255, 0, 0]) * width for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_sitecustomize(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sitecustomize = r"""from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

broker_dir = os.environ.get("GPTTY_LIVE_BROKER_DIR")
if broker_dir:
    from chatgpt_web_adapter.wkwebview_turn_broker import WKSystemTurnBrokerClient

    _original_init = WKSystemTurnBrokerClient.__init__

    def _isolated_init(self, helper, *, runtime_dir=None, idle_timeout=2.0):
        return _original_init(
            self,
            helper,
            runtime_dir=Path(broker_dir),
            idle_timeout=idle_timeout,
        )

    WKSystemTurnBrokerClient.__init__ = _isolated_init

phase_trace_path = os.environ.get("GPTTY_LIVE_PHASE_TRACE")


def _append_phase(record):
    if not phase_trace_path:
        return
    item = {"pid": os.getpid(), "ts": time.monotonic(), **record}
    try:
        fd = os.open(
            phase_trace_path,
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            0o600,
        )
        try:
            os.write(
                fd,
                (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8"),
            )
        finally:
            os.close(fd)
    except OSError:
        pass


try:
    from gptty.ui.renderer import PrettyRenderer

    _original_renderer_answer = PrettyRenderer.answer

    def _traced_renderer_answer(self, text):
        normalized = text if isinstance(text, str) else str(text)
        stream_state = getattr(getattr(self, "_answer_state", None), "text", "")
        if not isinstance(stream_state, str):
            stream_state = str(stream_state)
        _append_phase(
            {
                "kind": "renderer_answer",
                "length": len(normalized),
                "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "stream_length": len(stream_state),
                "stream_sha256": hashlib.sha256(stream_state.encode("utf-8")).hexdigest(),
                "stream_suppressed": bool(getattr(self, "_answer_stream_suppressed", False)),
            }
        )
        return _original_renderer_answer(self, text)

    PrettyRenderer.answer = _traced_renderer_answer
except Exception:
    pass


try:
    from chatgpt_web_adapter.wkwebview_helper_runtime import WKWebViewHelperRuntime

    _phase_original_stream = WKWebViewHelperRuntime._run_streaming_via_turn_broker

    def _traced_stream(self, *args, **kwargs):
        original_callback = kwargs.get("on_transport_event")

        def traced_callback(event):
            if isinstance(event, dict):
                _append_phase(
                    {
                        "kind": "transport",
                        "type": event.get("type"),
                        "state": event.get("state"),
                        "status": event.get("status"),
                    }
                )
            if callable(original_callback):
                original_callback(event)

        kwargs["on_transport_event"] = traced_callback
        _append_phase({"kind": "broker_enter"})
        try:
            result = _phase_original_stream(self, *args, **kwargs)
        except BaseException as exc:
            _append_phase(
                {
                    "kind": "broker_error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        else:
            _append_phase({"kind": "broker_return"})
            return result

    WKWebViewHelperRuntime._run_streaming_via_turn_broker = _traced_stream
except Exception:
    pass


try:
    import gptty.commands.chat as _gptty_chat

    _original_send_chat_prompt = _gptty_chat._send_chat_prompt

    def _traced_send_chat_prompt(*args, **kwargs):
        _append_phase(
            {
                "kind": "gptty_send_enter",
                "conversation_mode": kwargs.get("conversation_mode"),
            }
        )
        try:
            result = _original_send_chat_prompt(*args, **kwargs)
        except BaseException as exc:
            _append_phase(
                {
                    "kind": "gptty_send_error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        else:
            _append_phase({"kind": "gptty_send_return", "code": result})
            return result

    _gptty_chat._send_chat_prompt = _traced_send_chat_prompt
except Exception:
    pass

try:
    import gptty.sdk_client as _gptty_sdk

    _original_send_to_conversation = _gptty_sdk.GpttyClient.send_to_conversation

    def _traced_send_to_conversation(self, *args, **kwargs):
        _append_phase({"kind": "sdk_send_to_enter"})
        try:
            result = _original_send_to_conversation(self, *args, **kwargs)
        except BaseException as exc:
            _append_phase(
                {
                    "kind": "sdk_send_to_error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        else:
            _append_phase({"kind": "sdk_send_to_return"})
            return result

    _gptty_sdk.GpttyClient.send_to_conversation = _traced_send_to_conversation
except Exception:
    pass

try:
    from chatgpt_web_adapter.wkwebview_provider import WKWebViewTurnProvider

    _original_submit_text_streaming = WKWebViewTurnProvider.submit_text_streaming

    def _traced_submit_text_streaming(self, *args, **kwargs):
        _append_phase({"kind": "provider_submit_enter"})
        try:
            result = _original_submit_text_streaming(self, *args, **kwargs)
        except BaseException as exc:
            _append_phase(
                {
                    "kind": "provider_submit_error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        else:
            _append_phase({"kind": "provider_submit_return"})
            return result

    WKWebViewTurnProvider.submit_text_streaming = _traced_submit_text_streaming
except Exception:
    pass


trace_path = os.environ.get("GPTTY_LIVE_HTTP_TRACE")
if trace_path:
    try:
        from curl_cffi import requests as curl_requests

        _original_request = curl_requests.Session.request

        def _traced_request(self, method, url, *args, **kwargs):
            started = time.monotonic()
            rec = {
                "pid": os.getpid(),
                "ts": started,
                "method": str(method).upper(),
                "path": urlsplit(str(url)).path,
            }
            try:
                response = _original_request(self, method, url, *args, **kwargs)
                rec["status"] = getattr(response, "status_code", None)
                return response
            except Exception as exc:
                rec["error"] = type(exc).__name__
                raise
            finally:
                rec["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                try:
                    fd = os.open(
                        trace_path,
                        os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                        0o600,
                    )
                    try:
                        os.write(
                            fd,
                            (json.dumps(rec, separators=(",", ":")) + "\n").encode(
                                "utf-8"
                            ),
                        )
                    finally:
                        os.close(fd)
                except OSError:
                    pass

        curl_requests.Session.request = _traced_request
    except Exception:
        pass

topic_fault = os.environ.get("GPTTY_LIVE_TOPIC_FAULT")
if topic_fault:
    import chatgpt_web_adapter.wkwebview_lightweight_transport as _wk_lightweight
    from chatgpt_web_adapter.client import ChatGPTWebClient

    _original_topic_stream = ChatGPTWebClient.wk_transport_stream_topic
    _topic_fault_calls = 0

    if topic_fault == "idle_reconnect_once":
        _wk_lightweight._FOLLOW_TOPIC_IDLE_RECONNECT_SECONDS = 2.0
        _wk_lightweight._FOLLOW_TOPIC_RECONNECT_BACKOFF_SECONDS = (0.0,)

    def _faulted_topic_stream(
        self,
        topic_id,
        *,
        websocket_url,
        state,
        on_event=None,
        on_token=None,
        should_stop=None,
        stop_on_done=True,
    ):
        global _topic_fault_calls
        _topic_fault_calls += 1
        call_no = _topic_fault_calls
        _append_phase(
            {
                "kind": "topic_fault_enter",
                "mode": topic_fault,
                "call": call_no,
                "topic_id": topic_id,
            }
        )

        if topic_fault == "idle_reconnect_once" and call_no == 1:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if callable(should_stop) and should_stop():
                    _append_phase(
                        {
                            "kind": "topic_fault_idle_released",
                            "call": call_no,
                        }
                    )
                    return None
                time.sleep(0.05)
            raise RuntimeError("GPTTY_LIVE_IDLE_FAULT_WATCHDOG_DID_NOT_FIRE")

        if topic_fault == "silent_after_event" and call_no == 1:
            semantic_events = 0

            def _cut_after_real_events(event):
                nonlocal semantic_events
                if callable(on_event):
                    on_event(event)
                if isinstance(event, dict) and event.get("type") == "raw_ws_event":
                    semantic_events += 1
                    if semantic_events >= 2:
                        raise RuntimeError("GPTTY_LIVE_INJECTED_TOPIC_LOSS")

            try:
                return _original_topic_stream(
                    self,
                    topic_id,
                    websocket_url=websocket_url,
                    state=state,
                    on_event=_cut_after_real_events,
                    on_token=on_token,
                    should_stop=should_stop,
                    stop_on_done=stop_on_done,
                )
            except Exception as exc:
                if "GPTTY_LIVE_INJECTED_TOPIC_LOSS" not in str(exc):
                    raise
                _append_phase(
                    {
                        "kind": "topic_fault_delivery_lost",
                        "call": call_no,
                        "semantic_events": semantic_events,
                    }
                )
                deadline = time.monotonic() + 180.0
                while time.monotonic() < deadline:
                    if callable(should_stop) and should_stop():
                        _append_phase(
                            {
                                "kind": "topic_fault_completion_released",
                                "call": call_no,
                            }
                        )
                        return None
                    time.sleep(0.05)
                raise RuntimeError("GPTTY_LIVE_TOPIC_LOSS_COMPLETION_NOT_OBSERVED")

        return _original_topic_stream(
            self,
            topic_id,
            websocket_url=websocket_url,
            state=state,
            on_event=on_event,
            on_token=on_token,
            should_stop=should_stop,
            stop_on_done=stop_on_done,
        )

    ChatGPTWebClient.wk_transport_stream_topic = _faulted_topic_stream


if os.environ.get("GPTTY_LIVE_FAIL_FIRST_PRE_SUBMIT") == "1":
    from chatgpt_web_adapter.exceptions import RequestError
    from chatgpt_web_adapter.wkwebview_helper_runtime import WKWebViewHelperRuntime
    from chatgpt_web_adapter.wkwebview_provider import WKWebViewTurnProvider

    _original_stream = WKWebViewHelperRuntime._run_streaming_via_turn_broker
    _original_guard_curl = (
        WKWebViewTurnProvider._read_conversation_payload_via_coordinated_curl
    )
    _failed_once = False
    _force_guard_curl_miss_once = False

    def _guard_curl_fault(self, *args, **kwargs):
        global _force_guard_curl_miss_once
        if _force_guard_curl_miss_once:
            _force_guard_curl_miss_once = False
            trace = os.environ.get("GPTTY_LIVE_FAULT_TRACE")
            if trace:
                with Path(trace).open("a", encoding="utf-8") as handle:
                    handle.write("forced_guard_curl_miss\n")
            return None, "FORCED_GUARD_CURL_MISS"
        return _original_guard_curl(self, *args, **kwargs)

    WKWebViewTurnProvider._read_conversation_payload_via_coordinated_curl = (
        _guard_curl_fault
    )

    def _fault_once(self, *args, **kwargs):
        global _failed_once, _force_guard_curl_miss_once
        trace = os.environ.get("GPTTY_LIVE_FAULT_TRACE")
        if not _failed_once:
            _failed_once = True
            _force_guard_curl_miss_once = (
                os.environ.get("GPTTY_LIVE_FAIL_GUARD_CURL_ONCE") == "1"
            )
            if trace:
                Path(trace).write_text(
                    "forced_pre_submit_timeout\n",
                    encoding="utf-8",
                )
            raise RequestError(
                "WKWEBVIEW_PRE_SUBMIT_TIMEOUT",
                request_stage="wkwebview_authority_turn",
            )
        if trace:
            with Path(trace).open("a", encoding="utf-8") as handle:
                handle.write("retry_entered_real_broker\n")
        return _original_stream(self, *args, **kwargs)

    WKWebViewHelperRuntime._run_streaming_via_turn_broker = _fault_once
"""
    compile(sitecustomize, str(path / "sitecustomize.py"), "exec")
    (path / "sitecustomize.py").write_text(sitecustomize, encoding="utf-8")


@dataclass
class TurnResult:
    marker: str
    seconds: float
    conversation_id: str


class LiveHarness:
    def __init__(self, root: Path, *, scenario: str) -> None:
        self.root = root
        self.scenario = scenario
        self.root.mkdir(parents=True, exist_ok=True)
        self.site_dir = self.root / "site"
        write_sitecustomize(self.site_dir)
        self.http_trace = self.root / "http.jsonl"
        self.phase_trace = self.root / "phase.jsonl"
        self.http_trace.unlink(missing_ok=True)
        self.phase_trace.unlink(missing_ok=True)
        short = re.sub(r"[^a-z0-9]+", "-", scenario.lower())[:24]
        self.broker_dir = Path(f"/tmp/gptty-live-{os.getuid()}-{os.getpid()}-{short}")
        shutil.rmtree(self.broker_dir, ignore_errors=True)

    def env(
        self,
        *,
        fault_first_pre_submit: bool = False,
        topic_fault: str | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env.pop("NO_COLOR", None)
        env["GPTTY_LIVE_BROKER_DIR"] = str(self.broker_dir)
        env["GPTTY_LIVE_HTTP_TRACE"] = str(self.http_trace.resolve())
        env["GPTTY_LIVE_PHASE_TRACE"] = str(self.phase_trace.resolve())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.site_dir.resolve()), existing] if existing else [str(self.site_dir.resolve())]
        )
        if topic_fault:
            env["GPTTY_LIVE_TOPIC_FAULT"] = topic_fault
        else:
            env.pop("GPTTY_LIVE_TOPIC_FAULT", None)
        if fault_first_pre_submit:
            env["GPTTY_LIVE_FAIL_FIRST_PRE_SUBMIT"] = "1"
            env["GPTTY_LIVE_FAIL_GUARD_CURL_ONCE"] = "1"
            env["GPTTY_LIVE_FAULT_TRACE"] = str((self.root / "fault.trace").resolve())
        else:
            env.pop("GPTTY_LIVE_FAIL_FIRST_PRE_SUBMIT", None)
            env.pop("GPTTY_LIVE_FAIL_GUARD_CURL_ONCE", None)
            env.pop("GPTTY_LIVE_FAULT_TRACE", None)
        return env

    def canonical_payload(self, conversation_id: str) -> dict[str, Any]:
        client = GpttyClient(
            auth_file=AUTH,
            timeout=30,
            browser_authority_backend="wkwebview",
        )
        return client._client.runtime.get_conversation_payload(conversation_id)

    def canonical_current_text(self, conversation_id: str) -> str:
        payload = self.canonical_payload(conversation_id)
        mapping = payload.get("mapping") if isinstance(payload, dict) else {}
        current = payload.get("current_node") if isinstance(payload, dict) else None
        node = mapping.get(current) if isinstance(mapping, dict) and current else None
        message = node.get("message") if isinstance(node, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else []
        return "".join(part for part in parts if isinstance(part, str))

    def canonical_terminal_text(
        self,
        conversation_id: str,
        *,
        contains: str | None = None,
    ) -> str:
        payload = self.canonical_payload(conversation_id)
        mapping = payload.get("mapping") if isinstance(payload, dict) else {}
        candidates: list[tuple[float, str]] = []
        if not isinstance(mapping, dict):
            return ""
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            if not isinstance(author, dict) or author.get("role") != "assistant":
                continue
            recipient = message.get("recipient")
            if recipient not in {None, "", "all"}:
                continue
            metadata = message.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("is_visually_hidden_from_conversation") is True:
                continue
            if message.get("end_turn") is not True:
                continue
            channel = message.get("channel")
            if isinstance(channel, str) and channel == "commentary":
                continue
            content = message.get("content")
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            text = "".join(part for part in parts if isinstance(part, str))
            if not text:
                continue
            if contains is not None and contains not in text:
                continue
            created = message.get("create_time")
            created_at = float(created) if isinstance(created, (int, float)) else 0.0
            candidates.append((created_at, text))
        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1]

    def wait_canonical_terminal_text(
        self,
        conversation_id: str,
        *,
        contains: str,
        timeout: float = 12.0,
    ) -> str:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                last = self.canonical_terminal_text(
                    conversation_id,
                    contains=contains,
                )
                if last:
                    return last
            except Exception:
                pass
            time.sleep(0.25)
        raise AssertionError(
            f"canonical terminal answer containing {contains!r} not found for "
            f"{conversation_id}; last={last!r}"
        )

    def wait_canonical_text(
        self,
        conversation_id: str,
        expected: str,
        timeout: float = 8.0,
    ) -> str:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                last = self.canonical_current_text(conversation_id)
                if last.strip() == expected.strip():
                    return last
            except Exception:
                pass
            time.sleep(0.25)
        raise AssertionError(
            f"canonical final mismatch for {conversation_id}: expected={expected!r}, last={last!r}"
        )

    def wait_canonical_endswith_line(
        self,
        conversation_id: str,
        expected_line: str,
        timeout: float = 8.0,
    ) -> str:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                last = self.canonical_current_text(conversation_id)
                lines = [line.strip() for line in last.splitlines() if line.strip()]
                if lines and lines[-1] == expected_line:
                    return last
            except Exception:
                pass
            time.sleep(0.25)
        raise AssertionError(
            f"canonical final does not end with {expected_line!r} for "
            f"{conversation_id}: last={last!r}"
        )

    def assistant_texts(self, conversation_id: str) -> list[str]:
        payload = self.canonical_payload(conversation_id)
        mapping = payload.get("mapping") if isinstance(payload, dict) else {}
        texts: list[str] = []
        if not isinstance(mapping, dict):
            return texts
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            if not isinstance(author, dict) or author.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            text = "".join(part for part in parts if isinstance(part, str)).strip()
            if text:
                texts.append(text)
        return texts

    def count_user_prompt(self, conversation_id: str, prompt: str) -> int:
        payload = self.canonical_payload(conversation_id)
        mapping = payload.get("mapping") if isinstance(payload, dict) else {}
        count = 0
        if not isinstance(mapping, dict):
            return count
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            if not isinstance(author, dict) or author.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            text = "".join(part for part in parts if isinstance(part, str))
            if text == prompt:
                count += 1
        return count

    def phase_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.phase_trace.exists():
            return records
        for line in self.phase_trace.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def http_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.http_trace.exists():
            return records
        for line in self.http_trace.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def assert_network_clean(
        self,
        *,
        expected_posts: int | None = None,
        expected_ws_gets: int | None = None,
        allow_conversation_gets: bool = True,
    ) -> dict[str, int]:
        records = self.http_records()
        protected_posts = sum(
            1
            for rec in records
            if rec.get("method") == "POST"
            and rec.get("path") == "/backend-api/f/conversation"
        )
        ws_gets = sum(
            1
            for rec in records
            if rec.get("method") == "GET"
            and rec.get("path") == "/backend-api/celsius/ws/user"
        )
        conversation_gets = sum(
            1
            for rec in records
            if rec.get("method") == "GET"
            and str(rec.get("path", "")).startswith("/backend-api/conversation/")
        )
        rate_limited = sum(1 for rec in records if rec.get("status") == 429)
        if rate_limited:
            raise AssertionError(f"HTTP 429 observed {rate_limited} times")
        if expected_posts is not None and protected_posts != expected_posts:
            raise AssertionError(
                f"protected POST count mismatch: expected {expected_posts}, got {protected_posts}"
            )
        if expected_ws_gets is not None and ws_gets != expected_ws_gets:
            raise AssertionError(
                f"WS bootstrap GET count mismatch: expected {expected_ws_gets}, got {ws_gets}"
            )
        if not allow_conversation_gets and conversation_gets:
            raise AssertionError(
                f"unexpected conversation GETs: {conversation_gets}"
            )
        return {
            "protected_posts": protected_posts,
            "ws_bootstrap_gets": ws_gets,
            "conversation_gets": conversation_gets,
            "http_429": rate_limited,
        }


class PtySession:
    def __init__(
        self,
        harness: LiveHarness,
        name: str,
        *,
        state_path: Path | None = None,
        timeout: int = 90,
        fault_first_pre_submit: bool = False,
        topic_fault: str | None = None,
    ) -> None:
        self.harness = harness
        self.name = name
        self.state_path = state_path or (harness.root / f"{name}.state.json")
        self.log_path = harness.root / f"{name}.terminal.log"
        self.raw = ""
        self.proc: subprocess.Popen[bytes] | None = None
        self.master: int | None = None
        self.timeout = timeout
        self.fault_first_pre_submit = fault_first_pre_submit
        self.topic_fault = topic_fault

    def start(self, *, preserve_state: bool = False) -> None:
        if not preserve_state:
            self.state_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            [
                str(GPTTY),
                "--profile",
                "chatgpt-web",
                "chat",
                "--backend",
                "wkwebview",
                "--state",
                str(self.state_path),
                "--timeout",
                str(self.timeout),
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=self.harness.env(
                fault_first_pre_submit=self.fault_first_pre_submit,
                topic_fault=self.topic_fault,
            ),
            close_fds=True,
        )
        os.close(slave)
        self.proc = proc
        self.master = master
        self.wait_contains("profile:", 20)

    def _pump(self, timeout: float = 0.1) -> None:
        assert self.master is not None
        ready, _, _ = select.select([self.master], [], [], timeout)
        if not ready:
            return
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            return
        if chunk:
            # Full-screen prompt_toolkit asks the terminal for an absolute
            # cursor position (CPR). A raw PTY has no emulator to answer, so
            # emulate a real terminal response for acceptance runs.
            if b"\x1b[6n" in chunk:
                try:
                    os.write(self.master, b"\x1b[1;1R")
                except OSError:
                    pass
            self.raw += chunk.decode("utf-8", errors="replace")
            self.log_path.write_text(self.raw, encoding="utf-8")

    def text(self) -> str:
        return clean(self.raw)

    def send_line(self, line: str) -> float:
        assert self.master is not None
        os.write(self.master, (line + "\n").encode("utf-8"))
        return time.monotonic()

    def wait_phase(
        self,
        kind: str,
        *,
        after: float,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> float:
        if self.proc is None:
            raise AssertionError(f"{self.name}: process is not running")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            for record in reversed(self.harness.phase_records()):
                if (
                    record.get("pid") == self.proc.pid
                    and record.get("kind") == kind
                    and isinstance(record.get("ts"), (int, float))
                    and float(record["ts"]) >= after
                ):
                    return float(record["ts"])
            time.sleep(0.02)
        raise AssertionError(
            f"{self.name}: timeout waiting for phase {kind!r}"
        )

    def wait_contains(self, needle: str, timeout: float = DEFAULT_TIMEOUT) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.1)
            if needle in self.text():
                return time.monotonic()
        raise AssertionError(
            f"{self.name}: timeout waiting for {needle!r}; tail={self.text()[-4000:]!r}"
        )

    def wait_marker_after_working(
        self,
        marker_text: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            text = self.text()
            working = text.find("working")
            if working >= 0 and marker_text in text[working + len("working") :]:
                return time.monotonic()
        raise AssertionError(
            f"{self.name}: timeout waiting for {marker_text!r} after working banner"
        )

    def wait_contains_after(
        self,
        anchor: str,
        needle: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            text = self.text()
            anchor_at = text.rfind(anchor)
            if anchor_at >= 0 and needle in text[anchor_at + len(anchor) :]:
                return time.monotonic()
        raise AssertionError(
            f"{self.name}: timeout waiting for {needle!r} after {anchor!r}"
        )

    def wait_exact_line(
        self,
        line: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            if any(item.strip() == line for item in self.text().splitlines()):
                return time.monotonic()
        raise AssertionError(
            f"{self.name}: timeout waiting exact line {line!r}; tail={self.text()[-4000:]!r}"
        )

    def wait_quiet(self, quiet_for: float = 0.35, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        last_change = time.monotonic()
        last_len = len(self.raw)
        while time.monotonic() < deadline:
            self._pump(0.05)
            if len(self.raw) != last_len:
                last_len = len(self.raw)
                last_change = time.monotonic()
            elif time.monotonic() - last_change >= quiet_for:
                return

    def state(self) -> dict[str, Any]:
        path = self.state_path
        if not path.exists():
            suffix = path.suffix
            stem = path.stem if suffix else path.name
            candidates = sorted(
                path.parent.glob(f"{stem}.session-*{suffix}"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return {}
            path = candidates[0]
        return json.loads(path.read_text(encoding="utf-8"))

    def wait_conversation_id(self, timeout: float = 5.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            cid = self.state().get("current_conversation")
            if isinstance(cid, str) and cid:
                return cid
            time.sleep(0.05)
        raise AssertionError(f"{self.name}: conversation id not persisted")

    def wait_write_identity(self, timeout: float = 5.0) -> str:
        journal = self.harness.root / "stream-delivery.jsonl"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump(0.05)
            if journal.exists():
                identities: list[str] = []
                for raw in journal.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if record.get("event") != "browser_native_write_identity_resolved":
                        continue
                    cid = record.get("conversation_id")
                    if isinstance(cid, str) and cid.strip():
                        identities.append(cid.strip())
                if identities:
                    return identities[-1]
            time.sleep(0.05)
        raise AssertionError(f"{self.name}: write identity not observed")

    def exact_turn(
        self,
        marker_text: str,
        *,
        timeout: float = 45.0,
        prompt: str | None = None,
    ) -> TurnResult:
        prompt = prompt or f"Reply exactly {marker_text}. No other text."
        started = self.send_line(prompt)
        finished = self.wait_phase(
            "gptty_send_return",
            after=started,
            timeout=timeout,
        )
        self.wait_contains_after("working", "answer", 5)
        self.wait_quiet()
        cid = self.wait_conversation_id()
        self.harness.wait_canonical_text(cid, marker_text)
        return TurnResult(
            marker=marker_text,
            seconds=round(finished - started, 3),
            conversation_id=cid,
        )

    def terminate_for_test(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
        if self.master is not None:
            try:
                os.close(self.master)
            except OSError:
                pass

    def close(self) -> int:
        if self.proc is None or self.master is None:
            return 0
        if self.proc.poll() is None:
            try:
                self.send_line("/exit")
            except OSError:
                pass
            deadline = time.monotonic() + 10
            while self.proc.poll() is None and time.monotonic() < deadline:
                self._pump(0.1)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
        code = int(self.proc.returncode or 0)
        try:
            os.close(self.master)
        except OSError:
            pass
        return code


def assert_clean_exit(code: int, label: str) -> None:
    if code != 0 and sys.exc_info()[0] is None:
        raise AssertionError(f"{label} exit code {code}")


def assert_no_ui_transport_errors(session: PtySession) -> None:
    text = session.text().lower()
    forbidden = [
        "too many requests",
        "слишком много запросов",
        "falling back to canonical polling",
        "live stream unavailable",
        "chat request failed",
    ]
    hits = [needle for needle in forbidden if needle in text]
    if hits:
        raise AssertionError(f"{session.name}: forbidden UI errors present: {hits}")


def scenario_fresh_sequence(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "fresh-sequence")
    session.start()
    turns: list[dict[str, Any]] = []
    try:
        for i in range(3):
            result = session.exact_turn(marker(f"LIVE_SEQ_{i}"))
            turns.append(result.__dict__)
        assert len({item["conversation_id"] for item in turns}) == 1
        assert_no_ui_transport_errors(session)
        network = h.assert_network_clean(expected_posts=3)
        return {"ok": True, "turns": turns, "network": network}
    finally:
        assert_clean_exit(session.close(), "fresh sequence")


def scenario_resume_new(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "resume-new")
    session.start()
    try:
        first = session.exact_turn(marker("LIVE_RESUME_A"))
        session.send_line("/new")
        session.wait_contains("Started a new conversation.", 15)
        second = session.exact_turn(marker("LIVE_RESUME_B"))
        if first.conversation_id == second.conversation_id:
            raise AssertionError("/new did not create a different conversation")
        session.send_line(f"/resume {first.conversation_id}")
        session.wait_contains("Resumed:", 20)
        if session.wait_conversation_id() != first.conversation_id:
            raise AssertionError("/resume did not restore the requested conversation")
        third = session.exact_turn(marker("LIVE_RESUME_C"))
        if third.conversation_id != first.conversation_id:
            raise AssertionError("continuation after /resume used the wrong conversation")
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "first": first.__dict__,
            "second": second.__dict__,
            "third": third.__dict__,
            "network": h.assert_network_clean(expected_posts=3),
        }
    finally:
        assert_clean_exit(session.close(), "resume-new")


def run_tool_heavy(
    h: LiveHarness,
    session: PtySession,
    *,
    prefix: str,
    timeout: float = 45.0,
) -> dict[str, Any]:
    a = marker(f"{prefix}_PROGRESS_A")
    b = marker(f"{prefix}_PROGRESS_B")
    final = marker(f"{prefix}_FINAL")
    prompt = (
        "@CodexTool use CodexTool to inspect the current workspace and git branch. "
        f"Before the first tool call, send a user-visible progress sentence containing exactly {a}. "
        f"After the first tool result and before a later tool call, send another progress sentence containing exactly {b}. "
        "Make at least two tool calls. "
        f"Finish with {final} on its own line."
    )
    started = session.send_line(prompt)
    a_at = session.wait_marker_after_working(a, timeout)
    b_at = session.wait_marker_after_working(b, timeout)
    final_at = session.wait_phase(
        "gptty_send_return",
        after=started,
        timeout=timeout,
    )
    session.wait_contains_after("working", "answer", 5)
    session.wait_quiet()
    cid = session.wait_conversation_id()
    h.wait_canonical_endswith_line(cid, final)
    assistant = h.assistant_texts(cid)
    if not any(a in text for text in assistant):
        raise AssertionError("canonical assistant branch missing progress marker A")
    if not any(b in text for text in assistant):
        raise AssertionError("canonical assistant branch missing progress marker B")
    terminal_tools = set(
        re.findall(r"◇\s+([A-Za-z_][A-Za-z0-9_.-]*)", session.text())
    )
    if len(terminal_tools) < 2:
        raise AssertionError(
            f"expected at least two distinct rendered tool activities, got {terminal_tools}"
        )
    assert_no_ui_transport_errors(session)
    return {
        "conversation_id": cid,
        "progress_a_s": round(a_at - started, 3),
        "progress_b_s": round(b_at - started, 3),
        "final_s": round(final_at - started, 3),
        "tool_activity_lines": sorted(terminal_tools),
        "progress_a": a,
        "progress_b": b,
        "final": final,
    }


def scenario_tool_heavy(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "tool-heavy")
    session.start()
    try:
        result = run_tool_heavy(h, session, prefix="LIVE_TOOL")
        return {
            "ok": True,
            "turn": result,
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "tool-heavy")


def scenario_attached_follow_order(h: LiveHarness) -> dict[str, Any]:
    writer = PtySession(h, "attach-order-writer", timeout=120)
    follower = PtySession(h, "attach-order-follower", timeout=120)
    writer.start()
    follower_started = False
    try:
        seed = writer.exact_turn(marker("LIVE_ATTACH_SEED"), timeout=60)
        cid = seed.conversation_id
        progress_a = marker("LIVE_ATTACH_PROGRESS_A")
        progress_b = marker("LIVE_ATTACH_PROGRESS_B")
        final = marker("LIVE_ATTACH_FINAL")
        prompt = (
            "@CodexTool use CodexTool to inspect the current workspace and Git state. "
            f"Before the first tool call, send a user-visible progress sentence containing exactly {progress_a}. "
            f"After the first tool result and before a later tool call, send another user-visible progress sentence containing exactly {progress_b}. "
            "Make at least two tool calls. Do not repeat either progress marker in the final answer. "
            f"Finish the final answer with {final} on its own line."
        )
        started = writer.send_line(prompt)
        writer.wait_marker_after_working(progress_a, 90)

        follower.start()
        follower_started = True
        follower.send_line(f"/resume {cid}")
        follower.wait_contains("Resumed:", 30)
        follower.wait_contains("Following active response", 30)
        follower.wait_contains(progress_b, 90)
        writer.wait_phase("gptty_send_return", after=started, timeout=120)
        follower.wait_contains(final, 120)
        follower.wait_contains_after(final, "chat:", 15)
        follower.wait_quiet(quiet_for=0.5, timeout=5.0)

        segment = follower.text()
        anchor = segment.rfind("Following active response")
        if anchor < 0:
            anchor = segment.rfind("Resumed:")
        if anchor >= 0:
            segment = segment[anchor:]
        progress_b_at = segment.find(progress_b)
        final_at = segment.rfind(final)
        if progress_b_at < 0 or final_at < 0:
            raise AssertionError(
                f"attached follower missing ordered markers: progress_b={progress_b_at}, final={final_at}"
            )
        if progress_b_at >= final_at:
            raise AssertionError(
                f"attached follower rendered progress after final: progress_b={progress_b_at}, final={final_at}"
            )
        tail = segment[final_at + len(final) :]
        late_blocks = [
            needle
            for needle in (progress_a, progress_b, "Thinking", "◇ ")
            if needle in tail
        ]
        if late_blocks:
            raise AssertionError(
                f"attached follower rendered intermediate activity after final: {late_blocks}; tail={tail[-2000:]!r}"
            )
        assert_no_ui_transport_errors(writer)
        assert_no_ui_transport_errors(follower)
        return {
            "ok": True,
            "conversation_id": cid,
            "progress_b_before_final": True,
            "late_intermediate_after_final": late_blocks,
            "network": h.assert_network_clean(expected_posts=2),
        }
    finally:
        if follower_started:
            assert_clean_exit(follower.close(), "attached follower")
        assert_clean_exit(writer.close(), "attached writer")



def scenario_own_topic_loss_recovery(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(
        h,
        "own-topic-loss",
        timeout=180,
        topic_fault="silent_after_event",
    )
    session.start()
    try:
        final = marker("LIVE_OWN_LOSS_FINAL")
        prompt = (
            "@CodexTool use CodexTool to inspect the current workspace and Git state. "
            "Make at least two tool calls, then give a concise final answer. "
            f"Finish the final answer with {final} on its own line."
        )
        started = session.send_line(prompt)
        session.wait_phase("gptty_send_return", after=started, timeout=180)
        session.wait_contains(final, 20)
        session.wait_quiet(quiet_for=0.5, timeout=5.0)
        cid = session.wait_conversation_id(8)
        h.wait_canonical_endswith_line(cid, final, timeout=15)
        if h.count_user_prompt(cid, prompt) != 1:
            raise AssertionError("delivery recovery duplicated the user prompt")
        if session.proc is None:
            raise AssertionError("own-topic-loss process is missing")
        phases = [
            rec
            for rec in h.phase_records()
            if rec.get("pid") == session.proc.pid
        ]
        kinds = [rec.get("kind") for rec in phases]
        if "topic_fault_delivery_lost" not in kinds:
            raise AssertionError("topic loss injection did not trigger")
        if "topic_fault_completion_released" not in kinds:
            raise AssertionError("independent completion did not release lost topic")
        assert_no_ui_transport_errors(session)
        network = h.assert_network_clean(expected_posts=1)
        if network["conversation_gets"] < 1:
            raise AssertionError("lost delivery did not perform terminal canonical reconcile")
        return {
            "ok": True,
            "conversation_id": cid,
            "completion_recovered": True,
            "network": network,
        }
    finally:
        assert_clean_exit(session.close(), "own-topic-loss")


def scenario_idle_topic_reconnect(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(
        h,
        "idle-topic-reconnect",
        timeout=180,
        topic_fault="idle_reconnect_once",
    )
    session.start()
    try:
        final = marker("LIVE_IDLE_RECONNECT_FINAL")
        prompt = (
            "@CodexTool use CodexTool to inspect the current workspace and Git state. "
            "Make at least two tool calls. "
            f"Finish with {final} on its own line."
        )
        started = session.send_line(prompt)
        session.wait_phase("gptty_send_return", after=started, timeout=180)
        session.wait_exact_line(final, 20)
        session.wait_quiet(quiet_for=0.5, timeout=5.0)
        cid = session.wait_write_identity(8)
        if session.proc is None:
            raise AssertionError("idle-topic-reconnect process is missing")
        phases = [
            rec
            for rec in h.phase_records()
            if rec.get("pid") == session.proc.pid
        ]
        enters = [rec for rec in phases if rec.get("kind") == "topic_fault_enter"]
        if len(enters) < 2:
            raise AssertionError(f"idle watchdog did not re-subscribe: {enters!r}")
        if not any(rec.get("kind") == "topic_fault_idle_released" for rec in phases):
            raise AssertionError("idle watchdog did not trigger")
        answers = [rec for rec in phases if rec.get("kind") == "renderer_answer"]
        if len(answers) != 1:
            raise AssertionError(f"expected one rendered answer, got {answers!r}")
        answer = answers[0]
        if (
            answer.get("length") != answer.get("stream_length")
            or answer.get("sha256") != answer.get("stream_sha256")
        ):
            raise AssertionError(f"rendered answer diverged from streamed answer: {answer!r}")
        assert_no_ui_transport_errors(session)
        network = h.assert_network_clean(expected_posts=1, expected_ws_gets=1)
        return {
            "ok": True,
            "conversation_id": cid,
            "topic_subscriptions": len(enters),
            "answer_integrity": True,
            "network": network,
        }
    finally:
        assert_clean_exit(session.close(), "idle-topic-reconnect")


def scenario_attached_topic_loss_recovery(h: LiveHarness) -> dict[str, Any]:
    writer = PtySession(h, "attach-loss-writer", timeout=180)
    follower = PtySession(
        h,
        "attach-loss-follower",
        timeout=180,
        topic_fault="silent_after_event",
    )
    writer.start()
    follower_started = False
    try:
        seed = writer.exact_turn(marker("LIVE_ATTACH_LOSS_SEED"), timeout=60)
        cid = seed.conversation_id
        progress = marker("LIVE_ATTACH_LOSS_PROGRESS")
        final = marker("LIVE_ATTACH_LOSS_FINAL")
        prompt = (
            "@CodexTool use CodexTool to inspect the current workspace and Git state. "
            f"Before the first tool call, send a user-visible progress sentence containing exactly {progress}. "
            "Make at least two tool calls. Do not repeat the progress marker in the final answer. "
            f"Finish the final answer with {final} on its own line."
        )
        started = writer.send_line(prompt)
        writer.wait_marker_after_working(progress, 90)

        follower.start()
        follower_started = True
        follower.send_line(f"/resume {cid}")
        follower.wait_contains("Resumed:", 30)
        follower.wait_contains("Following active response", 30)
        writer.wait_phase("gptty_send_return", after=started, timeout=180)
        follower.wait_contains(final, 180)
        follower.wait_contains_after(final, "chat:", 20)
        follower.wait_quiet(quiet_for=0.5, timeout=5.0)

        if follower.proc is None:
            raise AssertionError("attached loss follower process is missing")
        phases = [
            rec
            for rec in h.phase_records()
            if rec.get("pid") == follower.proc.pid
        ]
        kinds = [rec.get("kind") for rec in phases]
        if "topic_fault_delivery_lost" not in kinds:
            raise AssertionError("attached delivery loss injection did not trigger")
        if "topic_fault_completion_released" not in kinds:
            raise AssertionError("attached follower did not observe independent completion")

        segment = follower.text()
        final_at = segment.rfind(final)
        if final_at < 0:
            raise AssertionError("attached follower did not render final")
        tail = segment[final_at + len(final):]
        if any(needle in tail for needle in ("Thinking", "◇ ", progress)):
            raise AssertionError(f"intermediate output appeared after recovered final: {tail[-2000:]!r}")
        if h.count_user_prompt(cid, prompt) != 1:
            raise AssertionError("attached recovery duplicated the user prompt")
        assert_no_ui_transport_errors(writer)
        assert_no_ui_transport_errors(follower)
        network = h.assert_network_clean(expected_posts=2)
        if network["conversation_gets"] < 1:
            raise AssertionError("attached loss did not perform terminal canonical reconcile")
        return {
            "ok": True,
            "conversation_id": cid,
            "completion_recovered": True,
            "network": network,
        }
    finally:
        if follower_started:
            assert_clean_exit(follower.close(), "attached-topic-loss follower")
        assert_clean_exit(writer.close(), "attached-topic-loss writer")


def scenario_external_attach_midturn_stall_recovery(h: LiveHarness) -> dict[str, Any]:
    writer = PtySession(h, "external-stall-writer", timeout=180)
    follower = PtySession(
        h,
        "external-stall-follower",
        timeout=180,
        topic_fault="silent_after_event",
    )
    writer.start()
    follower_started = False
    try:
        progress_a = marker("LIVE_EXTERNAL_STALL_A")
        progress_b = marker("LIVE_EXTERNAL_STALL_B")
        progress_c = marker("LIVE_EXTERNAL_STALL_C")
        final = marker("LIVE_EXTERNAL_STALL_FINAL")
        prompt = (
            "@CodexTool use CodexTool and perform these steps strictly sequentially, never in parallel. "
            f"First send a user-visible progress sentence containing exactly {progress_a}, then call bash with sleep 8; pwd. "
            f"After that result send a progress sentence containing exactly {progress_b}, then call bash with sleep 12; pwd. "
            f"After that result send a progress sentence containing exactly {progress_c}. "
            "Do not repeat any progress marker in the final answer. "
            f"Finish the final answer with {final} on its own line."
        )
        started = writer.send_line(prompt)
        writer.wait_marker_after_working(progress_a, 90)
        writer.wait_contains_after(progress_a, "◇ bash", 30)
        if writer.proc is None:
            raise AssertionError("external-stall writer process is missing")
        active_dir = (
            Path.home()
            / "Library"
            / "Application Support"
            / "chatgpt-web-adapter"
            / "active-streams"
        )
        cid = None
        active_path = None
        cid_deadline = time.monotonic() + 15.0
        while time.monotonic() < cid_deadline:
            for candidate in active_dir.glob("*.json"):
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if payload.get("pid") != writer.proc.pid:
                    continue
                value = payload.get("conversation_id")
                if isinstance(value, str) and value.strip():
                    cid = value.strip()
                    active_path = candidate
                    break
            if cid is not None:
                break
            time.sleep(0.05)
        if cid is None or active_path is None:
            raise AssertionError("writer active-stream conversation id not found mid-turn")
        active_path.unlink(missing_ok=True)

        follower.start()
        follower_started = True
        follower.send_line(f"/resume {cid}")
        follower.wait_contains("Resumed:", 30)
        follower.wait_contains("Following active response", 30)

        if follower.proc is None:
            raise AssertionError("external-stall follower process is missing")
        fault_at = follower.wait_phase(
            "topic_fault_delivery_lost",
            after=started,
            timeout=60,
        )

        # Marker B must be produced while the follower's first topic socket is
        # intentionally silent.
        writer.wait_marker_after_working(progress_b, 120)

        recovered_enter = None
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            follower._pump(0.05)
            reentries = [
                rec
                for rec in h.phase_records()
                if rec.get("pid") == follower.proc.pid
                and rec.get("kind") == "topic_fault_enter"
                and isinstance(rec.get("ts"), (int, float))
            ]
            if len(reentries) >= 2:
                recovered_enter = reentries[1]
                break
            time.sleep(0.05)
        if recovered_enter is None:
            raise AssertionError("silent topic stall was not replaced by a second subscription")

        stall_seconds = float(recovered_enter["ts"]) - fault_at
        if not 10.0 <= stall_seconds <= 18.0:
            raise AssertionError(
                f"production idle watchdog fired outside expected window: {stall_seconds:.3f}s"
            )

        # This marker was created during the silent interval and must arrive as
        # a cursor catchup on the replacement topic socket while the writer is
        # still actively executing the turn. Final ordering is covered by the
        # separate attached-follow ordering acceptance.
        follower.wait_contains_after("Following active response", progress_b, 30)
        writer._pump(0.05)
        if writer.proc is None:
            raise AssertionError("external-stall writer process is missing")
        writer_returns = [
            rec
            for rec in h.phase_records()
            if rec.get("pid") == writer.proc.pid
            and rec.get("kind") == "gptty_send_return"
            and isinstance(rec.get("ts"), (int, float))
            and float(rec["ts"]) >= started
        ]
        if writer_returns:
            raise AssertionError(
                "topic recovery was observed only after the writer had already completed"
            )

        segment = follower.text()
        follow_anchor = segment.rfind("Following active response")
        if follow_anchor >= 0:
            segment = segment[follow_anchor:]
        if progress_b not in segment:
            raise AssertionError("recovered mid-turn marker was not rendered after attach")
        assert_no_ui_transport_errors(writer)
        assert_no_ui_transport_errors(follower)
        network = h.assert_network_clean(expected_posts=1)
        if network["http_429"] != 0:
            raise AssertionError(f"mid-turn recovery hit 429: {network}")
        return {
            "ok": True,
            "conversation_id": cid,
            "active_stream_registry_removed": not active_path.exists(),
            "stall_seconds": round(stall_seconds, 3),
            "midturn_progress_recovered_while_writer_active": True,
            "network": network,
        }
    finally:
        if follower_started:
            assert_clean_exit(follower.close(), "external-stall follower")
        assert_clean_exit(writer.close(), "external-stall writer")


def scenario_tool_heavy_full_final(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "tool-heavy-full-final", timeout=120)
    session.start()
    try:
        progress_a = marker("LIVE_FULL_PROGRESS_A")
        progress_b = marker("LIVE_FULL_PROGRESS_B")
        begin = marker("LIVE_FULL_BEGIN")
        middle = marker("LIVE_FULL_MIDDLE")
        end = marker("LIVE_FULL_END")
        prompt = (
            "@CodexTool use CodexTool to inspect the current workspace and its Git state. "
            f"Before the first tool call, send a user-visible progress sentence containing exactly {progress_a}. "
            f"After the first tool result and before a later tool call, send another progress sentence containing exactly {progress_b}. "
            "Make at least two tool calls. After the last tool call, write a substantial final answer of at least 1800 characters "
            "with several paragraphs. Do not call any tools after starting the final answer. "
            f"Put {begin} on its own line at the start of the final answer, {middle} on its own line around the middle, "
            f"and {end} on its own line at the end."
        )
        started = session.send_line(prompt)
        progress_a_at = session.wait_marker_after_working(progress_a, 90)
        progress_b_at = session.wait_marker_after_working(progress_b, 90)
        returned_at = session.wait_phase("gptty_send_return", after=started, timeout=120)
        session.wait_quiet()
        cid = session.wait_conversation_id(8)
        canonical = h.wait_canonical_terminal_text(
            cid,
            contains=end,
            timeout=15,
        )
        if len(canonical) < 1000:
            raise AssertionError(
                f"canonical final unexpectedly short for truncation gate: {len(canonical)}"
            )
        for required in (begin, middle, end):
            if required not in canonical:
                raise AssertionError(f"canonical final missing required marker {required}")

        if session.proc is None:
            raise AssertionError("tool-heavy-full-final process is missing")
        renderer_records = [
            record
            for record in h.phase_records()
            if record.get("pid") == session.proc.pid
            and record.get("kind") == "renderer_answer"
            and isinstance(record.get("ts"), (int, float))
            and float(record["ts"]) >= started
        ]
        if not renderer_records:
            raise AssertionError("renderer.answer was not observed")
        rendered = renderer_records[-1]
        canonical_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if rendered.get("length") != len(canonical):
            raise AssertionError(
                f"renderer final length mismatch: renderer={rendered.get('length')} canonical={len(canonical)}"
            )
        if rendered.get("sha256") != canonical_sha256:
            raise AssertionError(
                "renderer final hash differs from canonical final"
            )
        terminal_text = session.text()
        working_at = terminal_text.find("working")
        visible_after_working = (
            terminal_text[working_at + len("working") :] if working_at >= 0 else ""
        )
        visible_marker_positions = {
            required: visible_after_working.find(required)
            for required in (begin, middle, end)
        }
        if any(position < 0 for position in visible_marker_positions.values()):
            raise AssertionError(
                f"physical terminal output missing final markers: {visible_marker_positions}"
            )
        if not (
            visible_marker_positions[begin]
            < visible_marker_positions[middle]
            < visible_marker_positions[end]
        ):
            raise AssertionError(
                f"physical terminal final markers out of order: {visible_marker_positions}"
            )
        terminal_tools = set(
            re.findall(r"◇\s+([A-Za-z_][A-Za-z0-9_.-]*)", terminal_text)
        )
        if len(terminal_tools) < 2:
            raise AssertionError(
                f"expected at least two rendered tool activities, got {terminal_tools}"
            )
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "conversation_id": cid,
            "seconds_to_progress_a": round(progress_a_at - started, 3),
            "seconds_to_progress_b": round(progress_b_at - started, 3),
            "seconds_to_return": round(returned_at - started, 3),
            "canonical_length": len(canonical),
            "canonical_sha256": canonical_sha256,
            "renderer_length": rendered.get("length"),
            "renderer_sha256": rendered.get("sha256"),
            "stream_length": rendered.get("stream_length"),
            "stream_sha256": rendered.get("stream_sha256"),
            "stream_suppressed": rendered.get("stream_suppressed"),
            "tool_activity_lines": sorted(terminal_tools),
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "tool-heavy-full-final")


def scenario_long_seed(h: LiveHarness) -> dict[str, Any]:
    pad = "live-long-context-" * 1200
    prep_marker = marker("LIVE_LONG_PREP")
    prep_prompt = (
        f"Disposable long-chat preparation. {pad}\n"
        f"Reply exactly {prep_marker}. No other text."
    )
    prep_client = GpttyClient(
        auth_file=AUTH,
        timeout=60,
        browser_authority_backend="wkwebview",
    )
    prep_started = time.monotonic()
    prep_response = prep_client.send(prep_prompt, stream=True)
    prep_seconds = round(time.monotonic() - prep_started, 3)
    if prep_response.text.strip() != prep_marker:
        raise AssertionError(
            f"long-context preparation mismatch: {prep_response.text!r}"
        )
    cid = prep_response.conversation.conversation_id
    seed_path = h.root.parent / "long-seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "conversation_id": cid,
                "marker": prep_marker,
                "prompt_chars": len(prep_prompt),
                "seconds": prep_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "conversation_id": cid,
        "marker": prep_marker,
        "prompt_chars": len(prep_prompt),
        "seconds_to_seed": prep_seconds,
        "seed_path": str(seed_path),
    }


def scenario_long_restart(h: LiveHarness) -> dict[str, Any]:
    seed_path = h.root.parent / "long-seed.json"
    if not seed_path.exists():
        raise AssertionError(
            "long seed is missing; run long_seed for the same --round first"
        )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    cid = str(seed["conversation_id"])
    state_path = h.root / "long-restart.state.json"
    state_path.write_text(
        json.dumps({"current_conversation": cid, "model": None}),
        encoding="utf-8",
    )
    second = PtySession(h, "long-restart", state_path=state_path)
    second.start(preserve_state=True)
    try:
        if second.wait_conversation_id() != cid:
            raise AssertionError("fresh process did not restore long-chat state")
        tool = run_tool_heavy(h, second, prefix="LIVE_LONG_TOOL", timeout=50)
        if tool["conversation_id"] != cid:
            raise AssertionError("long-chat continuation switched conversation")
        return {
            "ok": True,
            "seed": seed,
            "tool": tool,
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(second.close(), "long restart")


def scenario_stop_next(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "stop-next", timeout=120)
    session.start()
    try:
        prompt = (
            "Write a very long detailed essay of at least 2500 words about distributed systems. "
            "Do not summarize and do not finish early."
        )
        started = session.send_line(prompt)
        try:
            session.wait_contains("Thinking", 15)
        except AssertionError:
            time.sleep(5)
        session.send_line("/stop")
        session.wait_contains("Stopping ChatGPT", 10)
        session.wait_contains("Stopped by user.", 30)
        stopped_s = round(time.monotonic() - started, 3)
        cid = session.wait_conversation_id(8)
        next_turn = session.exact_turn(marker("LIVE_STOP_NEXT"), timeout=45)
        if next_turn.conversation_id != cid:
            raise AssertionError("next turn after stop used a different conversation")
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "stopped_s": stopped_s,
            "next": next_turn.__dict__,
            "network": h.assert_network_clean(expected_posts=2),
        }
    finally:
        assert_clean_exit(session.close(), "stop-next")


def scenario_temporary_first(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "temporary-first")
    session.start()
    try:
        session.send_line("/temporary")
        session.wait_contains("Started a new Temporary ChatGPT conversation.", 15)
        expected = marker("LIVE_TEMP_FIRST")
        started = session.send_line(f"Reply exactly {expected}. No other text.")
        finished = session.wait_phase(
            "gptty_send_return",
            after=started,
            timeout=40,
        )
        session.wait_contains_after("working", "answer", 5)
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "seconds_to_answer": round(finished - started, 3),
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "temporary-first")


def scenario_temporary(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "temporary")
    session.start()
    try:
        session.send_line("/temporary")
        session.wait_contains("Started a new Temporary ChatGPT conversation.", 15)
        a = marker("LIVE_TEMP_A")
        b = marker("LIVE_TEMP_B")
        a_started = session.send_line(f"Reply exactly {a}. No other text.")
        session.wait_phase("gptty_send_return", after=a_started, timeout=45)
        session.wait_contains_after("working", "answer", 5)
        b_started = session.send_line(f"Reply exactly {b}. No other text.")
        session.wait_phase("gptty_send_return", after=b_started, timeout=45)
        session.wait_contains_after("working", "answer", 5)
        before = {
            path: path.stat().st_mtime
            for path in (Path.home() / "Documents/gptty-exports").glob("*.md")
        }
        session.send_line("/export")
        session.wait_contains("Exported Markdown:", 20)
        time.sleep(0.3)
        export_dir = Path.home() / "Documents/gptty-exports"
        candidates = [
            path
            for path in export_dir.glob("*.md")
            if path not in before or path.stat().st_mtime > before.get(path, 0)
        ]
        if not candidates:
            raise AssertionError("temporary export did not create a Markdown file")
        exported = max(candidates, key=lambda p: p.stat().st_mtime).read_text(
            encoding="utf-8",
            errors="replace",
        )
        if a not in exported or b not in exported:
            raise AssertionError("temporary export missing one or both turns")
        session.send_line("/new")
        session.wait_contains("Started a new conversation.", 15)
        if session.state().get("current_conversation") is not None:
            raise AssertionError("/new did not clear state after Temporary")
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "export": str(max(candidates, key=lambda p: p.stat().st_mtime)),
            "network": h.assert_network_clean(expected_posts=2),
        }
    finally:
        assert_clean_exit(session.close(), "temporary")


def scenario_image(h: LiveHarness) -> dict[str, Any]:
    image = h.root / "solid-red.png"
    make_red_png(image)
    session = PtySession(h, "image")
    session.start()
    try:
        session.send_line(f'/image "{image.resolve()}"')
        session.wait_contains("Attached for next prompt:", 15)
        expected = marker("LIVE_IMAGE_RED")
        prompt = (
            f"If the attached image is solid red, reply exactly {expected}. "
            "Otherwise reply IMAGE_FAIL."
        )
        turn = session.exact_turn(expected, prompt=prompt, timeout=60)
        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "turn": turn.__dict__,
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "image")


def scenario_goal(h: LiveHarness) -> dict[str, Any]:
    session = PtySession(h, "goal")
    session.start()
    try:
        expected = marker("LIVE_GOAL")
        objective = (
            "Complete immediately. Your response must begin with GPTTY_GOAL: COMPLETE "
            f"and the body after that line must be exactly {expected}."
        )
        started = session.send_line(f'/goal "{objective}"')
        session.wait_contains("Goal · active ·", 15)
        session.wait_contains(expected, 60)
        complete = session.wait_contains("Goal · complete", 20)
        state = session.state()
        if state.get("goal", {}).get("status") != "complete":
            raise AssertionError(f"goal state is not complete: {state.get('goal')}")

        if "GPTTY_CHECKPOINT:" in session.text():
            raise AssertionError("Goal protocol checkpoint leaked into visible TUI")

        assert_no_ui_transport_errors(session)
        return {
            "ok": True,
            "seconds": round(complete - started, 3),
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "goal")


def scenario_parallel5(h: LiveHarness) -> dict[str, Any]:
    sessions = [PtySession(h, f"parallel-{i}") for i in range(5)]
    startup_errors: list[str | None] = [None] * 5

    def start_worker(i: int) -> None:
        try:
            sessions[i].start()
        except Exception as exc:
            startup_errors[i] = f"{type(exc).__name__}: {exc}"

    start_threads = [
        threading.Thread(target=start_worker, args=(i,))
        for i in range(5)
    ]
    for thread in start_threads:
        thread.start()
    for thread in start_threads:
        thread.join(timeout=20)
    if any(thread.is_alive() for thread in start_threads):
        raise AssertionError("parallel startup did not finish")
    if any(startup_errors):
        raise AssertionError(f"parallel startup failed: {startup_errors}")

    barrier = threading.Barrier(5)
    results: list[dict[str, Any] | None] = [None] * 5

    def worker(i: int) -> None:
        expected = marker(f"LIVE_PAR_{i}")
        barrier.wait()
        try:
            turn = sessions[i].exact_turn(expected, timeout=50)
            results[i] = {"ok": True, **turn.__dict__}
        except Exception as exc:
            results[i] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    try:
        if any(thread.is_alive() for thread in threads):
            raise AssertionError("parallel worker did not finish")
        if not all(item and item.get("ok") for item in results):
            raise AssertionError(f"parallel results failed: {results}")
        for session in sessions:
            assert_no_ui_transport_errors(session)
        network = h.assert_network_clean(
            expected_posts=5,
            expected_ws_gets=5,
            allow_conversation_gets=False,
        )
        return {"ok": True, "results": results, "network": network}
    finally:
        for session in sessions:
            session.terminate_for_test()


def scenario_fault_seed(h: LiveHarness) -> dict[str, Any]:
    state_path = h.root / "fault-seed.state.json"
    session = PtySession(h, "fault-seed", state_path=state_path)
    session.start()
    try:
        baseline = session.exact_turn(
            marker("LIVE_FAULT_BASELINE"),
            timeout=45,
        )
        assert_no_ui_transport_errors(session)
        seed_path = h.root.parent / "fault-seed.json"
        seed_path.write_text(
            json.dumps(baseline.__dict__, indent=2),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "baseline": baseline.__dict__,
            "seed_path": str(seed_path),
            "network": h.assert_network_clean(expected_posts=1),
        }
    finally:
        assert_clean_exit(session.close(), "fault seed")


def scenario_fault_pre_submit(h: LiveHarness) -> dict[str, Any]:
    seed_path = h.root.parent / "fault-seed.json"
    if not seed_path.exists():
        raise AssertionError(
            "fault seed is missing; run fault_seed for the same --round first"
        )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    baseline_conversation_id = str(seed["conversation_id"])
    state_path = h.root / "fault-pre-submit.state.json"
    state_path.write_text(
        json.dumps(
            {"current_conversation": baseline_conversation_id, "model": None}
        ),
        encoding="utf-8",
    )
    session = PtySession(
        h,
        "fault-pre-submit",
        state_path=state_path,
        fault_first_pre_submit=True,
    )
    session.start(preserve_state=True)
    try:
        if session.wait_conversation_id() != baseline_conversation_id:
            raise AssertionError("fault recovery did not restore baseline conversation")
        expected = marker("LIVE_FAULT_RECOVERY")
        prompt = f"Reply exactly {expected}. No other text."
        started = session.send_line(prompt)
        completed = session.wait_contains_after("answer", "chat:", 40)
        cid = session.wait_conversation_id(5)
        trace_path = h.root / "fault.trace"
        trace = (
            trace_path.read_text(encoding="utf-8").splitlines()
            if trace_path.exists()
            else []
        )
        if trace != [
            "forced_pre_submit_timeout",
            "forced_guard_curl_miss",
            "retry_entered_real_broker",
        ]:
            raise AssertionError(f"unexpected recovery trace: {trace}")
        canonical_deadline = time.monotonic() + 8
        prompt_count = -1
        canonical_text = ""
        while time.monotonic() < canonical_deadline:
            payload = h.canonical_payload(cid)
            mapping = payload.get("mapping") if isinstance(payload, dict) else {}
            current = payload.get("current_node") if isinstance(payload, dict) else None
            current_node = (
                mapping.get(current)
                if isinstance(mapping, dict) and isinstance(current, str)
                else None
            )
            message = (
                current_node.get("message")
                if isinstance(current_node, dict)
                else None
            )
            content = message.get("content") if isinstance(message, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                parts = []
            canonical_text = "".join(
                part for part in parts if isinstance(part, str)
            ).strip()
            prompt_count = 0
            if isinstance(mapping, dict):
                for node in mapping.values():
                    msg = node.get("message") if isinstance(node, dict) else None
                    author = msg.get("author") if isinstance(msg, dict) else None
                    body = msg.get("content") if isinstance(msg, dict) else None
                    msg_parts = body.get("parts") if isinstance(body, dict) else None
                    if not isinstance(msg_parts, list):
                        msg_parts = []
                    msg_text = "".join(
                        part for part in msg_parts if isinstance(part, str)
                    )
                    if (
                        isinstance(author, dict)
                        and author.get("role") == "user"
                        and msg_text == prompt
                    ):
                        prompt_count += 1
            if canonical_text == expected and prompt_count == 1:
                break
            time.sleep(0.25)
        if canonical_text != expected:
            raise AssertionError(
                f"fault recovery canonical final mismatch: {canonical_text!r}"
            )
        if prompt_count != 1:
            raise AssertionError(
                f"pre-submit recovery duplicated user prompt: count={prompt_count}"
            )
        assert_no_ui_transport_errors(session)
        network = h.assert_network_clean(expected_posts=1)
        return {
            "ok": True,
            "baseline": seed,
            "turn": {
                "marker": expected,
                "seconds": round(completed - started, 3),
                "conversation_id": cid,
            },
            "prompt_count": prompt_count,
            "fault_trace": trace,
            "network": network,
        }
    finally:
        assert_clean_exit(session.close(), "fault-pre-submit")


SCENARIOS: dict[str, Callable[[LiveHarness], dict[str, Any]]] = {
    "fresh_sequence": scenario_fresh_sequence,
    "resume_new": scenario_resume_new,
    "tool_heavy": scenario_tool_heavy,
    "attached_follow_order": scenario_attached_follow_order,
    "own_topic_loss_recovery": scenario_own_topic_loss_recovery,
    "idle_topic_reconnect": scenario_idle_topic_reconnect,
    "attached_topic_loss_recovery": scenario_attached_topic_loss_recovery,
    "external_attach_midturn_stall_recovery": scenario_external_attach_midturn_stall_recovery,
    "tool_heavy_full_final": scenario_tool_heavy_full_final,
    "long_seed": scenario_long_seed,
    "long_restart": scenario_long_restart,
    "stop_next": scenario_stop_next,
    "temporary_first": scenario_temporary_first,
    "temporary": scenario_temporary,
    "image": scenario_image,
    "goal": scenario_goal,
    "parallel5": scenario_parallel5,
    "fault_seed": scenario_fault_seed,
    "fault_pre_submit": scenario_fault_pre_submit,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hard live E2E acceptance for the installed gptty TUI."
    )
    parser.add_argument(
        "scenario",
        choices=[*SCENARIOS, "all"],
        help="Scenario to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".ai-bridge/live-acceptance"),
        help="Directory for logs and JSON reports.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round number included in output paths.",
    )
    args = parser.parse_args()

    if not GPTTY.exists():
        raise SystemExit(f"installed gptty not found at {GPTTY}")
    if not AUTH.exists():
        raise SystemExit(f"ChatGPT auth profile not found at {AUTH}")

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    overall: dict[str, Any] = {
        "round": args.round,
        "scenarios": {},
        "started_at": time.time(),
    }
    ok = True
    for name in names:
        root = args.output / f"round-{args.round}" / name
        shutil.rmtree(root, ignore_errors=True)
        harness = LiveHarness(root, scenario=name)
        started = time.monotonic()
        try:
            result = SCENARIOS[name](harness)
            result["seconds"] = round(time.monotonic() - started, 3)
            overall["scenarios"][name] = result
            print(f"PASS {name} {result['seconds']}s", flush=True)
        except Exception as exc:
            ok = False
            overall["scenarios"][name] = {
                "ok": False,
                "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"FAIL {name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            (root / "result.json").write_text(
                json.dumps(
                    overall["scenarios"].get(name, {"ok": False}),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            shutil.rmtree(harness.broker_dir, ignore_errors=True)
        if not overall["scenarios"][name].get("ok") and args.scenario != "all":
            break

    overall["ok"] = ok and all(
        bool(item.get("ok")) for item in overall["scenarios"].values()
    )
    overall["finished_at"] = time.time()
    summary_path = args.output / f"round-{args.round}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0 if overall["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
