from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from gptty.sdk_client import GpttyClient


ACTIVE_STREAM_DIR = (
    Path.home() / "Library/Application Support/chatgpt-web-adapter/active-streams"
)
AUTH_FILE = Path.home() / ".local/share/gptty/profiles/chatgpt-web/auth_data.json"
CMUX = Path("/Applications/cmux.app/Contents/Resources/bin/cmux")
UNFINISHED = {
    "running",
    "tool_running",
    "streaming",
    "in_progress",
    "awaiting_tool_approval",
}


def _conversation_id(value: str) -> str:
    candidate = value.strip().rstrip("/")
    if "/c/" in candidate:
        candidate = candidate.rsplit("/c/", 1)[1]
    if not candidate:
        raise ValueError("conversation id is empty")
    return candidate


def _read_registry(conversation_id: str) -> dict[str, Any] | None:
    path = ACTIVE_STREAM_DIR / f"{conversation_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    saved_at = value.get("saved_at")
    if isinstance(saved_at, (int, float)) and not isinstance(saved_at, bool):
        value["saved_age_seconds"] = max(0.0, time.time() - float(saved_at))
    return value


def _process_info(pid: Any) -> dict[str, Any] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,etime=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return {"pid": pid, "alive": False}
    return {"pid": pid, "alive": True, "ps": text}


def _terminal_snapshot(pid: Any, *, lines: int) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"available": False, "reason": "no active-stream pid"}
    env_row = subprocess.run(
        ["ps", "eww", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    env = os.environ.copy()
    for key in (
        "CMUX_SOCKET_CAPABILITY",
        "CMUX_SOCKET_PATH",
        "CMUX_WORKSPACE_ID",
        "CMUX_SURFACE_ID",
    ):
        match = re.search(r"(?:^| )" + re.escape(key) + r"=([^ ]*)", env_row)
        if match:
            env[key] = match.group(1)
    surface = env.get("CMUX_SURFACE_ID")
    if not surface or not CMUX.exists():
        return {"available": False, "reason": "cmux surface unavailable"}
    proc = subprocess.run(
        [
            str(CMUX),
            "read-screen",
            "--surface",
            surface,
            "--scrollback",
            "--lines",
            str(lines),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": proc.stderr.strip() or f"cmux exit {proc.returncode}",
        }
    text = proc.stdout
    status_lines = [
        line.strip()
        for line in text.splitlines()
        if any(
            marker in line
            for marker in (
                "working ",
                "STALLED ",
                "server quiet",
                "reconnecting delivery",
                "terminal proof",
            )
        )
    ]
    return {
        "available": True,
        "surface": surface,
        "tail_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "last_status": status_lines[-1] if status_lines else None,
        "tail": "\n".join(text.splitlines()[-lines:]),
    }


def _decode_agent_browser_output(stdout: str) -> Any:
    value: Any = stdout.strip()
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def _agent_browser_reload() -> dict[str, Any]:
    proc = subprocess.run(
        ["agent-browser", "reload"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def _browser_snapshot(conversation_id: str) -> dict[str, Any]:
    script = r'''JSON.stringify((()=>{
const turns=Array.from(document.querySelectorAll("[data-testid*=conversation-turn]"));
const latest=turns[turns.length-1];
const buttons=Array.from(document.querySelectorAll("button"));
return {
  url: location.href,
  turn_count: turns.length,
  stop_answering: buttons.some(b=>(b.getAttribute("aria-label")||"")==="Stop answering"),
  latest_text: latest ? (latest.innerText||"").slice(-5000) : ""
};
})())'''
    proc = subprocess.run(
        ["agent-browser", "eval", script],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": proc.stderr.strip() or f"agent-browser exit {proc.returncode}",
        }
    value = _decode_agent_browser_output(proc.stdout)
    if not isinstance(value, dict):
        return {"available": False, "reason": "agent-browser returned invalid snapshot"}
    url = str(value.get("url") or "")
    latest_text = str(value.get("latest_text") or "")
    return {
        "available": True,
        "url": url,
        "target_open": conversation_id in url,
        "turn_count": value.get("turn_count"),
        "stop_answering": bool(value.get("stop_answering")),
        "delivery_timeout": "Message delivery timed out" in latest_text,
        "retry_visible": "Retry" in latest_text,
        "latest_text_tail": latest_text[-1800:],
    }


def _server_snapshot(conversation_id: str) -> dict[str, Any]:
    client = GpttyClient(
        auth_file=AUTH_FILE,
        timeout=30,
        browser_authority_backend="wkwebview",
    )
    snapshot = client._client.runtime.conversation_follow_snapshot(
        conversation_id,
        emitted_message_ids=(),
        limit=32,
    )
    status_value = snapshot.get("status")
    status = getattr(status_value, "status", None)
    return {
        "status": status,
        "unfinished": status in UNFINISHED,
        "topic_id": snapshot.get("stream_topic_id"),
        "turn_exchange_id": snapshot.get("turn_exchange_id"),
        "answer_message_id": snapshot.get("stream_answer_message_id"),
        "answer_text_length": len(snapshot.get("stream_answer_text") or ""),
        "current_turn_event_count": len(snapshot.get("current_turn_event_ids") or ()),
    }


def _classify(report: dict[str, Any]) -> list[str]:
    server = report.get("server") or {}
    browser = report.get("browser") or {}
    registry = report.get("active_stream") or {}
    process = report.get("process") or {}

    findings: list[str] = []
    if browser.get("delivery_timeout") and server.get("unfinished"):
        findings.append("browser_delivery_timed_out_server_continues")
    elif browser.get("delivery_timeout") and not server.get("unfinished"):
        findings.append("browser_stale_after_server_terminal")

    if server.get("unfinished") and not registry:
        findings.append("server_running_without_local_follower")
    elif registry and not process.get("alive"):
        findings.append("stale_active_stream_registry")

    if (
        browser.get("target_open")
        and server.get("turn_exchange_id")
        and browser.get("delivery_timeout")
        and server.get("current_turn_event_count", 0)
    ):
        findings.append("browser_server_timeline_diverged")

    terminal = report.get("terminal") or {}
    status = str(terminal.get("last_status") or "")
    if "STALLED finality" in status:
        findings.append("terminal_finality_stalled")
    elif "STALLED backend" in status:
        findings.append("terminal_backend_stalled")

    return findings or ["no_known_stream_health_fault_detected"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only browser/server/terminal health probe for one ChatGPT conversation."
    )
    parser.add_argument("conversation")
    parser.add_argument("--terminal-lines", type=int, default=80)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--repair-browser",
        action="store_true",
        help=(
            "If the target tab shows Message delivery timed out while the server "
            "turn is still unfinished, reload that tab and re-check. Never clicks "
            "Retry and never sends a chat write."
        ),
    )
    args = parser.parse_args()

    conversation_id = _conversation_id(args.conversation)
    report: dict[str, Any] = {
        "conversation_id": conversation_id,
        "observed_at_epoch": time.time(),
        "read_only": True,
    }
    try:
        report["server"] = _server_snapshot(conversation_id)
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        report["server"] = {"error": f"{type(exc).__name__}: {exc}"}

    registry = _read_registry(conversation_id)
    report["active_stream"] = registry
    pid = registry.get("pid") if isinstance(registry, dict) else None
    report["process"] = _process_info(pid)
    report["terminal"] = _terminal_snapshot(pid, lines=max(20, args.terminal_lines))

    if args.no_browser:
        report["browser"] = {"available": False, "reason": "disabled"}
    else:
        try:
            report["browser"] = _browser_snapshot(conversation_id)
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            report["browser"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    report["findings"] = _classify(report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
