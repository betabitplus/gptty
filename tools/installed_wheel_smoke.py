from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True):
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", default="dist")
    parser.add_argument("--expected-version", default="0.1.2")
    parser.add_argument("--find-links", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / args.wheel_dir).glob("gptty_web-*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"expected one gptty wheel, found {len(wheels)}")
    wheel = wheels[0].resolve()

    with tempfile.TemporaryDirectory(prefix="gptty-wheel-smoke-") as tmp:
        tmp_path = Path(tmp)
        env_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
        if os.name == "nt":
            python = env_dir / "Scripts" / "python.exe"
            bin_dir = env_dir / "Scripts"
        else:
            python = env_dir / "bin" / "python"
            bin_dir = env_dir / "bin"

        command = [str(python), "-m", "pip", "install"]
        for directory in args.find_links:
            command.extend(["--find-links", str(Path(directory).resolve())])
        command.append(str(wheel))
        _run(command, cwd=tmp_path)
        _run([str(python), "-m", "pip", "check"], cwd=tmp_path)

        probe = (
            "import importlib.metadata,json,shutil\n"
            "import gptty.exporter_bridge as bridge\n"
            "from chatgpt_web_adapter.artifact_manifest import "
            "CANONICAL_VISIBLE_GRAPH_REPRESENTATION,VISIBLE_GRAPH_ARTIFACT_KIND\n"
            "print(json.dumps({"
            "'gptty':importlib.metadata.version('gptty-web'),"
            "'cwa':importlib.metadata.version('chatgpt-web-adapter'),"
            "'exporter':importlib.metadata.version('chatgpt-conversation-exporter'),"
            "'exporter_command':bridge.resolve_exporter_command()[0],"
            "'which_exporter':shutil.which('chatgpt-export-one'),"
            "'representation':CANONICAL_VISIBLE_GRAPH_REPRESENTATION,"
            "'artifact_kind':VISIBLE_GRAPH_ARTIFACT_KIND"
            "}))\n"
        )
        data = json.loads(_run([str(python), "-c", probe], cwd=tmp_path).stdout)
        if data["gptty"] != args.expected_version:
            raise AssertionError(data)
        if data["representation"] != "canonical_visible_graph":
            raise AssertionError(data)
        if not data["which_exporter"]:
            raise AssertionError("chatgpt-export-one is not installed")

        gptty = str(bin_dir / "gptty")
        exporter = str(bin_dir / "chatgpt-export-one")
        version_text = _run([gptty, "--version"], cwd=tmp_path).stdout.strip()
        _run([gptty, "export", "--help"], cwd=tmp_path)
        _run([exporter, "--help"], cwd=tmp_path)
        auth = _run(
            [gptty, "auth", "status", "--auth", str(tmp_path / "missing-auth.json")],
            cwd=tmp_path,
            check=False,
        )
        if auth.returncode not in {0, 1}:
            raise AssertionError(f"auth status smoke exited {auth.returncode}: {auth.stderr}")

    result = {
        "ok": True,
        "schema": 1,
        "gptty_version": data["gptty"],
        "cwa_version": data["cwa"],
        "exporter_version": data["exporter"],
        "artifact_kind": data["artifact_kind"],
        "representation": data["representation"],
        "version_output": version_text,
    }
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else "installed gptty wheel smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
