from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path

PROJECT_NAME = "gptty-web"
DIST_STEM = "gptty_web"
TAG_PREFIX = "gptty-web-v"
CWA_REQ = "chatgpt-web-adapter>=0.3.2,<0.4.0"
EXPORTER_REQ = "chatgpt-conversation-exporter>=0.1.0,<0.2.0"
REQUIRED_WHEEL_FILES = {
    "gptty/exporter_bridge.py",
    "gptty/sdk_client.py",
    "gptty/commands/export.py",
    "gptty/ui/session.py",
}


def _project(root: Path) -> dict[str, object]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    if match is None:
        raise AssertionError("missing [project]")
    block = match.group(1)

    def scalar(name: str) -> str:
        value = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"([^"]+)"\s*$', block)
        if value is None:
            raise AssertionError(f"missing project.{name}")
        return value.group(1)

    deps = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)\]\s*$", block)
    dependencies = re.findall(r'"([^"]+)"', deps.group(1)) if deps else []
    return {"name": scalar("name"), "version": scalar("version"), "dependencies": dependencies}


def _dated_changelog(root: Path, version: str) -> str:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^##\s+{re.escape(version)}\s+-\s+(\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"CHANGELOG.md lacks dated {version} heading")
    return match.group(1)


def _metadata(path: Path) -> email.message.Message:
    with zipfile.ZipFile(path) as zf:
        name = next(item for item in zf.namelist() if item.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(zf.read(name))


def _entry_points(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        name = next(item for item in zf.namelist() if item.endswith(".dist-info/entry_points.txt"))
        text = zf.read(name).decode("utf-8")
    result: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section == "console_scripts" and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirement_present(requirements: list[str], name: str, lower: str, upper: str) -> bool:
    for requirement in requirements:
        compact = requirement.replace(" ", "")
        if compact.startswith(name) and lower in compact and upper in compact:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--tag")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project = _project(root)
    version = str(project["version"])
    if project["name"] != PROJECT_NAME:
        raise AssertionError(project)
    dependencies = [str(item) for item in project["dependencies"]]
    for required in (CWA_REQ, EXPORTER_REQ):
        if required not in dependencies:
            raise AssertionError(f"missing exact dependency: {required}")

    if args.tag is not None:
        if not args.tag.startswith(TAG_PREFIX):
            raise AssertionError(f"release tag must start with {TAG_PREFIX!r}")
        if args.tag[len(TAG_PREFIX):] != version:
            raise AssertionError("tag/version mismatch")

    release_date = _dated_changelog(root, version)
    dist = (root / args.dist_dir).resolve()
    wheel = dist / f"{DIST_STEM}-{version}-py3-none-any.whl"
    sdist = dist / f"{DIST_STEM}-{version}.tar.gz"
    distributables = {
        path.resolve()
        for path in dist.glob("*")
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    }
    if distributables != {wheel.resolve(), sdist.resolve()}:
        raise AssertionError(f"unexpected dist contents: {[p.name for p in sorted(distributables)]}")

    metadata = _metadata(wheel)
    if metadata["Name"] != PROJECT_NAME or metadata["Version"] != version:
        raise AssertionError("wheel metadata identity mismatch")
    requirements = metadata.get_all("Requires-Dist") or []
    if not _requirement_present(requirements, "chatgpt-web-adapter", ">=0.3.2", "<0.4.0"):
        raise AssertionError(f"CWA requirement drifted: {requirements!r}")
    if not _requirement_present(
        requirements, "chatgpt-conversation-exporter", ">=0.1.0", "<0.2.0"
    ):
        raise AssertionError(f"exporter requirement drifted: {requirements!r}")

    if _entry_points(wheel) != {"gptty": "gptty.cli:main"}:
        raise AssertionError("console entry point drifted")

    with zipfile.ZipFile(wheel) as zf:
        files = set(zf.namelist())
    missing = sorted(REQUIRED_WHEEL_FILES - files)
    if missing:
        raise AssertionError(f"wheel missing runtime files: {missing!r}")

    with tarfile.open(sdist, "r:gz") as tf:
        sdist_files = {member.name for member in tf.getmembers() if member.isfile()}
    prefix = f"{DIST_STEM}-{version}/"
    required_sdist = ("pyproject.toml", "README.md", "CHANGELOG.md", "docs/release.md")
    missing_sdist = [item for item in required_sdist if prefix + item not in sdist_files]
    if missing_sdist:
        raise AssertionError(f"sdist missing files: {missing_sdist!r}")

    result = {
        "ok": True,
        "schema": 1,
        "project": PROJECT_NAME,
        "version": version,
        "tag": args.tag,
        "changelog_release_date": release_date,
        "dependencies": {"cwa": CWA_REQ, "exporter": EXPORTER_REQ},
        "artifacts": {
            "wheel": {"filename": wheel.name, "sha256": _sha256(wheel)},
            "sdist": {"filename": sdist.name, "sha256": _sha256(sdist)},
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"{PROJECT_NAME} {version} release gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
