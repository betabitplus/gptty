# Release Checklist

This checklist is for publishing `gptty-web` to PyPI.

## Release train

`gptty-web 0.1.2` has two non-optional upstream release prerequisites:

1. publish and post-publish smoke `chatgpt-web-adapter 0.3.2`;
2. publish and post-publish smoke `chatgpt-conversation-exporter 0.1.0`;
3. only then publish `gptty-web 0.1.2`.

Do not lower these floors or fall back to PATH-only discovery to make a release install succeed.

## Before tagging

1. Confirm `pyproject.toml` version and dated `CHANGELOG.md` entry.
2. Confirm CI is green on the exact release commit.
3. Confirm public package indexes contain both runtime dependency floors above.
4. Build and validate exact artifacts:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   python tools/release_gate.py --dist-dir dist --tag gptty-web-v0.1.2
   ```

5. Smoke the exact wheel in a clean environment with dependencies resolved from the intended source. Before upstream publication, local rehearsal may use `--find-links` pointing at the exact CWA/exporter wheelhouse; tagged publication must resolve the published packages normally.

   ```bash
   python tools/installed_wheel_smoke.py --wheel-dir dist --expected-version 0.1.2
   ```

The smoke verifies installed package metadata, CWA visible-graph constants, the exporter console command, `gptty --version`, `gptty export --help`, and `pip check` outside the source checkout.

## PyPI Trusted Publishing

The publish workflow expects:

- repository: `kymuco/gptty`
- workflow: `publish.yml`
- environment: `pypi`
- package: `gptty-web`
- release tag: `gptty-web-vX.Y.Z`

Publishing must stop if dependency resolution, the tagged release contract, or installed-wheel smoke fails.

## Post-release smoke

Install from PyPI in a new environment, without checkout paths:

```bash
python -m pip install gptty-web==0.1.2
python -m pip check
gptty --version
chatgpt-export-one --help
gptty export --help
gptty auth status
```

Use real ChatGPT authorization only locally. Product writes are not required merely because packaging/version metadata changed; reuse the accepted current WK live/resource/network gate unless runtime code changed.
