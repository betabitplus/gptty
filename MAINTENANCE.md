# Downstream maintenance

This checkout is the maintained Betabit fork of `gptty`.

- `origin`: `https://github.com/betabitplus/gptty.git` — writable downstream fork.
- `upstream`: `https://github.com/kymuco/gptty.git` — original repository; fetch only.
- `main`: the maintained downstream branch.

## Updating from upstream

```bash
git fetch upstream
git switch main
git merge upstream/main
HOME=/tmp/gptty-test-home PYTHONPATH=src python -m pytest -q
git push origin main
```

After CWA changes, reinstall the local pair used for live testing:

```bash
uv tool install --force ../chatgpt-web-adapter-cwa
uv tool install --force . --with ../chatgpt-web-adapter-cwa
```

Then run a real `gptty send --new ...` smoke before considering browser-facing changes done.

## Responsibility boundary

`gptty` is a thin terminal client. Keep ChatGPT Web transport, browser extension/native host behavior, passive stream observation, session handling, retries, and finality in `chatgpt-web-adapter`.

Use this repository only for terminal commands, local state/locks, rendering, concise live-output formatting, and CLI ergonomics. Do not duplicate CWA browser logic here.

## Adding features

Create a short-lived branch from `main`, implement the smallest change in the correct repository, test it, then merge to downstream `main`. If upstream later implements an equivalent feature, prefer upstream and delete the downstream duplicate.
