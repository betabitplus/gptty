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
uv tool install --force ../chatgpt-web-adapter
uv tool install --force . --with ../chatgpt-web-adapter
```

Then run a real `gptty send --new ...` smoke before considering browser-facing changes done.

## Responsibility boundary

`gptty` is a thin terminal client. Keep ChatGPT Web transport, browser extension/native host behavior, passive stream observation, session handling, retries, and finality in `chatgpt-web-adapter`.

Use this repository only for terminal commands, local state/locks, rendering, concise live-output formatting, and CLI ergonomics. Do not duplicate CWA browser logic here. Conversation/model catalogs and canonical history/status belong to CWA's authenticated browser-context read plane; gptty must not duplicate backend endpoints, auth, or a local conversation index as a competing source of truth. `/resume` may follow an already-active selected chat with one canonical snapshot per 15 seconds, but only while that chat is active; idle UI must not poll. Compatibility-only canonical status waits must stay low-frequency; the adapter clamps them to a 15-second minimum poll interval instead of repeatedly hitting ChatGPT.

The downstream interactive UI is intentionally isolated under `src/gptty/ui/`: `prompt_toolkit` owns input/history/menus and Rich owns line-oriented presentation. Keep normal terminal scrollback; do not introduce an alternate-screen/full-screen TUI unless there is a separately proven need. The non-TTY and `--plain` paths must remain compatible with upstream CLI/scripting behavior. If upstream gains equivalent input, palette, or rendering features, prefer upstream and remove the replaced downstream code instead of maintaining two implementations.

## Adding features

Create a short-lived branch from `main`, implement the smallest change in the correct repository, test it, then merge to downstream `main`. If upstream later implements an equivalent feature, prefer upstream and delete the downstream duplicate.
