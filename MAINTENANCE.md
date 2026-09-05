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

Use this repository only for terminal commands, local state/locks, rendering, concise live-output formatting, and CLI ergonomics. Do not duplicate CWA browser logic here. Conversation/model catalogs and canonical history/status belong to CWA's authenticated browser-context read plane; gptty must not duplicate backend endpoints, auth, or a local conversation index as a competing source of truth. The no-override text send default is CWA's native `DEEP` profile (product `HIGH`), not a hard-coded model slug. CWA intentionally fails closed when attachment paths and the profile selector are combined; therefore default rich-input turns must resolve the live catalog once per client session and choose the strongest normal non-Work, non-mini thinking frontier slug instead of weakening the High policy or changing CWA's proven composition guard. Explicit `/model <slug>` overrides both paths; `/model default` restores the latest-frontier High policy. `/resume` may follow an already-active selected chat with one canonical snapshot per 15 seconds, but only while that chat is active; idle UI must not poll. Compatibility-only canonical status waits must stay low-frequency; the adapter clamps them to a 15-second minimum poll interval instead of repeatedly hitting ChatGPT.

The downstream interactive UI is intentionally isolated under `src/gptty/ui/`: `prompt_toolkit` owns input/history/menus and Rich owns line-oriented presentation. Keep normal terminal scrollback; successful `/new`, `/resume`, and `/detach` may clear the current viewport (`CSI 2 J` + home) to establish a fresh visual context, but must not erase scrollback or switch to an alternate screen. Attached-chat headers must expose the full canonical ChatGPT URL. Pending image attachments are one-turn local UI state: file paths stay user-owned, clipboard images are materialized with native macOS `osascript` pasteboard PNG coercion into a session temp directory, and only gptty-owned clipboard files may be deleted. Context switches clear pending attachments to prevent cross-chat sends. Do not add clipboard polling or require `pngpaste`/Homebrew while the native path works. The generation timer is presentation-only: one Rich live line may refresh locally once per second while a turn/follow is active, with no added ChatGPT network polling. During an enhanced turn `Ctrl-C` means remote Stop generating, not local SDK-stack interruption: the blocking CWA send stays in a worker thread, the main TUI thread sends CWA's out-of-band stop request, then waits for the original send/readback path to reconcile before rendering the partial answer. `/stop` uses the same CWA primitive. `Ctrl-\\` is the explicit local-only escape hatch: it exits gptty without sending Stop, allowing the browser/native turn to continue; if requested before `browser_native_write_completed`, process exit must be deferred until that safe handoff event arrives. Do not duplicate DOM stop selectors or browser cancellation logic in gptty. Completion notification is best-effort local macOS `osascript`; notification failure must never fail a chat turn, and user-stopped turns must not generate a completion notification. Do not introduce an alternate-screen/full-screen TUI unless there is a separately proven need. The non-TTY and `--plain` paths must remain compatible with upstream CLI/scripting behavior. If upstream gains equivalent input, palette, or rendering features, prefer upstream and remove the replaced downstream code instead of maintaining two implementations.

## Adding features

Create a short-lived branch from `main`, implement the smallest change in the correct repository, test it, then merge to downstream `main`. If upstream later implements an equivalent feature, prefer upstream and delete the downstream duplicate.
