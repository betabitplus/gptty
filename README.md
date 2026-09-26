# gptty

English version. Russian version: [README.ru.md](README.ru.md)

Terminal client for existing ChatGPT web sessions.

> [!WARNING]
> Not the official OpenAI API.
> Uses an existing ChatGPT web session.
> Web backend behavior may change.

`gptty` is the successor to `webchat-openai-cli`. The project is being migrated from a standalone script into a terminal-native product powered by [`chatgpt-web-adapter`](https://github.com/kymuco/chatgpt-web-adapter).

The package distribution name is `gptty-web` because the PyPI name `gptty` is already occupied. The installed command is still `gptty`.

## Product Direction

```text
SDK = chatgpt-web-adapter
CLI = gptty
```

`gptty` is intended for terminal workflows:

```bash
gptty chat
gptty auth status
gptty auth refresh --mode wait
gptty ask "explain this error"
gptty ask --image screenshot.png "describe this UI"
{ printf 'Review this patch:\n\n'; git diff; } | gptty ask
gptty attach https://chatgpt.com/c/...
gptty send "continue from here"
gptty messages --last 5 --format markdown
gptty status --format json
gptty export --format markdown --output conversation.md
```

`gptty ask`, `gptty send`, the default `gptty chat` path, conversation inspection commands, and conversation export are SDK-backed. The legacy interactive runtime remains available through `gptty chat --legacy` while feature parity is migrated in later PRs.

## Current Features

- minimal SDK-backed interactive chat through `gptty chat`
- inspect auth data through `gptty auth status`
- refresh `auth_data.json` through `gptty auth refresh`
- attach existing conversations through `gptty attach`
- send prompts to attached, explicit, or new conversations through `gptty send`
- SDK-backed image and general-file prompts through `gptty ask/send --image` and `--file`
- typed ChatGPT connector/required-action lifecycle observations in terminal and JSONL, with approval/execution kept fail-closed to ChatGPT web
- inspect attached or explicit conversations through `gptty messages` and `gptty status`
- export attached or explicit conversations through `gptty export`
- output modes for `messages`, `status`, `send`, and `export`: `plain`, `json`, `markdown`; `ask` and `send` also provide versioned `jsonl` automation streams
- legacy interactive chat fallback through `gptty chat --legacy`
- one-shot SDK-backed prompts through `gptty ask`
- bounded text stdin for scripting (4 MiB by default, configurable with `--stdin-max-bytes`), with NUL/binary-looking input rejected before a write
- transparent pipe-friendly prompts: stdin and positional prompt text are never silently rewritten or combined
- streaming replies in the terminal
- transactional local session state in `local-state.sqlite3` (SQLite/WAL), shared with local run/TUI/delivery evidence
- independent interactive runtime sessions by default; use `--session NAME` (or `GPTTY_SESSION_ID`) only when you intentionally want several commands/processes to reuse one local conversation/model/effort selection
- one-time migration from older `gptty_state.json` / `gptty_state.session-*.json` files; those JSON files are no longer runtime authority after import
- legacy state file for `--legacy`: `webchat_state.json`
- transactional/atomic local-state writes and atomic `auth_data.json` updates
- legacy image prompts through `/img` in `gptty chat --legacy`
- `auto` and `wait` auth capture modes
- English and Russian CLI localization in the legacy runtime

## Requirements

- Python 3.10+
- system `curl` available in `PATH`
- Chrome or Chromium for auth capture
- valid `auth_data.json` for an existing ChatGPT web session

## Installation

Base install:

```bash
python -m pip install gptty-web
```

Install with browser auth-capture dependencies:

```bash
python -m pip install "gptty-web[auth]"
```

From checkout for development:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[auth,test]"
```

On Windows `cmd.exe`:

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[auth,test]"
```

## Reusable ChatGPT authorization

Check the current reusable auth state without opening a browser:

```bash
gptty auth status
```

Use JSON output for scripts:

```bash
gptty auth status --format json
```

Refresh auth data through the CLI wrapper:

```bash
gptty auth refresh --mode wait
```

Migrate or remove reusable local authorization explicitly:

```bash
gptty auth migrate --backend keyring
gptty auth migrate --backend file
gptty auth logout
```

Fast mode for an already logged-in browser session:

```bash
gptty auth refresh --mode auto
```

In `wait` mode the browser stays open until the chat is ready. After that, send any message manually in the browser to trigger auth capture.

Optional: override the one-shot probe prompt used by `auto` mode:

```bash
gptty auth refresh --mode auto --probe-prompt "Ping"
```

The legacy script entrypoints remain available from a checkout:

```cmd
venv\Scripts\python.exe auth_fetcher.py --mode wait
venv\Scripts\python.exe auth_fetcher_wait.py
```

With `gptty-web[auth]`, reusable authorization is persisted through CWA's credential-store authority. A recognized OS-backed secure `keyring` provider is preferred; plaintext/null keyring fallbacks are rejected. `auth_data.json` then contains only non-secret backend metadata/expiry hints. If no secure OS store is available, CWA retains the hardened owner-only file fallback. Use `gptty auth migrate --backend keyring` or `--backend file` to change an existing backend explicitly, and `gptty auth logout` to remove reusable local authorization. See [docs/auth.md](docs/auth.md) for lifecycle and failure semantics.

## Run the CLI

Attach an existing ChatGPT conversation:

```bash
gptty attach https://chatgpt.com/c/...
```

Send a prompt to the attached conversation:

```bash
gptty send "continue from here"
```

Pipe an exact prompt into the attached conversation:

```bash
{ printf 'Review this patch:\n\n'; git diff; } | gptty send
```

stdin and positional prompt text are mutually exclusive. If a pipe is present but should be ignored, use `--no-stdin`; if a custom safety bound is needed, use `--stdin-max-bytes BYTES`.

Send to an explicit conversation without changing the currently attached local session:

```bash
gptty send --to https://chatgpt.com/c/... "continue there"
```

Start a new conversation and store its returned conversation reference in the current transactional local session:

```bash
gptty send --new "start a new conversation"
```

Send image prompts through the SDK-backed commands:

```bash
gptty ask --image screenshot.png "describe this UI"
gptty ask --image https://example.com/chart.png "summarize this chart"
gptty send --image diagram.webp "continue with this image"
gptty send --image before.png --image after.png "compare these images"
gptty ask --file notes.pdf "summarize this document"
gptty send --file data.csv "continue using this file"
```

`--image` and `--file` use the same CWA rich-input media contract and accept local file paths, `http(s)` URLs, and data URIs. Both may be repeated or mixed; their CLI order is preserved. `--image` is the image-oriented convenience surface (PNG, JPEG/JPG, GIF, WebP), while `--file` exposes CWA's live-proven general-file path rather than a separate uploader.

Inspect the attached conversation:

```bash
gptty messages --last 5
gptty status
```

Use JSON or Markdown output for scripts and exports, or versioned JSONL for streaming automation:

```bash
gptty messages --last 5 --format json
gptty messages --last 5 --format markdown
gptty status --format json
gptty send --format json "summarize the current thread"
gptty send --format jsonl "summarize the current thread"
gptty ask --format jsonl "explain this error"
gptty export --format markdown --output conversation.md
gptty export --format json --output conversation.json
```

`gptty send --format json` returns a rich versioned final record with conversation/message identity, model and effort provenance, finality and observations. `--format jsonl` emits one JSON object per event and ends with the same `gptty.turn.result` contract. Typed connector lifecycle events preserve stable connector/activity/action ids and phases when ChatGPT exposes them; visible authorization cards without a stable action id remain point evidence only. gptty does **not** infer connector approval from labels or prose and currently provides no local approve/deny action: when ChatGPT requires authorization, complete it in ChatGPT web. Markdown remains a human-readable final-text surface and is non-streaming.

You can also inspect or export an explicit conversation without attaching it:

```bash
gptty messages https://chatgpt.com/c/... --last 5
gptty status https://chatgpt.com/c/...
gptty export https://chatgpt.com/c/... --last 20 --output conversation.md
```

`gptty export` defaults to Markdown output. When `--output` points to an existing file, add `--overwrite` to replace it.

Interactive chat:

```bash
gptty chat
```

In a TTY, press `/` and Enter to open the lightweight action menu:

```text
/new
/temporary
/resume
/detach
/reload
/stop
/queue
/history
/goal
/export
/image
/file
/paste
/model
/effort
/exit
```

`/image <path>` stages an image and `/file <path>` stages a general file for the next accepted prompt; either command without an argument opens the same persistent path picker with the appropriate label, so files can be dragged from Finder into the terminal. Repeat or mix them for multiple attachments. `/image clear` removes pending images only; `/file clear` removes pending general files only. `/paste` materializes the current macOS clipboard image as a temporary PNG using the native pasteboard (`osascript`). The composer shows a generic `[N attachments]` marker. Successful `/new`, `/temporary`, `/resume`, or `/detach` clears still-unbound attachments to prevent accidental cross-chat sends. When text is queued while another turn/resume/follow is active, that queued turn immediately snapshots the current media list, so later `/image`, `/file`, or `/paste` staging belongs to a later prompt rather than drifting onto an older queued prompt. `/reload` preserves still-unbound attachments. Clipboard temp files are removed after their bound prompt finishes, when the queued item is removed/cleared, or when the session/context is closed.

Queued turns in the enhanced UI are deliberately **memory-only** and bounded. Each accepted queued turn freezes its text, media list, conversation mode/reference, model/reasoning policy slot, Goal id/generation, origin and timestamp. `/queue` shows content-free queue metadata; `/queue remove <index|id>` removes one item, `/queue clear` discards all queued turns, and `/queue send` explicitly releases/rebinds held turns to the current context. Normal completion releases compatible queued turns FIFO. A user Stop (`/stop`, Ctrl-C or SIGINT), an abnormal terminal turn, a failed resume, or a chat/model/Goal binding mismatch holds the queue instead of auto-sending or silently discarding drafts. Held work stays local until explicit `/queue send` or `/queue clear`; process exit still loses it by design rather than pretending the queue is durable.

`/temporary` starts a fresh real ChatGPT Temporary Chat using CWA's live temporary lifecycle. Continuation is supported while that same gptty process/runtime remains active; the temporary conversation id is deliberately not persisted or treated as resumable authority. Temporary prompts are not written to persistent prompt history and are removed from normal in-process history when Temporary mode ends. `/new`, `/resume`, `/detach`, and normal `/exit` end the live temporary lifecycle. `/export` writes the entire currently attached user-visible conversation to a new Markdown file under `~/Documents/gptty-exports/` and immediately prints the absolute path. Normal conversations are exported from CWA's canonical history; the currently attached Temporary Chat is exported from the live transcript held by this gptty session. Existing export files are never overwritten.

Enhanced normal chats are also self-archiving. Every prompt actually submitted through gptty and every answer observed for that turn is appended under `~/.local/share/gptty/chat-archive/conversations/<conversation-id>/events.jsonl`; `transcript.md` is a readable projection and `meta.json` records the scope. This ledger is deliberately `tui-observed`: `/resume` and `/reload` do not import web-only history into it, Temporary Chat is excluded, and a later ChatGPT web snapshot cannot erase an event that gptty already observed. `GPTTY_ARCHIVE_HOME` can override the archive root.

Local retention is explicit and privacy-safe. `gptty privacy status` prints content-free counts for local run/archive/pending/export surfaces. `gptty privacy prune --older-than-days 30` removes old completed run records/projections and orphan pending prompts; add `--include-archives` to remove old local TUI conversation copies and `--include-exports` to remove only timestamped files created in gptty's default export directory. Runs still marked `running`, recent data, and files written to an explicit user-selected `--output` path are not auto-pruned. Crash-orphaned pending prompts older than 24 hours are also reconciled on TUI startup. Delivery evidence remains content-safe metadata (hashes/ids/status evidence rather than assistant text) and is retained for recovery/diagnostics. Production run/turn diagnostics redact credential-like keys, auth/cookie material, token-bearing URLs, common key formats and local home paths before durable persistence or machine failure output; this does not sanitize explicit conversation exports, which intentionally contain the conversation text the user requested.

Terminal rendering also treats conversation/provider text as untrusted. Raw OSC/DCS/APC/PM/SOS controls, cursor/erase/private CSI, unsafe C0/C1 bytes and malformed escape fragments are stripped before display; only validated SGR presentation styling may survive the transcript boundary. Markdown rendered for a direct terminal is filtered again after Rich rendering, so a model-provided Markdown link cannot inject an OSC-8 terminal hyperlink. The explicit `chat:` hyperlink printed by gptty itself is application-owned UI rather than model-controlled terminal content.

`/goal` turns a normal ChatGPT conversation into a durable run-until-done workflow. A profile can hold many Goals at once and different gptty processes may actively run different Goals concurrently from the same profile; ownership is exclusive only per Goal. Each normal conversation is routed to at most one unfinished Goal, while one Goal may own a chain of conversation generations after Continue-As-New rollover. On an attached chat, bare `/goal` continues the task/plan already established there; `/goal <objective>` starts a Goal for that chat (or creates a new normal chat when none is attached). `/goal list` shows unfinished Goals, `/goal list all` also shows terminal history, and `/goal open <id-prefix>` switches through the normal safe resume path. The `/resume` picker annotates conversations that belong to unfinished Goals. Goal authority and conversation routing are profile-wide transactional SQLite at `goals/goal-state.sqlite3` (`~/.local/share/gptty/profiles/default/goals/goal-state.sqlite3` for the default profile), using WAL + full synchronous commits, optimistic revisions, and an append-only machine event journal. Local TUI selection is deliberately not profile-global: each interactive terminal writes a sibling `gptty_state.session-<source>-<hash>.json`, keyed by `GPTTY_SESSION_ID` when explicitly set, otherwise by `CMUX_SURFACE_ID` in cmux, a supported terminal session id, or the TTY identity. This lets two cmux surfaces keep different `current_conversation` values without last-writer-wins corruption while still sharing the same GoalStore, history, and UI settings. The old `gptty_state.json` remains the legacy/base seed for a new terminal session and for non-interactive commands. `goals/index.json` is a readable multi-Goal index; `goals/<goal-id>/goal.json`, `checkpoint.md`, and `events.jsonl` are portable projections/backups, never concurrency authority. If a terminal-local chat-selection file is lost or corrupt, gptty does not guess which Goal is current; `/goal list` remains the recovery entry point. Goal turns use `GPTTY_GOAL: CONTINUE|COMPLETE|BLOCKED` plus a structured checkpoint; `COMPLETE` is rejected if its checkpoint is missing, still has pending work, has no concrete completed claim, or an observed tool side effect is unresolved. Before every Goal turn gptty persists a durable operation id; observed tool calls/results and committed fresh-chat identities are journaled, so ambiguous writes are never blindly replayed after restart or transport failure. Recovery first reconciles external state; a missing original result can only be closed after a later machine-observed verification call/result pair plus the structured completion claim. Hard chat failure uses Continue-As-New semantics: the same Goal ID advances to a new generation/conversation while retaining every prior conversation binding. A missing/invalid status is treated as unfinished work and recovered rather than accepted as completion. `Ctrl-C`/`/stop`, local exit, or process restart pause only the affected Goal rather than unrelated Goals in the profile. Use `/goal pause`, `/goal resume`, `/goal status`, and `/goal clear` for the attached Goal; `clear` removes its conversation bindings while retaining durable history. Goal mode is intentionally unavailable in Temporary Chat because its continuation authority is not persistent.

`/resume` opens the real ChatGPT conversation catalog, supports fuzzy filtering, renders the selected conversation's user-visible history, and continues that same conversation. `/resume <URL-or-ID>` skips the picker. `/reload` atomically refreshes the currently attached normal conversation through the same canonical snapshot/reconciliation path without detaching first; if the refresh fails, the existing attachment remains intact, and an active goal for that same conversation is not paused. The selected conversation snapshot is loaded as a non-blocking resume operation: the composer stays usable, normal text queues until the snapshot finishes, `/exit`/`Ctrl-\\` can leave immediately, and stale `tool_running`/`tool_calling` states are shown as unfinished without starting a polling/follow loop. An attached chat header shows its full `https://chatgpt.com/c/...` URL, and a new chat prints that URL as soon as the first completed response supplies its id. During an enhanced interactive turn, a live `elapsed MM:SS`/`HH:MM:SS · Ctrl-C stop · Ctrl-\\ quit` timer stays at the bottom while thinking/tool output prints above it. Pressing `Ctrl-C` while a response is active invokes ChatGPT's real Stop generating control, waits for canonical readback to settle, keeps the conversation attached, and renders the saved partial answer; `/stop` exposes the same remote stop action for an already-active attached chat. Pressing `Ctrl-\\` exits gptty locally without sending Stop, so the active ChatGPT response continues in the browser; if pressed before CWA has confirmed the browser write, gptty briefly waits for that safe handoff before exiting. For a newly-created normal chat, the committed conversation id from that handoff is persisted before local exit so the same chat remains resumable on the next gptty launch. A user-stopped turn does not fire the completion notification. A normally completed interactive response schedules a best-effort native macOS notification via `osascript` without blocking the TUI. Notifications are privacy-safe by default: title `ChatGPT`, body `ChatGPT response complete.`, and no conversation/response preview. Preview is explicit opt-in and is still suppressed for Temporary Chat. Successful `/new`, `/temporary`, `/resume`, `/reload`, and `/detach` clear only the current viewport before rendering the new context; terminal scrollback remains available. Model choice and reasoning effort are stored as separate session intents. With no explicit overrides, text-only turns use the product's `DEEP/HIGH` default profile: model policy `latest frontier`, effort policy `Default · High`. `/model` opens the live normal-chat model catalog and `/model default` clears only the explicit model override. `/effort` opens a separate `default | instant | medium | high` picker, and `/effort default` clears only the saved effort override; the same intent is available to scripted `ask/send/chat` as `--effort`. The current CWA ProductRuntime does not yet expose a proven arbitrary custom-model + independent-effort combination, and rich-input turns cannot combine attachments with semantic profile selection, so gptty fails those explicit combinations **before the product write** instead of silently substituting a model/effort. Default image turns retain the existing live-catalog fallback to the strongest normal non-Work thinking frontier slug. `/detach` only clears the local attachment and does not modify the ChatGPT chat.

Use `Ctrl-R` for prompt history and `Alt-Enter` for a newline. `/history clear` deletes both persisted and loaded prompt history. Enhanced-chat UI settings live in the profile `ui.json`: `history_limit` defaults to `2000` (`0` disables persistent prompt history), `notifications` defaults to `true`, `notification_preview` defaults to `false`, and `notification_sound` defaults to `true`. Temporary Chat never persists prompt history and never exposes title/response previews in notifications, even when preview is enabled for normal chats. `gptty chat --plain` keeps the older line-oriented fallback without the enhanced action menu.

Run the full legacy interactive runtime:

```bash
gptty chat --legacy
```

One-shot SDK-backed prompt:

```bash
gptty ask "explain this error"
```

Pipe an exact prompt through stdin:

```bash
{ printf 'Review this patch:\n\n'; git diff; } | gptty ask
```

stdin and positional prompt text cannot be combined implicitly; gptty never inserts its own framing into user input. Use stdin alone, or use `--no-stdin` when a positional prompt should ignore a pipe. stdin is capped at 4 MiB by default, rejects NUL/binary-looking input, and can be bounded differently with `--stdin-max-bytes BYTES`.

Force reading stdin even when it looks interactive:

```bash
gptty ask --stdin
```

Ignore piped stdin:

```bash
cat noisy.log | gptty ask --no-stdin "explain this from the prompt only"
```

Disable streaming and print the final response:

```bash
gptty ask --no-stream "summarize this session"
gptty send --no-stream "summarize this conversation"
```

Legacy entrypoint, still supported from a checkout:

```bash
python main.py
```

You can also override local paths:

```bash
gptty auth status --auth ./auth_data.json
gptty auth refresh --auth ./auth_data.json --mode wait
gptty attach https://chatgpt.com/c/... --auth ./auth_data.json --state ./gptty_state.json
gptty send --auth ./auth_data.json --state ./gptty_state.json "hello"
gptty send --auth ./auth_data.json --state ./gptty_state.json --image ./screenshot.png "describe this"
gptty export --auth ./auth_data.json --state ./gptty_state.json --output conversation.md
gptty chat --auth ./auth_data.json --state ./gptty_state.json
gptty chat --legacy --auth ./auth_data.json --state ./webchat_state.json
gptty ask --auth ./auth_data.json --timeout 120 "hello"
```

## Useful Legacy Chat Commands

Available in `gptty chat --legacy`:

- `/help`
- `/models`
- `/new`
- `/list`
- `/use <chat_id>`
- `/reset`
- `/img <path_or_url> :: <prompt>`
- `/settings`
- `/model <name>`
- `/lang <en|ru>`
- `/ws <true|false>`
- `/effort <standard|extended|off>`
- `/metrics <true|false>`

## Important Files

- `auth_data.json` - local auth data, do not commit it
- `local-state.sqlite3` (plus SQLite `-wal` / `-shm` companions while open) - current transactional authority for local sessions and normal run/TUI/delivery evidence; profile installs keep it under the profile's `runs/` directory, while a custom state path uses `.gptty_runs/` beside that state path
- `gptty_state.json` and `gptty_state.session-*.json` - legacy/migration inputs for the modern runtime; retained files may be useful for rollback/backup but are not rewritten as authoritative session state
- `webchat_state.json` - legacy `--legacy` chat history and runtime settings, do not commit it

A plain `gptty chat` starts a fresh local runtime session so simultaneous terminals do not overwrite each other's current-chat/model/effort selection. Use `gptty chat --session NAME`, or the same `--session NAME` on `send`, `attach`, `messages`, `status`, `observe`, and `export`, when intentional reuse is required. `GPTTY_SESSION_ID=NAME` is the environment equivalent. Scripted commands without an explicit session use the stable `default` local session.

## Notes

- `auth_data.json` is the primary auth source.
- ChatGPT web-session auth may expire after some time; in practice, expect to refresh it periodically.
- Run `gptty auth status` when requests start failing or before long terminal sessions.
- `.env` is optional. If present, `accessToken` is used as a fallback even when `auth_data.json` is missing, but a full `auth_data.json` remains the most compatible setup.
- In `auto` mode, auth refresh sends one probe message to trigger capture. The default text is `"Hello"`, and you can override it with `--probe-prompt`.
- In `wait` mode, auth refresh does not send the probe automatically. Log in or register, then send any message manually in the browser to trigger capture.
- New auth captures write canonical `accessToken` plus the legacy-compatible `api_key` field.
- Do not mix `cookies` and `api_key/accessToken` from different accounts.
- Local state and auth files are written atomically to reduce the chance of truncated JSON after interruption.
- If `main.py` says that `curl` is missing, install system `curl.exe` and check `curl --version`.

## Troubleshooting

- `curl` not found
  Install system `curl.exe` and make sure `curl --version` works.
- `auth_data.json` is missing
  Run `gptty auth refresh --mode wait`, complete login in the browser, then send any message in the chat window.
- Auth may be expired
  Run `gptty auth status`. If it reports `expired`, run `gptty auth refresh --mode wait`.
- `gptty auth refresh` says auth dependencies are missing
  Reinstall auth dependencies with `python -m pip install -e ".[auth]"` from checkout, or `python -m pip install "gptty-web[auth]"` from an installed package.
- `gptty send`, `gptty messages`, `gptty status`, or `gptty export` says there is no attached conversation
  Run `gptty attach <url-or-id>` first, pass a conversation URL/id directly to the command, or use `gptty send --new`.
- `gptty ask/send --image` says an image file does not exist, or `--file` says a file does not exist
  Check the local path, or pass an `http(s)` URL instead.
- `ImportError: cannot import name 'nodriver'`
  Reinstall auth dependencies with `python -m pip install -e ".[auth]"`. Recent `g4f` releases use `zendriver` instead of the older `nodriver` package name.
- The wrong account opens in auth refresh
  The browser profile already contains another session. Log out there first, or use the wait mode and sign in to the intended account.
- Requests start failing after working before
  Your session cookies or `api_key/accessToken` may have expired. Regenerate `auth_data.json` with `gptty auth refresh --mode wait`.
- `gptty chat` starts but cannot answer
  Check that `auth_data.json` exists and the captured browser session still belongs to the same account.
- `gptty chat --legacy` starts but cannot answer
  Check that `auth_data.json` exists, `curl` is installed, and the captured browser session still belongs to the same account.

## Status

This repository is in transition from `webchat-openai-cli` to `gptty`.

PR0 establishes the package skeleton and console command. PR1 adds the SDK client boundary. PR2 adds the first SDK-backed command, `gptty ask`. PR3 centralizes stdin pipe handling. PR4 migrates the default `gptty chat` path to a minimal SDK-backed loop with legacy fallback. PR5 adds attach/messages/status conversation operations. PR6 adds send-to-attached, explicit, and new conversation workflows. PR7 adds shared output modes for messages/status/send. PR8 adds conversation export. PR9 adds SDK-backed image prompts for ask/send. PR10 adds auth status/refresh UX. PR11 prepares the first `gptty-web` release flow. Later PRs will add richer pipe workflows and SDK chat `/img` parity.

See [CHANGELOG.md](CHANGELOG.md) and [docs/release.md](docs/release.md) for release details.
