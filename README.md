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
git diff | gptty ask "review this patch"
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
- SDK-backed image prompts through `gptty ask --image` and `gptty send --image`
- inspect attached or explicit conversations through `gptty messages` and `gptty status`
- export attached or explicit conversations through `gptty export`
- output modes for `messages`, `status`, `send`, and `export`: `plain`, `json`, `markdown`
- legacy interactive chat fallback through `gptty chat --legacy`
- one-shot SDK-backed prompts through `gptty ask`
- centralized stdin policy for pipe-friendly prompts
- pipe-friendly prompts, for example `git diff | gptty ask "review this patch"`
- streaming replies in the terminal
- minimal SDK chat state file: `gptty_state.json`
- legacy state file for `--legacy`: `webchat_state.json`
- atomic writes for local state and `auth_data.json`
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

## Get `auth_data.json`

Check the current auth file without opening a browser:

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

After a successful capture, `auth_data.json` will appear in the project directory. See [docs/auth.md](docs/auth.md) for lifecycle details and troubleshooting notes.

## Run the CLI

Attach an existing ChatGPT conversation:

```bash
gptty attach https://chatgpt.com/c/...
```

Send a prompt to the attached conversation:

```bash
gptty send "continue from here"
```

Pipe stdin into the attached conversation:

```bash
git diff | gptty send "review this patch"
```

Send to an explicit conversation without changing first through `attach`:

```bash
gptty send --to https://chatgpt.com/c/... "continue there"
```

Start a new conversation and store its returned conversation reference in `gptty_state.json`:

```bash
gptty send --new "start a new conversation"
```

Send image prompts through the SDK-backed commands:

```bash
gptty ask --image screenshot.png "describe this UI"
gptty ask --image https://example.com/chart.png "summarize this chart"
gptty send --image diagram.webp "continue with this image"
gptty send --image before.png --image after.png "compare these images"
```

`--image` accepts local file paths, `http(s)` URLs, and data URIs. It can be used more than once. Supported SDK image formats are PNG, JPEG/JPG, GIF, and WebP.

Inspect the attached conversation:

```bash
gptty messages --last 5
gptty status
```

Use JSON or Markdown output for scripts and exports:

```bash
gptty messages --last 5 --format json
gptty messages --last 5 --format markdown
gptty status --format json
gptty send --format json "summarize the current thread"
gptty export --format markdown --output conversation.md
gptty export --format json --output conversation.json
```

When `gptty send` uses `--format json` or `--format markdown`, streaming is disabled internally so the output stays complete and parseable.

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
/resume
/detach
/stop
/image
/paste
/model
/exit
```

`/image <path>` queues an image for the next prompt; `/image` without an argument opens a path prompt, so a file can be dragged from Finder into the terminal. Repeat it to attach multiple images. `/paste` materializes the current macOS clipboard image as a temporary PNG using the native pasteboard (`osascript`), so screenshots copied to the clipboard can be sent without manually saving them. Pending images are shown in the prompt as `[N images]`; `/image clear` removes them, and successful `/new`, `/resume`, or `/detach` clears pending images to prevent accidental cross-chat sends. Clipboard temp files are removed after the prompt is successfully sent or when the session/context is closed.

`/resume` opens the real ChatGPT conversation catalog, supports fuzzy filtering, renders the selected conversation's user-visible history, and continues that same conversation. `/resume <URL-or-ID>` skips the picker. An attached chat header shows its full `https://chatgpt.com/c/...` URL, and a new chat prints that URL as soon as the first completed response supplies its id. During an enhanced interactive turn, a live `elapsed MM:SS`/`HH:MM:SS · Ctrl-C stop · Ctrl-\\ quit` timer stays at the bottom while thinking/tool output prints above it. Pressing `Ctrl-C` while a response is active invokes ChatGPT's real Stop generating control, waits for canonical readback to settle, keeps the conversation attached, and renders the saved partial answer; `/stop` exposes the same remote stop action for an already-active attached chat. Pressing `Ctrl-\\` exits gptty locally without sending Stop, so the active ChatGPT response continues in the browser; if pressed before CWA has confirmed the browser write, gptty briefly waits for that safe handoff before exiting. A user-stopped turn does not fire the completion notification. A normally completed interactive response triggers a best-effort native macOS notification via `osascript`; the title uses the real ChatGPT conversation title and the body contains the latest user prompt, normalized and length-bounded, so concurrent chat notifications are immediately recognizable without exposing technical IDs. Successful `/new`, `/resume`, and `/detach` clear only the current viewport before rendering the new context; terminal scrollback remains available. The default model policy is `latest frontier · High`: text-only turns use ChatGPT's native `DEEP/HIGH` product profile. Rich-input turns cannot currently combine CWA's profile selector with attachments, so gptty resolves the live model catalog once per session and uses the strongest normal non-Work thinking frontier slug for default image turns; currently this is GPT-5.6 Sol High. `/model` opens the live normal-chat model catalog for explicit overrides, and `/model default` returns to this latest-frontier High policy. `/detach` only clears the local attachment and does not modify the ChatGPT chat.

Use `Ctrl-R` for prompt history and `Alt-Enter` for a newline. `gptty chat --plain` keeps the older line-oriented fallback without the enhanced action menu.

Run the full legacy interactive runtime:

```bash
gptty chat --legacy
```

One-shot SDK-backed prompt:

```bash
gptty ask "explain this error"
```

Pipe stdin into the prompt:

```bash
git diff | gptty ask "review this patch"
```

When stdin and a prompt are both present, `gptty ask` and `gptty send` send stdin as context, followed by the prompt under `User prompt:`.

Force reading stdin:

```bash
gptty ask --stdin "summarize this input"
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
- `gptty_state.json` - minimal SDK-backed chat state, do not commit it
- `webchat_state.json` - legacy chat history and runtime settings, do not commit it

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
- `gptty ask --image` or `gptty send --image` says an image file does not exist
  Check the local path, or pass an `http(s)` image URL instead.
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
