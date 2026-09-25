# Changelog

All notable changes to this project should be documented in this file.

The format is intentionally lightweight. Keep entries focused on user-visible behavior, compatibility notes, and release-impacting changes.

## Unreleased

- recovery: supersede stale local chat-level terminal evidence only from stronger typed proof, use explicit CWA canonical-read provenance for resume/follow reconciliation, keep resolution append-only and race-safe, and preserve recurring terminal states after resolution without leaking reconciliation bookkeeping into the human transcript
- stop/recovery: normalize provider Stop proof and verified conversation identity, allow Stop to supersede stale chat-level evidence only for the exact proven conversation, and refuse to rebind an attached chat from an unverified drifted Stop route
- goal/recovery: centralize Goal terminal disposition for result and failure paths so ambiguous writes, dead chats, blocking states, service backoff and truncated recovery cannot diverge between entry points
- recovery: classify CWA turn failures from structured provider evidence before compatibility text fallbacks, and block Goal after ambiguous post-submit writes instead of automatically resending or rolling over
- sessions: replace terminal-hash JSON session authority with a revisioned SQLite session registry, add intentional `--session NAME` reuse, atomically claim one-time legacy imports, retain legacy Goal payload as crash-recovery input, and fail closed on stale local writers without age-evicting unleased live sessions or making an already-completed ChatGPT turn retryable
- durability: make one SQLite/WAL local event store authoritative for run events and normal TUI observations, with bounded recent queries, transactional dedupe, private storage, incremental portable projections, and projection repair after crashes
- diagnostics: move stream-delivery evidence into the shared transactional store, import legacy rotated JSONL evidence once, and replace multiprocess-unsafe rename rotation with a bounded lock-serialized support projection
- performance: tail the optional CodexPro activity journal incrementally by file identity and offset, recover from truncation/rotation, and cap the in-memory activity window instead of reparsing all history on every refresh
- concurrency: replace conversation stale-file/PID lock recovery with the same kernel-backed ownership model used by Goal runs; retained sidecars are diagnostic only and `observe` probes the kernel lock rather than file existence
- packaging: require `chatgpt-web-adapter>=0.3.1,<0.4.0`, the first exact CWA release candidate verified against the current gptty browser-authority/WK runtime contract
- feat: add profile-aware auth/state path resolution with `gptty profile` commands
- feat: add local conversation locks for `gptty send` and `gptty chat`
- feat: add `gptty observe` for local active run status and recent output
- docs: document profile usage, resolution priority, and storage paths
- docs: document conversation lock behavior and lock wait options
- docs: document observe usage and local live-status limitations

## 0.1.1 - 2026-06-24

- feat: show a clear terminal message when ChatGPT stops on a connector required-action card such as Gmail OAuth/linking
- changed: require `chatgpt-web-adapter>=0.1.5,<0.2.0` so required-action detection is available in supported installs

## 0.1.0

Initial `gptty-web` release candidate.

### Added

- package layout with the `gptty` console command.
- SDK client boundary powered by `chatgpt-web-adapter`.
- one-shot prompts through `gptty ask`.
- minimal SDK-backed interactive chat through `gptty chat`.
- legacy runtime fallback through `gptty chat --legacy`.
- stdin pipe support for `ask` and `send`.
- conversation attach, messages, status, send, and export commands.
- output formats for script-friendly workflows: `plain`, `json`, and `markdown` where supported.
- SDK-backed image prompt support for `gptty ask --image` and `gptty send --image`.
- auth inspection through `gptty auth status`.
- auth refresh wrapper through `gptty auth refresh`.
- English and Russian README documentation.
- auth lifecycle documentation in `docs/auth.md`.

### Changed

- `gptty` now treats `chatgpt-web-adapter` as the SDK engine instead of keeping web-session transport logic in the CLI layer.
- new auth captures write canonical `accessToken` plus the legacy-compatible `api_key` field.
- local state and auth files are written atomically.

### Compatibility Notes

- The PyPI distribution name is `gptty-web`; the installed command is `gptty`.
- The project uses existing ChatGPT web-session auth, not the official OpenAI API.
- `gptty auth refresh` requires optional browser-capture dependencies installed with `gptty-web[auth]`.
- `auth_data.json` may need periodic refresh when ChatGPT web-session auth expires.
