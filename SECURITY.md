# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please do not open a public issue.

Instead, report it privately by contacting:

- `ikymuco@gmail.com`

Please include:

- a short description of the issue
- steps to reproduce it
- the affected file or feature
- the potential impact

## Sensitive Data

This project may use local session data such as:

- `auth_data.json`
- cookies
- `accessToken` / `api_key`
- `.env`

Do not include any secrets, tokens, cookies, OS credential-store entries, secret-bearing auth fallback files, or personal session files in bug reports, screenshots, logs, or public issues.

Reusable auth persistence is owned by `chatgpt-web-adapter`: a working OS `keyring` backend is preferred, while a `0600` POSIX file remains the explicit portable fallback. A keyring-backed profile fails closed if that store is unavailable rather than silently recreating plaintext credentials. Use `gptty auth migrate --backend file` for a deliberate fallback and `gptty auth logout` to remove reusable local auth. Neither command is a claim of server-side session revocation.

If needed, redact sensitive values before sharing anything.

## Scope

The most relevant security areas for this project are:

- local auth/session handling
- accidental exposure of tokens or cookies
- unsafe handling of local files or image inputs
- command execution or injection risks

## Supported Versions

Security fixes are currently provided on the latest version of the repository only.
