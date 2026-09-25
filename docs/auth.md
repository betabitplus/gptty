# Auth Lifecycle

`gptty` uses an existing ChatGPT web session through `chatgpt-web-adapter` (CWA). It does not use the OpenAI API and does not treat an OpenAI API key as its ChatGPT product-session credential.

CWA is the single authority for reusable auth persistence. `gptty` captures browser session material when requested, then delegates storage, loading, migration, and deletion to CWA.

## Storage backends

Install the auth extra for browser capture plus OS credential-store support:

```bash
python -m pip install "gptty-web[auth]"
```

The auth extra includes `keyring`. CWA only accepts recognized OS-backed secure providers (macOS Keychain, Windows Credential Manager, and supported Linux Secret Service/KWallet-style backends); plaintext/null alternatives such as `keyrings.alt` are treated as unavailable. When a usable OS backend is available, CWA stores the reusable credential blob there. The companion `auth_data.json` then contains only non-secret backend/account metadata and expiry/timestamp hints.

If no usable OS credential store exists, CWA uses its hardened file fallback. The fallback file contains bearer-equivalent session material and is restricted to the owner (`0600` on POSIX). Do not commit or share it.

Transient proof/turnstile material is not written to the reusable credential store.

## Check auth status

```bash
gptty auth status
gptty auth status --format json
```

Status is content-free: it reports health, expiry, cookie/header presence, and credential-backend provenance without printing credential values.

Use a custom auth metadata/fallback path with `--auth`:

```bash
gptty auth status --auth ./auth_data.json
```

## Refresh auth data

The recommended interactive path is `wait` mode:

```bash
gptty auth refresh --mode wait
gptty auth refresh --mode wait --credential-store file
```

`--credential-store` accepts `auto` (default), `keyring`, or `file`. An already keyring-backed profile should be deliberately downgraded with `gptty auth migrate --backend file` rather than relying on refresh to leave a stale OS-store copy.

It keeps the browser open until ChatGPT is ready and then waits for you to send a message manually in the browser so capture can observe current session material.

For an already logged-in browser session, `auto` mode is faster but has a product side effect: it sends one probe prompt to trigger capture.

```bash
gptty auth refresh --mode auto
gptty auth refresh --mode auto --probe-prompt "Ping"
```

## Migrate the credential backend

Move an existing secure-file credential set into the OS store:

```bash
gptty auth migrate --backend keyring
```

Create the explicit portable/headless file fallback instead:

```bash
gptty auth migrate --backend file
```

Migration is ordered to avoid credential loss. File-to-keyring writes and verifies the OS-store item before replacing the secret-bearing file with metadata. Keyring-to-file writes the private fallback first and only then removes the OS-store item. If the final removal fails, the command reports an error and leaves both usable copies rather than deleting the last good credential set. If the OS-store write exists but the non-secret metadata pointer was lost, status/login can still recover the deterministic keyring entry and `gptty auth migrate --backend keyring` recreates the metadata without writing the secret back to plaintext.

Once a profile is marked keyring-backed, a temporary keyring outage fails closed. CWA does not silently repopulate plaintext credentials. Use `auth migrate --backend file` while the OS store is available if a deliberate downgrade is required.

## Logout / clear reusable local auth

```bash
gptty auth logout
```

For keyring-backed auth, the OS-store credential is deleted before the metadata file. A deletion failure keeps metadata in place for a safe retry. This command removes reusable local auth material; it does not claim to revoke the server-side ChatGPT session or delete a separate signed-in browser profile.

## Legacy capture scripts

The checkout-level `auth_fetcher.py` / `auth_fetcher_wait.py` entry points remain for compatibility, but their reusable persistence now delegates to the same CWA credential-store authority. They no longer maintain a second auth JSON format.
