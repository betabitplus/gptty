# GPTTY — CHEATSHEET

Продолжить текущий прикреплённый чат:

```bash
gptty chat
```

Прикрепить существующий ChatGPT-чат:

```bash
gptty attach https://chatgpt.com/c/CHAT_ID
```

Посмотреть последние сообщения:

```bash
gptty messages --last 20
```

Отправить одно сообщение в текущий чат:

```bash
gptty send "текст"
```

Отправить сообщение в конкретный чат:

```bash
gptty send --to https://chatgpt.com/c/CHAT_ID "текст"
```

Создать новый чат:

```bash
gptty send --new "первое сообщение"
```

Новый чат с картинкой:

```bash
gptty send --new --image ./image.png "что на картинке?"
```

Статус текущего чата:

```bash
gptty status
```

Live-режим включён по умолчанию. Для turn, отправленного через `gptty`, по ходу работы видно примерно:

```text
[tool] ...
[tool done] ...
[thinking] Thinking…
[thinking] ...
```

Отключить live:

```bash
gptty chat --no-stream
```

Выйти из интерактивного чата:

```text
/exit
```

Дефолтный timeout для `send` / `chat` / `ask` — 2 часа. При необходимости:

```bash
gptty send --timeout 20000 "..."
```

Если появляется `CHATGPT_TURN_HTTP_STATUS:403`, перезапусти отдельный CWA Chrome:

```bash
pkill -f 'chatgpt-web-adapter/browser-profile'
open -na '/Users/stas/.agent-browser/browsers/chrome-151.0.7922.34/Google Chrome for Testing.app' --args --user-data-dir='/Users/stas/Library/Application Support/chatgpt-web-adapter/browser-profile' --remote-debugging-port=9333 --load-extension='/Users/stas/.local/share/uv/tools/chatgpt-web-adapter/lib/python3.10/site-packages/chatgpt_web_adapter/browser_native_extension' https://chatgpt.com/
```

Важно:

- списка всех ChatGPT-чатов в `gptty` сейчас нет;
- passive live-view turn, запущенного вручную в ChatGPT Web, нет;
- image continuation в существующий чат CWA 0.3 официально не поддерживает; `--image --new` работает;
- CodexPro вызывается самим ChatGPT внутри native connector/tool loop.
