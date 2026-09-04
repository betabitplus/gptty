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

Live-режим включён по умолчанию. После safe-detach `gptty` примерно каждые ~5 секунд читает один canonical snapshot и сразу печатает новые user-visible thinking-параграфы и краткие tool-call строки:

```text
[thinking]
Первый файл большой, поэтому читаю его диапазонами...

[tool call] api_tool.call_tool Reading git status...
[tool call] api_tool.call_tool Reading README.md...
```

Tool arguments/results не печатаются. Пустой `Thinking…` тоже не печатается. Показываются короткие thinking/preamble paragraphs и reasoning recap/title, которые реально присутствуют в ChatGPT conversation/Web UI; raw/private hidden `thoughts` не выводятся. Canonical snapshot одновременно используется и для live-блоков, и для финального readback: отдельных частых запросов ради live нет. `429` получает 15-секундный backoff.

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
open -na '/Users/stas/.agent-browser/browsers/chrome-151.0.7922.34/Google Chrome for Testing.app' --args --user-data-dir='/Users/stas/Library/Application Support/chatgpt-web-adapter/browser-profile' --remote-debugging-port=9333 --disable-background-timer-throttling --disable-backgrounding-occluded-windows --disable-renderer-backgrounding --load-extension='/Users/stas/.local/share/uv/tools/chatgpt-web-adapter/lib/python3.10/site-packages/chatgpt_web_adapter/browser_native_extension' https://chatgpt.com/
```

Важно:

- в отдельном CWA Chrome нормальное состояние — 2 managed ChatGPT-вкладки: runtime + canonical-read; обычные `gptty send --new` переиспользуют их, а не создают новую вкладку на каждый turn;
- списка всех ChatGPT-чатов в `gptty` сейчас нет;
- passive live-view turn, запущенного вручную в ChatGPT Web, нет;
- image continuation в существующий чат CWA 0.3 официально не поддерживает; `--image --new` работает;
- CodexPro вызывается самим ChatGPT внутри native connector/tool loop.
