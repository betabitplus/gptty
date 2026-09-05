# GPTTY — CHEATSHEET

Продолжить текущий прикреплённый чат:

```bash
gptty chat
```

В интерактивном режиме команды помнить не нужно:

```text
/            открыть меню действий
Ctrl-R       поиск по истории prompts
Alt-Enter    новая строка внутри prompt
```

Основные действия доступны из `/`-меню: `new`, `resume`, `detach`, `model`, `exit`. `/resume` читает реальный список ChatGPT-чатов через CWA, даёт fuzzy-поиск, показывает полную user-visible текущую ветку истории и после выбора продолжает именно этот conversation. `/resume URL_OR_ID` подключает чат напрямую без списка. В header каждого подключённого чата печатается полный `https://chatgpt.com/c/...` URL (в поддерживающих терминалах он кликабельный); у только что созданного чата ссылка появляется сразу после первого полного ответа, когда ChatGPT выдаёт его id. Во время генерации default TTY показывает обновляемый `elapsed MM:SS`/`HH:MM:SS`; после final время замораживается. После полного ответа gptty посылает нативное macOS notification через системный `osascript`; ошибки notification не влияют на чат. `/detach` только локально снимает привязку и ничего не меняет в ChatGPT. Успешные `/new`, `/resume` и `/detach` очищают только текущий viewport перед новым состоянием; terminal scrollback не стирается. Дефолтная модель — `latest frontier · High`: gptty передаёт нативный product profile `DEEP/HIGH`, поэтому сейчас это GPT-5.6 Sol High, а при смене frontier ChatGPT сам подхватит более новую доступную тарифу модель. `/model` показывает реальный текущий normal-chat model catalog и позволяет сделать явный override; `/model default` возвращает `latest frontier · High`. Work-mode backend entries и Deep Research mode в обычный model picker не попадают.

Отключить enhanced UI и вернуть простой line-oriented режим:

```bash
gptty chat --plain
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

Live-режим включён по умолчанию. После safe-detach CWA пассивно читает копию того же response/SSE stream, который ChatGPT Web уже получает для своего UI. Дополнительных HTTP poll-запросов ради live нет: новые законченные user-visible thinking-параграфы и краткие tool-call строки приходят сразу из browser stream.

```text
[thinking]
Первый файл большой, поэтому читаю его диапазонами...

[tool call] api_tool.call_tool Reading git status...
[tool call] api_tool.call_tool Reading README.md...
```

Tool arguments/results не печатаются. Пустой `Thinking…` тоже не печатается. Показываются короткие thinking/preamble paragraphs и reasoning recap/title, которые реально присутствуют в ChatGPT conversation/Web UI; raw/private hidden `thoughts` не выводятся. После настоящего terminal/end-turn CWA делает canonical reconcile финального ответа; если canonical plane ещё не успел materialize финал, retry идёт редко и с backoff. Polling остаётся только аварийным fallback, если passive observer не смог подняться.

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

- в отдельном CWA Chrome нормальное состояние — 2 managed same-origin вкладки: runtime ChatGPT + лёгкая canonical-read `robots.txt`; обычные turns переиспользуют их;
- `/resume` запрашивает полный ChatGPT catalog только по явной команде; фонового listing/polling в idle нет;
- если выбранный через `/resume` чат уже активен, gptty делает один canonical snapshot раз в 15 секунд только до terminal status (максимум 2 часа); Ctrl-C прекращает follow, но оставляет чат подключённым;
- `/model` использует product slug из живого `/backend-api/models`; ChatGPT может внутри выбранного product option маршрутизировать ответ на конкретный serving variant, поэтому canonical `model_slug` ответа не обязан буквально совпадать с route slug;
- image continuation в существующий чат CWA 0.3 официально не поддерживает; `--image --new` работает;
- CodexPro вызывается самим ChatGPT внутри native connector/tool loop.
