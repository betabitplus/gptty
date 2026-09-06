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

Основные действия доступны из `/`-меню: `new`, `temporary`, `resume`, `detach`, `stop`, `goal`, `export`, `image`, `paste`, `model`, `exit`. `/temporary` создаёт настоящий новый Temporary ChatGPT-чат. Он продолжается только пока жив этот же gptty/runtime: temporary conversation id намеренно не сохраняется как обычный resumable chat. `/new`, `/resume`, `/detach` и обычный `/exit` завершают temporary lifecycle. `/export` без дополнительных вопросов сохраняет весь текущий attached-чат в новый Markdown-файл в `~/Documents/gptty-exports/` и сразу печатает абсолютный путь. Для обычного чата история берётся целиком из CWA, для текущего Temporary Chat — из live transcript этой gptty-сессии; старые экспорты никогда не перезаписываются. `/image PATH` прикрепляет файл к следующему prompt; `/image` без аргумента открывает отдельное поле пути, куда можно просто перетащить файл из Finder. `/paste` берёт текущую картинку прямо из macOS clipboard (включая screenshot clipboard), сам временно материализует её в PNG и после успешной отправки удаляет. Пока есть вложения, prompt показывает `[N images]`; `/image clear` их сбрасывает. При `new/temporary/resume/detach` pending images тоже очищаются, чтобы случайно не отправить их в другой чат. `/resume` читает реальный список ChatGPT-чатов через CWA, даёт fuzzy-поиск, показывает полную user-visible текущую ветку истории и после выбора продолжает именно этот conversation. `/resume URL_OR_ID` подключает чат напрямую без списка. В header каждого подключённого чата печатается полный `https://chatgpt.com/c/...` URL (в поддерживающих терминалах он кликабельный); у только что созданного чата ссылка появляется сразу после первого полного ответа, когда ChatGPT выдаёт его id. Во время генерации composer остаётся активным: нижняя строка показывает `working MM:SS · queued N · / commands · Ctrl-C stop · Ctrl-\\ quit`. Обычный текст + Enter не вмешивается в текущий ответ, а становится FIFO-подсказкой для следующего turn; после завершения текущего ответа gptty отправляет очередь автоматически. `/stop` доступен прямо во время `working` и использует тот же реальный ChatGPT Stop path, что `Ctrl-C`; подтверждённый Stop очищает ещё не отправленные queued prompts, чтобы работа неожиданно не возобновилась. `/goal pause`, `/goal status`, `/image PATH`, `/image clear` и `/paste` тоже доступны во время работы; команды смены контекста и read-heavy действия (`/new`, `/temporary`, `/resume`, `/detach`, `/model`, `/export`) требуют сначала закончить или остановить текущий response. `Ctrl-C` нажимает реальный ChatGPT Stop generating, сразу фиксирует остановку локально, затем кратко завершает canonical readback, показывает сохранённый частичный ответ и оставляет чат прикреплённым. Если после уже подтверждённого Stop локальный readback всё же подвис, второй `Ctrl-C` немедленно выходит из gptty вместо повторной попытки остановить уже остановленный ChatGPT. `Ctrl-\\` завершает только локальный gptty и не отправляет Stop — активный ответ продолжает генерироваться в ChatGPT; если клавиша нажата ещё до подтверждённой передачи browser write, gptty сначала кратко дожидается safe handoff и только потом выходит. Остановленный пользователем turn не присылает completion notification. После обычного полного ответа gptty посылает нативное macOS notification со звуком через системный `osascript`: в заголовке — начало реального ChatGPT title этого разговора, а в тексте — начало последнего завершённого ответа ассистента; оба поля нормализованы и коротко ограничены по длине, поэтому уведомления разных чатов легко узнаются без технических ID и служебных префиксов; ошибки notification не влияют на чат. `/detach` только локально снимает привязку и ничего не меняет в ChatGPT. Успешные `/new`, `/temporary`, `/resume` и `/detach` очищают только текущий viewport перед новым состоянием; terminal scrollback не стирается. Дефолтная модель — `latest frontier · High`: для text-only gptty использует нативный `DEEP/HIGH`; для image-turn, где CWA пока не разрешает одновременно attachment и profile-selector, gptty один раз читает live model catalog и выбирает strongest normal non-Work thinking frontier slug. Сейчас это GPT-5.6 Sol High. `/model` показывает реальный текущий normal-chat model catalog и позволяет сделать явный override; `/model default` возвращает `latest frontier · High`. Work-mode backend entries и Deep Research mode в обычный model picker не попадают.

Goal mode:

```text
/goal                  продолжать уже согласованную задачу текущего чата до конца
/goal <цель>           задать явную цель; без attached-чата создаст новый normal chat
/goal pause            приостановить автопродолжение
/goal resume           продолжить сохранённую цель
/goal status           показать состояние/счётчики
/goal clear            забыть сохранённую цель
```

В Goal mode финальный ответ каждого turn начинается с `GPTTY_GOAL: CONTINUE`, `GPTTY_GOAL: COMPLETE` или `GPTTY_GOAL: BLOCKED`. `CONTINUE` автоматически запускает следующий turn без macOS notification; `COMPLETE` останавливает цикл и присылает обычное финальное уведомление; `BLOCKED` останавливает цикл и сообщает, что нужен пользователь. Если status marker отсутствует, gptty считает работу незавершённой и пробует продолжить с восстановлением протокола; после трёх последовательных пропусков goal аварийно прерывается вместо бесконечного цикла. Во время Goal turn обычный введённый текст становится queued steering: он отправится раньше автоматического `CONTINUE` и заменит уже ожидающий auto-continuation, поэтому правки можно вносить прямо во время работы. `/goal pause` во время `working` не обрывает текущий ответ, но после него не запускается следующий auto-turn. `Ctrl-C` или `/stop` во время Goal turn делают настоящий ChatGPT Stop и одновременно ставят goal на pause. `Ctrl-\\`, restart gptty, `/new`, `/temporary`, переключение через `/resume`, `/detach` и `/exit` тоже не дают goal самопроизвольно продолжиться: состояние сохраняется paused и требует явного `/goal resume`. Goal mode намеренно не работает внутри Temporary Chat.

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
