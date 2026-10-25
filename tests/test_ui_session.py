from __future__ import annotations

import asyncio

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.base import Size

from gptty.ui.session import InteractiveSession


class ResizableDummyOutput(DummyOutput):
    def __init__(self, columns: int, rows: int = 24) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


def test_persistent_application_survives_multiple_submits(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            first = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)
            app_task = session._application_task
            assert app_task is not None and not app_task.done()

            pipe.send_text("one\r")
            assert await asyncio.wait_for(first, timeout=5) == "one"
            assert session._application_task is app_task
            assert not app_task.done()

            second = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.02)
            pipe.send_text("two\r")
            assert await asyncio.wait_for(second, timeout=5) == "two"
            assert session._application_task is app_task
            assert not app_task.done()

            await session.stop_async()
            assert app_task.done()

    asyncio.run(scenario())


def test_alt_enter_inserts_newline_before_submit(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("first\x1b\rsecond\r")

        assert session.read_prompt() == "first\nsecond"
