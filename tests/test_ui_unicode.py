from __future__ import annotations

import random

from prompt_toolkit.layout.processors import TabsProcessor
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.base import Size
from prompt_toolkit.utils import get_cwidth

from gptty.ui.session import (
    TABSTOP,
    InteractiveSession,
    _TranscriptLine,
    _tab_span,
    _text_width,
)


class ResizableDummyOutput(DummyOutput):
    def __init__(self, columns: int, rows: int = 24) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


def _row_text(row: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in row)


def _cell_width(value: str) -> int:
    return sum(max(0, get_cwidth(char)) for char in value)


def _expand_tabs_like_prompt_toolkit(value: str) -> str:
    display_position = 0
    rendered: list[str] = []
    for char in value:
        if char == "\t":
            span = _tab_span(display_position)
            rendered.append(" " * span)
            display_position += span
        else:
            rendered.append(char)
            display_position += 1
    return "".join(rendered)


def test_tabs_expand_explicitly_in_transcript_and_composer(tmp_path) -> None:
    line = _TranscriptLine()
    line.append("", "界\tA")
    rows = line.wrapped(5)

    assert [_row_text(row) for row in rows] == ["界   ", "A"]
    assert all("\t" not in _row_text(row) for row in rows)

    output = ResizableDummyOutput(7)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )
    processors = session._input_window.content.input_processors
    assert any(isinstance(processor, TabsProcessor) for processor in processors)

    buffer = session._session.default_buffer
    buffer.text = "abcd\tX"
    positions = session._input_visual_positions()

    # Prompt width is two cells, so the first visual row has five input cells.
    # The four-space tab crosses that boundary one cell at a time.
    assert _text_width(session._prompt_text()) == 2
    assert positions[4] == (0, 4)
    assert positions[5] == (1, 3)
    assert positions[6] == (1, 4)


def test_transcript_unicode_fuzz_preserves_order_and_cell_bounds() -> None:
    rng = random.Random(20260926)
    tokens = (
        "a",
        "Z",
        "界",
        "漢",
        "🙂",
        "🚀",
        "e\u0301",
        "n\u0303",
        "א",
        "ב",
        "م",
        "ر",
        "\t",
        "·",
        "—",
    )

    for _case in range(120):
        text = "".join(rng.choice(tokens) for _ in range(rng.randint(8, 48)))
        line = _TranscriptLine()
        line.append("", text)
        expected = _expand_tabs_like_prompt_toolkit(text)

        first_width = rng.choice((4, 5, 8, 13, 21))
        second_width = rng.choice((4, 6, 9, 17, 24))
        first = line.wrapped(first_width)
        second = line.wrapped(second_width)
        first_again = line.wrapped(first_width)

        assert first_again == first
        assert "".join(_row_text(row) for row in first) == expected
        assert "".join(_row_text(row) for row in second) == expected
        assert all("\t" not in _row_text(row) for row in first)
        assert all("\t" not in _row_text(row) for row in second)
        assert all(_cell_width(_row_text(row)) <= first_width for row in first)
        assert all(_cell_width(_row_text(row)) <= second_width for row in second)


def test_composer_unicode_fuzz_is_deterministic_and_navigation_is_bounded(
    tmp_path,
) -> None:
    rng = random.Random(20260926)
    tokens = (
        "a",
        "界",
        "🙂",
        "e\u0301",
        "א",
        "م",
        "\t",
        " ",
        "\n",
    )
    output = ResizableDummyOutput(17)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )
    buffer = session._session.default_buffer

    for _case in range(80):
        text = "".join(rng.choice(tokens) for _ in range(rng.randint(6, 36)))
        buffer.text = text

        output.columns = 17
        wide = list(session._input_visual_positions())
        output.columns = 9
        narrow = list(session._input_visual_positions())
        output.columns = 17
        wide_again = list(session._input_visual_positions())

        assert wide_again == wide
        for width, positions in ((17, wide), (9, narrow)):
            assert len(positions) == len(text) + 1
            assert all(row >= 0 and 0 <= col < width for row, col in positions)
            assert all(
                positions[index][0] <= positions[index + 1][0]
                for index in range(len(positions) - 1)
            )
            assert all(
                positions[index][1] <= positions[index + 1][1]
                for index in range(len(positions) - 1)
                if positions[index][0] == positions[index + 1][0]
            )

        output.columns = 9
        probe_indexes = {
            0,
            len(text),
            *(rng.randrange(0, len(text) + 1) for _ in range(min(8, len(text) + 1))),
        }
        positions = session._input_visual_positions()
        for index in probe_indexes:
            buffer.cursor_position = index
            session._move_input_visual_rows(1)
            assert index <= buffer.cursor_position <= len(text)

            buffer.cursor_position = index
            session._move_input_visual_rows(-1)
            assert 0 <= buffer.cursor_position <= index

        # The source buffer stays exact; tab expansion is render-only.
        assert buffer.text == text


def test_tab_span_matches_prompt_toolkit_default_contract() -> None:
    assert TABSTOP == 4
    assert [_tab_span(position) for position in range(8)] == [4, 3, 2, 1, 4, 3, 2, 1]
