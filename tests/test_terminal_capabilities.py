from __future__ import annotations

from gptty.ui.terminal_capabilities import TerminalCapabilities


class FakeOutput:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flush_calls = 0
        self.fail_write = False
        self.fail_flush_count = 0

    def write_raw(self, value: str) -> None:
        if self.fail_write:
            raise RuntimeError('write failed')
        self.writes.append(value)

    def flush(self) -> None:
        self.flush_calls += 1
        if self.fail_flush_count:
            self.fail_flush_count -= 1
            raise RuntimeError('flush failed')


def test_terminal_capabilities_enter_restore_are_idempotent() -> None:
    output = FakeOutput()
    caps = TerminalCapabilities(output)

    caps.enter_persistent_tui()
    caps.enter_persistent_tui()
    assert caps.alternate_scroll_enabled is True
    assert output.writes == ['\x1b[?1007h']

    caps.restore()
    caps.restore()
    assert caps.alternate_scroll_enabled is False
    assert output.writes == ['\x1b[?1007h', '\x1b[?1007l']


def test_enter_write_failure_does_not_arm_restoration() -> None:
    output = FakeOutput()
    output.fail_write = True
    caps = TerminalCapabilities(output)

    caps.enter_persistent_tui()

    assert caps.alternate_scroll_enabled is False
    caps.restore()
    assert output.writes == []


def test_enter_flush_failure_keeps_restoration_armed() -> None:
    output = FakeOutput()
    output.fail_flush_count = 1
    caps = TerminalCapabilities(output)

    caps.enter_persistent_tui()
    assert caps.alternate_scroll_enabled is True

    caps.restore()
    assert caps.alternate_scroll_enabled is False
    assert output.writes == ['\x1b[?1007h', '\x1b[?1007l']


def test_restore_failure_remains_armed_for_retry() -> None:
    output = FakeOutput()
    caps = TerminalCapabilities(output)
    caps.enter_persistent_tui()
    output.fail_flush_count = 1

    caps.restore()
    assert caps.alternate_scroll_enabled is True

    caps.restore()
    assert caps.alternate_scroll_enabled is False
    assert output.writes == ['\x1b[?1007h', '\x1b[?1007l', '\x1b[?1007l']
