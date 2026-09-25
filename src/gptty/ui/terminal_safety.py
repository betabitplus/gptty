from __future__ import annotations

ESC = "\x1b"
BEL = "\x07"
C1_CSI = "\x9b"
C1_DCS = "\x90"
C1_SOS = "\x98"
C1_OSC = "\x9d"
C1_PM = "\x9e"
C1_APC = "\x9f"
C1_ST = "\x9c"

_CONTROL_STRING_STARTS = {"]", "P", "X", "^", "_"}
_C1_CONTROL_STRING_STARTS = {C1_DCS, C1_SOS, C1_OSC, C1_PM, C1_APC}
_SAFE_SGR_PARAMETER_CHARS = frozenset("0123456789;:")


def sanitize_terminal_text(value: str, *, allow_sgr: bool = False) -> str:
    """Return terminal-safe text, optionally retaining validated SGR styling.

    Untrusted model/tool/user text must never carry terminal control sequences.
    The only escape sequence retained when allow_sgr is true is a CSI SGR
    sequence whose parameters contain digits, semicolons and colons only.
    """

    text = str(value).replace("\r\n", "\n").replace("\r", "")
    output: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        codepoint = ord(char)

        if char == ESC:
            index, sgr = _consume_escape(text, index, allow_sgr=allow_sgr)
            if sgr is not None:
                output.append(sgr)
            continue

        if char == C1_CSI:
            index, sgr = _consume_csi(
                text,
                index + 1,
                allow_sgr=allow_sgr,
            )
            if sgr is not None:
                output.append(sgr)
            continue

        if char in _C1_CONTROL_STRING_STARTS:
            index = _consume_control_string(text, index + 1)
            continue

        if char == C1_ST or codepoint == 0x7F or 0x80 <= codepoint <= 0x9F:
            index += 1
            continue

        if codepoint < 0x20:
            if char in {"\n", "\t"}:
                output.append(char)
            index += 1
            continue

        output.append(char)
        index += 1

    return "".join(output)


def _consume_escape(
    text: str,
    index: int,
    *,
    allow_sgr: bool,
) -> tuple[int, str | None]:
    if index + 1 >= len(text):
        return len(text), None

    next_char = text[index + 1]
    if next_char == "[":
        return _consume_csi(text, index + 2, allow_sgr=allow_sgr)
    if next_char in _CONTROL_STRING_STARTS:
        return _consume_control_string(text, index + 2), None
    if next_char == "\\":
        return index + 2, None

    # Other ESC sequences (save/restore cursor, charset selection, keypad mode,
    # etc.) are terminal controls too. Strip the introducer and command byte.
    return index + 2, None


def _consume_csi(
    text: str,
    parameter_start: int,
    *,
    allow_sgr: bool,
) -> tuple[int, str | None]:
    index = parameter_start
    while index < len(text):
        char = text[index]
        codepoint = ord(char)
        if 0x40 <= codepoint <= 0x7E:
            parameters = text[parameter_start:index]
            safe_sgr = (
                allow_sgr
                and char == "m"
                and all(item in _SAFE_SGR_PARAMETER_CHARS for item in parameters)
            )
            sgr = f"{ESC}[{parameters}m" if safe_sgr else None
            return index + 1, sgr
        if char == ESC or char in _C1_CONTROL_STRING_STARTS or char == C1_ST:
            # Malformed CSI: stop before a new control introducer so the outer
            # scanner can consume that sequence independently.
            return index, None
        if codepoint < 0x20 or codepoint == 0x7F:
            return index + 1, None
        index += 1

    # Unterminated CSI is unsafe; consume the remainder rather than returning a
    # raw ESC fragment to a terminal-facing parser.
    return len(text), None


def _consume_control_string(text: str, index: int) -> int:
    while index < len(text):
        char = text[index]
        if char in {BEL, C1_ST}:
            return index + 1
        if char == ESC and index + 1 < len(text) and text[index + 1] == "\\":
            return index + 2
        index += 1
    # Unterminated OSC/DCS/APC/PM/SOS is dropped through end-of-input.
    return len(text)


__all__ = ["sanitize_terminal_text"]
