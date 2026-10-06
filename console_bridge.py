"""別プロセスのコンソールへキー入力を書き込む／表示中の文字を読む（local-claude 直送モード用、2026-10-07）。

  python console_bridge.py inject <pid> <文字列> [--enter]   文字（と Enter）をキー入力として積む
  python console_bridge.py enter <pid>                     Enter だけを積む
  python console_bridge.py screen <pid>                    いま見えている範囲の文字を UTF-8 で出す

貼り付け（クリップボード＋Ctrl+V）と違い、フォーカスがどこにあっても <pid> のコンソールにだけ届く。
AttachConsole の前に自分のコンソールから離れる（FreeConsole）ので、dictation.py の中では呼ばず、
使い捨ての子プロセスとして起動すること（dictation.py の画面表示が消えるため）。
"""
import ctypes
import sys
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
INVALID_HANDLE = wintypes.HANDLE(-1).value
KEY_EVENT = 0x0001
VK_RETURN = 0x0D


class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                ("UnicodeChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]


class _EVENT(ctypes.Union):
    _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("_pad", ctypes.c_byte * 16)]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]


class COORD(ctypes.Structure):
    _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]


class SMALL_RECT(ctypes.Structure):
    _fields_ = [("Left", wintypes.SHORT), ("Top", wintypes.SHORT), ("Right", wintypes.SHORT), ("Bottom", wintypes.SHORT)]


class CSBI(ctypes.Structure):
    _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", wintypes.WORD),
                ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]


def _attach(pid):
    k32.FreeConsole()
    if not k32.AttachConsole(wintypes.DWORD(pid)):
        raise OSError(ctypes.get_last_error(), f"AttachConsole({pid}) に失敗（そのプロセスが無い・コンソールが無い）")


def _open(name):
    h = k32.CreateFileW(name, 0xC0000000, 0x3, None, 3, 0, None)  # GENERIC_READ|WRITE, SHARE_READ|WRITE, OPEN_EXISTING
    if h in (None, INVALID_HANDLE):
        raise OSError(ctypes.get_last_error(), f"{name} を開けない")
    return h


def _key_records(text, enter):
    units = []  # UTF-16 の単位で積む（絵文字などのサロゲートペアも1単位ずつ）
    for ch in text:
        b = ch.encode("utf-16-le")
        units += [b[i:i + 2].decode("utf-16-le", "surrogatepass") for i in range(0, len(b), 2)]
    keys = [(u, 0) for u in units]
    if enter:
        keys.append(("\r", VK_RETURN))
    recs = (INPUT_RECORD * (len(keys) * 2))()
    for i, (u, vk) in enumerate(keys):
        for j, down in enumerate((True, False)):
            r = recs[i * 2 + j]
            r.EventType = KEY_EVENT
            e = r.Event.KeyEvent
            e.bKeyDown, e.wRepeatCount, e.wVirtualKeyCode = down, 1, vk
            e.wVirtualScanCode, e.UnicodeChar, e.dwControlKeyState = 0, u, 0
    return recs


def inject(pid, text, enter=False):
    _attach(pid)
    try:
        h = _open("CONIN$")
        recs = _key_records(text, enter)
        n = wintypes.DWORD()
        ok = k32.WriteConsoleInputW(h, recs, len(recs), ctypes.byref(n))
        k32.CloseHandle(h)
        if not ok:
            raise OSError(ctypes.get_last_error(), "WriteConsoleInputW に失敗")
        return n.value
    finally:
        k32.FreeConsole()


def screen(pid):
    _attach(pid)
    try:
        h = _open("CONOUT$")
        info = CSBI()
        if not k32.GetConsoleScreenBufferInfo(h, ctypes.byref(info)):
            raise OSError(ctypes.get_last_error(), "GetConsoleScreenBufferInfo に失敗")
        width = info.dwSize.X
        buf = ctypes.create_unicode_buffer(width)
        n = wintypes.DWORD()
        lines = []
        for y in range(info.srWindow.Top, info.srWindow.Bottom + 1):
            k32.ReadConsoleOutputCharacterW(h, buf, width, COORD(0, y), ctypes.byref(n))
            lines.append(buf.value[: n.value].rstrip())
        k32.CloseHandle(h)
        return "\n".join(lines)
    finally:
        k32.FreeConsole()


if __name__ == "__main__":
    out = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)
    try:
        cmd, pid = sys.argv[1], int(sys.argv[2])
        if cmd == "inject":
            ok = inject(pid, sys.argv[3], "--enter" in sys.argv[4:]) > 0
        elif cmd == "enter":
            ok = inject(pid, "", True) > 0
        elif cmd == "screen":
            out.write(screen(pid))
            ok = True
        else:
            raise SystemExit(f"不明なコマンド: {cmd}")
    except OSError as e:
        out.write(f"ERROR: {e}\n")
        ok = False
    out.flush()
    sys.exit(0 if ok else 1)
