# -*- coding: utf-8 -*-
"""
OmniDictate-CLI Launcher
GUI launcher for managing dictation settings (Language, Device, AI Mode).
Replaces the need for multiple batch files.
"""

import tkinter as tk
from tkinter import font as tkfont
import subprocess
import sys
import os
import json

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(APP_DIR, "venv", "Scripts", "python.exe")
CONFIG_FILE = os.path.join(APP_DIR, "launcher_config.json")

# --- Color Palette (Dark Theme) ---
COLORS = {
    "bg":           "#1a1a2e",
    "bg_card":      "#16213e",
    "bg_hover":     "#1f3055",
    "accent":       "#e94560",
    "accent_hover":  "#ff6b81",
    "text":         "#eaeaea",
    "text_dim":     "#8892a0",
    "text_label":   "#c4c9d4",
    "border":       "#2a3a5c",
    "toggle_on":    "#4ecca3",
    "toggle_off":   "#555e70",
    "gpu":          "#f0a500",
    "cpu":          "#4ecdc4",
    "status_ready": "#4ecca3",
    "status_run":   "#f0a500",
}


class ToggleButton(tk.Canvas):
    """Custom toggle switch widget."""

    def __init__(self, parent, width=56, height=28, on_color=None, off_color=None,
                 command=None, initial=False, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=COLORS["bg_card"], highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.on_color = on_color or COLORS["toggle_on"]
        self.off_color = off_color or COLORS["toggle_off"]
        self.command = command
        self.state = initial
        self.circle_pad = 3
        self.circle_r = (height // 2) - self.circle_pad
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self):
        self.delete("all")
        r = self.height // 2
        color = self.on_color if self.state else self.off_color
        # Background pill
        self.create_oval(0, 0, self.height, self.height, fill=color, outline="")
        self.create_oval(self.width - self.height, 0, self.width, self.height,
                         fill=color, outline="")
        self.create_rectangle(r, 0, self.width - r, self.height,
                              fill=color, outline="")
        # Circle knob
        if self.state:
            cx = self.width - r
        else:
            cx = r
        self.create_oval(cx - self.circle_r, self.circle_pad,
                         cx + self.circle_r, self.height - self.circle_pad,
                         fill="white", outline="")

    def _toggle(self, event=None):
        self.state = not self.state
        self._draw()
        if self.command:
            self.command(self.state)

    def set_state(self, state):
        self.state = state
        self._draw()

    def get_state(self):
        return self.state


class OptionSelector(tk.Frame):
    """Custom segmented control for selecting between options."""

    def __init__(self, parent, options, command=None, initial=0, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], **kwargs)
        self.options = options
        self.command = command
        self.selected = initial
        self.buttons = []

        for i, (label, color) in enumerate(options):
            btn = tk.Label(
                self, text=label, font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                padx=20, pady=8, cursor="hand2"
            )
            btn.grid(row=0, column=i, padx=(0, 2))
            btn.bind("<Button-1>", lambda e, idx=i: self._select(idx))
            self.buttons.append((btn, color))

        self._update_visual()

    def _select(self, idx):
        self.selected = idx
        self._update_visual()
        if self.command:
            self.command(idx)

    def _update_visual(self):
        for i, (btn, color) in enumerate(self.buttons):
            if i == self.selected:
                btn.config(bg=color, fg="white")
            else:
                btn.config(bg=COLORS["bg_card"], fg=COLORS["text_dim"])

    def get_selected(self):
        return self.selected

    def set_selected(self, idx):
        self.selected = idx
        self._update_visual()


class OmniDictateLauncher:
    """Main launcher window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OmniDictate Launcher")
        self.root.geometry("420x590")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS["bg"])
        self.process = None

        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 210
        y = (self.root.winfo_screenheight() // 2) - 295
        self.root.geometry(f"420x590+{x}+{y}")

        # Icon (optional)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        self._load_config()
        self._build_ui()
        self._update_status()

    def _load_config(self):
        """Load saved settings or use defaults."""
        self.config = {"language": 0, "device": 0, "ai_mode": False, "irodori": False}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.config.update(saved)
            except Exception:
                pass

    def _save_config(self):
        """Persist current settings."""
        self.config["language"] = self.lang_selector.get_selected()
        self.config["device"] = self.device_selector.get_selected()
        self.config["ai_mode"] = self.ai_toggle.get_state()
        self.config["irodori"] = self.irodori_toggle.get_state()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_ui(self):
        # ===== Header =====
        header = tk.Frame(self.root, bg=COLORS["bg"], pady=16)
        header.pack(fill="x")

        title = tk.Label(header, text="🎙 OmniDictate", font=("Segoe UI", 22, "bold"),
                         bg=COLORS["bg"], fg=COLORS["text"])
        title.pack()

        subtitle = tk.Label(header, text="Real-time Voice Dictation",
                            font=("Segoe UI", 10), bg=COLORS["bg"], fg=COLORS["text_dim"])
        subtitle.pack()

        # ===== Settings Card =====
        card = tk.Frame(self.root, bg=COLORS["bg_card"], padx=28, pady=20,
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack(fill="x", padx=24, pady=(4, 12))

        # --- Language ---
        self._section_label(card, "言語 / Language", row=0)

        self.lang_selector = OptionSelector(
            card,
            options=[("🇯🇵 日本語", "#e94560"), ("🇺🇸 English", "#4ecdc4")],
            initial=self.config["language"],
            command=lambda _: self._on_setting_changed()
        )
        self.lang_selector.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 16))

        # --- Device ---
        self._section_label(card, "デバイス / Device", row=2)

        self.device_selector = OptionSelector(
            card,
            options=[("⚡ GPU", COLORS["gpu"]), ("🖥 CPU", COLORS["cpu"])],
            initial=self.config["device"],
            command=lambda _: self._on_setting_changed()
        )
        self.device_selector.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 16))

        # --- AI Mode ---
        ai_row = tk.Frame(card, bg=COLORS["bg_card"])
        ai_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        ai_label_frame = tk.Frame(ai_row, bg=COLORS["bg_card"])
        ai_label_frame.pack(side="left")

        tk.Label(ai_label_frame, text="高精度モード / AI Correction",
                 font=("Segoe UI", 10, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["text_label"]).pack(anchor="w")
        tk.Label(ai_label_frame, text="Ollama で漢字・文法を校正",
                 font=("Segoe UI", 8),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(anchor="w")

        self.ai_toggle = ToggleButton(
            ai_row, initial=self.config["ai_mode"],
            command=lambda _: self._on_setting_changed()
        )
        self.ai_toggle.pack(side="right", padx=(12, 0))

        # --- Irodori 直送 ---
        # 貼り付けの代わりに Irodori（音声チャット）の入力欄へ直接送る。起動時の初期状態で、
        # 起動中も Ctrl+Alt+F9 で切り替えられる。日本語版（dictation.py）のみ対応。
        iro_row = tk.Frame(card, bg=COLORS["bg_card"])
        iro_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 4))

        iro_label_frame = tk.Frame(iro_row, bg=COLORS["bg_card"])
        iro_label_frame.pack(side="left")

        tk.Label(iro_label_frame, text="Irodori へ直送 / Send to Irodori",
                 font=("Segoe UI", 10, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["text_label"]).pack(anchor="w")
        tk.Label(iro_label_frame, text="フォーカスに関係なく入力欄へ（Ctrl+Alt+F9 で切替）",
                 font=("Segoe UI", 8),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(anchor="w")

        self.irodori_toggle = ToggleButton(
            iro_row, initial=self.config["irodori"],
            command=lambda _: self._on_setting_changed()
        )
        self.irodori_toggle.pack(side="right", padx=(12, 0))

        # ===== Info Card =====
        info_card = tk.Frame(self.root, bg=COLORS["bg_card"], padx=28, pady=14,
                             highlightbackground=COLORS["border"], highlightthickness=1)
        info_card.pack(fill="x", padx=24, pady=(0, 12))

        self.status_dot = tk.Label(info_card, text="●", font=("Segoe UI", 10),
                                   bg=COLORS["bg_card"], fg=COLORS["status_ready"])
        self.status_dot.pack(side="left")

        self.status_label = tk.Label(info_card, text="準備完了",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.status_label.pack(side="left", padx=(6, 0))

        self.config_summary = tk.Label(info_card, text="",
                                        font=("Segoe UI", 9),
                                        bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.config_summary.pack(side="right")

        # ===== Launch Button =====
        self.launch_btn = tk.Label(
            self.root, text="▶  起 動", font=("Segoe UI", 14, "bold"),
            bg=COLORS["accent"], fg="white",
            padx=40, pady=14, cursor="hand2"
        )
        self.launch_btn.pack(pady=(8, 6))
        self.launch_btn.bind("<Button-1>", self._on_launch)
        self.launch_btn.bind("<Enter>", lambda e: self.launch_btn.config(bg=COLORS["accent_hover"]))
        self.launch_btn.bind("<Leave>", lambda e: self.launch_btn.config(
            bg=COLORS["status_run"] if self.process else COLORS["accent"]))

        # ===== Footer =====
        footer = tk.Label(self.root, text="マイク: 常時ON  |  Esc: 終了  |  Ctrl+Alt+F9: Irodori 直送",
                          font=("Segoe UI", 8), bg=COLORS["bg"], fg=COLORS["text_dim"])
        footer.pack(side="bottom", pady=(0, 10))

    def _section_label(self, parent, text, row):
        lbl = tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                       bg=COLORS["bg_card"], fg=COLORS["text_label"])
        lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))

    def _on_setting_changed(self):
        self._save_config()
        self._update_status()

    def _update_status(self):
        lang = "日本語" if self.lang_selector.get_selected() == 0 else "English"
        dev = "GPU" if self.device_selector.get_selected() == 0 else "CPU"
        ai = " + AI" if self.ai_toggle.get_state() else ""
        iro = " + Irodori" if self.irodori_toggle.get_state() and self.lang_selector.get_selected() == 0 else ""
        self.config_summary.config(text=f"{lang} | {dev}{ai}{iro}")

    def _build_command(self):
        """Build the command line based on current settings."""
        lang_idx = self.lang_selector.get_selected()
        dev_idx = self.device_selector.get_selected()
        ai_on = self.ai_toggle.get_state()

        script = "dictation.py" if lang_idx == 0 else "dictation_en.py"
        args = [VENV_PYTHON, os.path.join(APP_DIR, script)]

        if dev_idx == 1:  # CPU
            args.append("--cpu")
        if ai_on:
            args.append("--ai")
        if self.irodori_toggle.get_state() and lang_idx == 0:  # 英語版は未対応
            args.append("--irodori")

        return args

    def _on_launch(self, event=None):
        if self.process and self.process.poll() is None:
            # Already running — stop it
            self._stop_process()
            return
        self._start_process()

    def _start_process(self):
        self._save_config()
        cmd = self._build_command()

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        try:
            self.process = subprocess.Popen(
                cmd, cwd=APP_DIR, env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            self.launch_btn.config(text="■  停 止", bg=COLORS["status_run"])
            self.status_label.config(text="実行中...")
            self.status_dot.config(fg=COLORS["status_run"])
            self._poll_process()
        except Exception as e:
            self.status_label.config(text=f"起動失敗: {e}")
            self.status_dot.config(fg=COLORS["accent"])

    def _stop_process(self):
        """Stop the dictation process (user-initiated)."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process = None
        self.launch_btn.config(text="▶  起 動", bg=COLORS["accent"])
        self.status_label.config(text="停止しました")
        self.status_dot.config(fg=COLORS["status_ready"])

    def _poll_process(self):
        """Check if the dictation process is still running."""
        if self.process and self.process.poll() is not None:
            self.process = None
            self.launch_btn.config(text="▶  起 動", bg=COLORS["accent"])
            self.status_label.config(text="準備完了")
            self.status_dot.config(fg=COLORS["status_ready"])
        else:
            self.root.after(1000, self._poll_process)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = OmniDictateLauncher()
    app.run()
