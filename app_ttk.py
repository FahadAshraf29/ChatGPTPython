from __future__ import annotations

import datetime as _dt
import os
import queue
import re
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Tuple

from ai_backend import CancelFlag, ChatMessage, OllamaBackend
from settings_store import load_settings, save_settings
from speech import SpeechConfig, SpeechError, SpeechRecognizer
from tts import TTSConfig, TTSEngine


# -------------------- Small helpers --------------------

def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%H:%M")


def _safe_title_from_text(text: str, fallback: str = "New chat") -> str:
    t = " ".join(text.strip().split())
    if not t:
        return fallback
    if len(t) <= 26:
        return t
    return t[:26].rstrip() + "…"


# -------------------- Theme --------------------


@dataclass(frozen=True)
class Theme:
    bg: str
    panel: str
    panel_2: str
    field: str
    fg: str
    muted: str
    accent: str
    accent_2: str
    user_bubble: str
    assistant_bubble: str
    bubble_text: str


DARK = Theme(
    bg="#0B0F14",
    panel="#111826",
    panel_2="#0F1623",
    field="#0A0F16",
    fg="#E6EDF6",
    muted="#9AA5B1",
    accent="#4F8CFF",
    accent_2="#7AA2FF",
    user_bubble="#1F6FEB",
    assistant_bubble="#1B2433",
    bubble_text="#E6EDF6",
)


# -------------------- Bubble chat widgets --------------------


class _BubbleMessage:
    """A chat message bubble that can be updated while streaming."""

    def __init__(
        self,
        row: tk.Frame,
        role: str,
        text_label: tk.Label,
        time_label: tk.Label,
        theme: Theme,
    ) -> None:
        self.row = row
        self.role = role
        self._label = text_label
        self._time = time_label
        self._theme = theme
        self._text = ""
        self._typing = False
        self._typing_job: Optional[str] = None

    @property
    def text(self) -> str:
        return self._text

    def set_time(self, stamp: str) -> None:
        self._time.configure(text=stamp)

    def set_text(self, text: str) -> None:
        self._text = text
        self._label.configure(text=text)

    def append(self, delta: str) -> None:
        if not delta:
            return
        self._text += delta
        self._label.configure(text=self._text)

    def start_typing(self) -> None:
        if self._typing:
            return
        self._typing = True
        self._text = ""
        self._label.configure(text="")
        self._schedule_typing()

    def stop_typing(self) -> None:
        self._typing = False
        if self._typing_job is not None:
            try:
                self._label.after_cancel(self._typing_job)
            except Exception:
                pass
        self._typing_job = None

    def _schedule_typing(self, step: int = 0) -> None:
        if not self._typing:
            return
        dots = ["·", "··", "···"]
        self._label.configure(text=dots[step % 3])
        self._typing_job = self._label.after(350, lambda: self._schedule_typing(step + 1))


class BubbleChatView(ttk.Frame):
    """Scrollable list of message bubbles (ChatGPT-ish)."""

    def __init__(self, master: tk.Misc, theme: Theme) -> None:
        super().__init__(master)
        self._theme = theme

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, background=theme.bg)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=theme.bg)
        self._inner_win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Better mousewheel on Windows/mac/Linux.
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add=True)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add=True)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add=True)

        self._bubbles: List[_BubbleMessage] = []
        self._wrap_px = 560

    def clear(self) -> None:
        for child in list(self.inner.winfo_children()):
            child.destroy()
        self._bubbles = []
        self._scroll_to_end()

    def bubbles(self) -> List[_BubbleMessage]:
        return list(self._bubbles)

    def add_message(self, role: str, text: str, stamp: Optional[str] = None) -> _BubbleMessage:
        """Add a bubble. role in {"user","assistant","system"}."""

        stamp = stamp or _now_stamp()

        row = tk.Frame(self.inner, bg=self._theme.bg)
        row.pack(fill="x", pady=8, padx=14)

        is_user = role == "user"
        is_system = role == "system"

        # Alignment containers
        left = tk.Frame(row, bg=self._theme.bg)
        right = tk.Frame(row, bg=self._theme.bg)
        left.pack(side="left", fill="x", expand=True)
        right.pack(side="right", fill="x", expand=True)

        avatar_txt = "You" if is_user else ("SYS" if is_system else "AI")
        avatar_bg = self._theme.user_bubble if is_user else (self._theme.panel if is_system else self._theme.accent)

        avatar = tk.Label(
            (right if is_user else left),
            text=avatar_txt,
            bg=avatar_bg,
            fg=self._theme.bubble_text,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
        )

        bubble_bg = (
            self._theme.user_bubble
            if is_user
            else (self._theme.panel if is_system else self._theme.assistant_bubble)
        )

        bubble = tk.Frame(
            (right if is_user else left),
            bg=bubble_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground="#1E2A3D" if not is_user else "#2A76FF",
            highlightcolor="#1E2A3D" if not is_user else "#2A76FF",
        )

        # Message text label
        msg = tk.Label(
            bubble,
            text=text,
            bg=bubble_bg,
            fg=self._theme.bubble_text,
            justify="left",
            anchor="w",
            wraplength=self._wrap_px,
            font=("Segoe UI", 11),
        )
        msg.pack(fill="both", expand=True, padx=12, pady=(10, 6))

        meta = tk.Label(
            bubble,
            text=stamp,
            bg=bubble_bg,
            fg=self._theme.muted,
            font=("Segoe UI", 9),
            anchor="e",
        )
        meta.pack(fill="x", padx=12, pady=(0, 8))

        # Pack avatar + bubble. User bubbles align right.
        if is_user:
            bubble.pack(side="right", padx=(12, 0))
            avatar.pack(side="right")
        else:
            avatar.pack(side="left")
            bubble.pack(side="left", padx=(12, 0))

        bubble_obj = _BubbleMessage(row=row, role=role, text_label=msg, time_label=meta, theme=self._theme)
        bubble_obj.set_text(text)
        bubble_obj.set_time(stamp)
        self._bubbles.append(bubble_obj)

        self._scroll_to_end()
        return bubble_obj

    def _scroll_to_end(self) -> None:
        self.update_idletasks()
        try:
            self.canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _on_inner_configure(self, _e: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e: tk.Event) -> None:
        # Keep the inner frame width in sync with the canvas.
        try:
            self.canvas.itemconfigure(self._inner_win, width=e.width)
        except Exception:
            return

        # Update wrap length to keep bubbles readable.
        # Keep a margin for avatar + padding.
        new_wrap = max(280, min(720, int(e.width * 0.62)))
        if abs(new_wrap - self._wrap_px) < 12:
            return
        self._wrap_px = new_wrap
        for b in self._bubbles:
            try:
                b._label.configure(wraplength=self._wrap_px)
            except Exception:
                pass

    def _on_mousewheel(self, event: tk.Event) -> None:
        # Windows: event.delta, Linux: Button-4/5
        try:
            if getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
                return
            if getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
                return
            delta = int(getattr(event, "delta", 0))
            if delta != 0:
                self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        except Exception:
            return

    def export_plain_text(self) -> str:
        out: List[str] = []
        for b in self._bubbles:
            role = "You" if b.role == "user" else ("System" if b.role == "system" else "Assistant")
            out.append(f"{role} [{b._time.cget('text')}]:\n{b.text}\n")
        return "\n".join(out).strip() + "\n"


# -------------------- Chat pane (one tab) --------------------


class ChatPane(ttk.Frame):
    def __init__(self, master: tk.Misc, app: "ChatGPTishApp", title: str) -> None:
        super().__init__(master)
        self.app = app
        self.title = title

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.view = BubbleChatView(self, theme=self.app.theme)
        self.view.grid(row=0, column=0, sticky="nsew")

        self._build_input_bar()

        self.messages: List[ChatMessage] = []
        self._active_assistant_bubble: Optional[_BubbleMessage] = None
        self._tts_stream_buffer = ""

        self.view.add_message(
            "system",
            "Free local AI backend: **Ollama** (no OpenAI credits).\n"
            "Tip: Install Ollama, then run `ollama pull llama3.1:8b`, then Refresh Models.",
        )

    def _build_input_bar(self) -> None:
        bar = tk.Frame(self, bg=self.app.theme.bg)
        bar.grid(row=1, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        # Input field (multiline like ChatGPT)
        self.input = tk.Text(
            bar,
            height=3,
            wrap="word",
            bg=self.app.theme.field,
            fg=self.app.theme.fg,
            insertbackground=self.app.theme.fg,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=("Segoe UI", 11),
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(14, 10), pady=12)

        # Keybinds: Enter to send, Shift+Enter for newline
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", self._on_shift_enter)

        btns = tk.Frame(bar, bg=self.app.theme.bg)
        btns.grid(row=0, column=1, sticky="e", padx=(0, 14), pady=12)

        self.send_btn = ttk.Button(btns, text="Send", command=self.on_send)
        self.send_btn.pack(side="left", padx=(0, 8))

        self.mic_btn = ttk.Button(btns, text="🎙︎", width=4, command=self.on_speak)
        self.mic_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side="left")

    def _on_enter(self, event: tk.Event) -> str:
        # Prevent newline insertion.
        self.on_send()
        return "break"

    def _on_shift_enter(self, event: tk.Event) -> None:
        # Allow newline.
        return

    def set_controls_state(self, generating: bool, listening: bool) -> None:
        # Keep input editable even while generating (fixes "can't see what I type").
        try:
            self.send_btn.configure(state="disabled" if (generating or listening) else "normal")
            self.mic_btn.configure(state="disabled" if (generating or listening) else "normal")
            self.stop_btn.configure(state="normal" if (generating or listening) else "disabled")
        except Exception:
            pass

    def on_send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        if self.app.state_generating or self.app.state_listening:
            return

        # Clear box immediately.
        self.input.delete("1.0", "end")

        # Set tab title from first message (ChatGPT-ish).
        if self.title.startswith("Chat"):
            new_title = _safe_title_from_text(text, fallback=self.title)
            self.app.rename_chat(self, new_title)

        self.view.add_message("user", text)
        self.messages.append(ChatMessage(role="user", content=text))

        self.app.start_generation(chat=self)

    def on_speak(self) -> None:
        if self.app.state_generating or self.app.state_listening:
            return
        self.app.start_listening(chat=self)

    def on_stop(self) -> None:
        self.app.stop_everything()

    # Streaming helpers
    def begin_assistant_stream(self) -> None:
        self._tts_stream_buffer = ""
        self._active_assistant_bubble = self.view.add_message("assistant", "", stamp=_now_stamp())
        self._active_assistant_bubble.start_typing()

    def append_assistant_delta(self, delta: str) -> None:
        if self._active_assistant_bubble is None:
            self.begin_assistant_stream()
        assert self._active_assistant_bubble is not None

        # First real token stops typing indicator.
        if self._active_assistant_bubble.text in ("", "·", "··", "···"):
            self._active_assistant_bubble.stop_typing()
            self._active_assistant_bubble.set_text("")

        self._active_assistant_bubble.append(delta)

        # Optional: stream speech in sentence chunks.
        if self.app.speak_enabled() and self.app.speak_live_var.get():
            self._tts_stream_buffer += delta
            for chunk, rest in _pop_speakable_chunks(self._tts_stream_buffer):
                self.app.tts.speak(chunk)
                self._tts_stream_buffer = rest

    def end_assistant_stream(self, full_text: str, cancelled: bool) -> None:
        if self._active_assistant_bubble is not None:
            self._active_assistant_bubble.stop_typing()

        if cancelled:
            self.view.add_message("system", "Generation cancelled.")
            self._active_assistant_bubble = None
            self._tts_stream_buffer = ""
            return

        full = (full_text or "").strip()
        if not full:
            self._active_assistant_bubble = None
            self._tts_stream_buffer = ""
            return

        # Ensure UI bubble matches final content.
        if self._active_assistant_bubble is None:
            self._active_assistant_bubble = self.view.add_message("assistant", full)
        else:
            self._active_assistant_bubble.set_text(full)
            self._active_assistant_bubble.set_time(_now_stamp())

        self.messages.append(ChatMessage(role="assistant", content=full))

        # If not speaking live, speak after completion.
        if self.app.speak_enabled() and (not self.app.speak_live_var.get()):
            self.app.tts.speak(full)

        # If speaking live, flush any remaining buffer.
        if self.app.speak_enabled() and self.app.speak_live_var.get():
            tail = self._tts_stream_buffer.strip()
            if tail:
                self.app.tts.speak(tail)
        self._tts_stream_buffer = ""
        self._active_assistant_bubble = None


# -------------------- Sentence chunking for live TTS --------------------


_SENT_END = re.compile(r"[.!?]+\s+|\n{2,}")


def _pop_speakable_chunks(buffer: str) -> List[Tuple[str, str]]:
    """Return (chunk, rest) pairs to speak, based on sentence-ish boundaries."""

    # Keep buffer short-ish.
    if len(buffer) < 80:
        return []

    chunks: List[Tuple[str, str]] = []
    last_end = 0
    for m in _SENT_END.finditer(buffer):
        end = m.end()
        # Only speak reasonably sized chunks.
        if end - last_end >= 60:
            chunk = buffer[last_end:end].strip()
            rest = buffer[end:]
            chunks.append((chunk, rest))
            return chunks

    return []


# -------------------- Main application --------------------


class ChatGPTishApp(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master = master
        self.theme = DARK

        # State
        self.state_generating = False
        self.state_listening = False
        self.cancel_flag: Optional[CancelFlag] = None

        # Async UI queue
        self.ui_q: "queue.Queue[tuple]" = queue.Queue()

        # Backends
        self.backend = OllamaBackend()
        self.speech = SpeechRecognizer()
        self.tts = TTSEngine(TTSConfig(enabled=True, rate=165))
        self.tts.start()

        # Settings
        self.system_prompt = (
            "You are a helpful desktop assistant. Be concise by default, but provide detail when asked."
        )

        self.base_url_var = tk.StringVar(value=self.backend.base_url)
        self.model_var = tk.StringVar(value=self.backend.model)
        self.temperature_var = tk.DoubleVar(value=0.6)
        self.keep_context_var = tk.BooleanVar(value=True)
        self.speak_var = tk.BooleanVar(value=True)
        self.speak_live_var = tk.BooleanVar(value=True)  # speak while streaming (sentence-ish)
        self.tts_rate_var = tk.IntVar(value=165)

        self._apply_theme()
        self._build_ui()
        self._load_settings()

        # Create initial chat
        self._chat_counter = 0
        self.new_chat()

        # Poll UI queue
        self.after(50, self._process_ui_queue)
        self.after(900, self._periodic_healthcheck)

        # Auto populate models
        self.test_backend()
        self.refresh_models()

    # ---------------- UI ----------------
    def _apply_theme(self) -> None:
        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        t = self.theme
        self.master.configure(bg=t.bg)

        style.configure("TFrame", background=t.bg)
        style.configure("TLabel", background=t.bg, foreground=t.fg)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=t.fg)
        style.configure("Subtle.TLabel", foreground=t.muted)

        style.configure("TButton", padding=(12, 8))
        style.map("TButton", foreground=[("disabled", t.muted)])

        style.configure("TEntry", fieldbackground=t.field, foreground=t.fg, padding=(10, 8))
        style.configure("TCombobox", padding=(10, 8))
        style.configure("Horizontal.TScale", background=t.bg)

        style.configure("TNotebook", background=t.bg, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=t.panel,
            foreground=t.fg,
            padding=(14, 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", t.panel_2)],
            foreground=[("selected", t.fg)],
        )

        style.configure("TLabelframe", background=t.bg, foreground=t.fg)
        style.configure("TLabelframe.Label", background=t.bg, foreground=t.fg)

    def _build_ui(self) -> None:
        self.pack(fill="both", expand=True)

        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=14, pady=(14, 8))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Local Chat Assistant", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        self.status_lbl = ttk.Label(top, text="Ollama: checking…", style="Subtle.TLabel")
        self.status_lbl.grid(row=0, column=1, sticky="e", padx=(0, 10))

        self.new_chat_btn = ttk.Button(top, text="+ New chat", command=self.new_chat)
        self.new_chat_btn.grid(row=0, column=2, sticky="e")

        # Main tabs ("a lot of tabs")
        self.main_tabs = ttk.Notebook(self)
        self.main_tabs.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.chat_tab = ttk.Frame(self.main_tabs)
        self.settings_tab = ttk.Frame(self.main_tabs)
        self.voice_tab = ttk.Frame(self.main_tabs)
        self.about_tab = ttk.Frame(self.main_tabs)

        self.main_tabs.add(self.chat_tab, text="Chats")
        self.main_tabs.add(self.settings_tab, text="Settings")
        self.main_tabs.add(self.voice_tab, text="Voice")
        self.main_tabs.add(self.about_tab, text="About")

        self._build_chat_tab(self.chat_tab)
        self._build_settings_tab(self.settings_tab)
        self._build_voice_tab(self.voice_tab)
        self._build_about_tab(self.about_tab)

    def _build_chat_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Chat tabs (per conversation)
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        bar.columnconfigure(2, weight=1)

        ttk.Label(bar, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_combo_top = ttk.Combobox(bar, textvariable=self.model_var, state="readonly", width=26)
        self.model_combo_top.grid(row=0, column=1, sticky="w", padx=(6, 10))
        self.model_combo_top.bind("<<ComboboxSelected>>", lambda _e: self._on_model_selected())

        self.refresh_models_btn_top = ttk.Button(bar, text="Refresh", command=self.refresh_models)
        self.refresh_models_btn_top.grid(row=0, column=3, sticky="e", padx=(0, 8))

        self.stop_all_btn_top = ttk.Button(bar, text="Stop", command=self.stop_everything)
        self.stop_all_btn_top.grid(row=0, column=4, sticky="e")

        self.chat_tabs = ttk.Notebook(parent)
        self.chat_tabs.grid(row=1, column=0, sticky="nsew")
        self.chat_tabs.bind("<<NotebookTabChanged>>", lambda _e: self._sync_controls_state())

        self._chats: List[ChatPane] = []

        # Right-click to close tab.
        self.chat_tabs.bind("<Button-3>", self._on_tab_right_click)

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        t = self.theme

        # Connection / model
        model_box = ttk.LabelFrame(parent, text="Ollama")
        model_box.grid(row=0, column=0, sticky="ew", pady=(10, 0), padx=10)
        model_box.columnconfigure(1, weight=1)

        ttk.Label(model_box, text="Base URL").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))
        self.base_url_entry = ttk.Entry(model_box, textvariable=self.base_url_var)
        self.base_url_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 6))

        ttk.Label(model_box, text="Model").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.model_combo = ttk.Combobox(model_box, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=1, column=1, sticky="ew", padx=10, pady=6)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_selected())

        btns = ttk.Frame(model_box)
        btns.grid(row=2, column=1, sticky="e", padx=10, pady=(6, 10))
        ttk.Button(btns, text="Test", command=self.test_backend).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Refresh models", command=self.refresh_models).pack(side="right")

        # Generation
        gen_box = ttk.LabelFrame(parent, text="Generation")
        gen_box.grid(row=1, column=0, sticky="ew", pady=(10, 0), padx=10)
        gen_box.columnconfigure(0, weight=1)

        ttk.Label(gen_box, text="Temperature").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.temp_scale = ttk.Scale(gen_box, from_=0.0, to=1.2, variable=self.temperature_var)
        self.temp_scale.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        self.keep_context_chk = ttk.Checkbutton(gen_box, text="Keep conversation context", variable=self.keep_context_var)
        self.keep_context_chk.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))

        # Tools
        tools_box = ttk.LabelFrame(parent, text="Tools")
        tools_box.grid(row=2, column=0, sticky="ew", pady=(10, 10), padx=10)
        tools_box.columnconfigure(0, weight=1)
        tools_box.columnconfigure(1, weight=1)

        ttk.Button(tools_box, text="Export current chat…", command=self.export_current_chat).grid(
            row=0, column=0, sticky="ew", padx=(10, 6), pady=10
        )
        ttk.Button(tools_box, text="Clear current chat", command=self.clear_current_chat).grid(
            row=0, column=1, sticky="ew", padx=(6, 10), pady=10
        )

        ttk.Button(tools_box, text="System prompt…", command=self.edit_system_prompt).grid(
            row=1, column=0, sticky="ew", padx=(10, 6), pady=(0, 10)
        )
        ttk.Button(tools_box, text="Save settings", command=self.save_settings_ui).grid(
            row=1, column=1, sticky="ew", padx=(6, 10), pady=(0, 10)
        )

    def _build_voice_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        voice = ttk.LabelFrame(parent, text="Speech")
        voice.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        voice.columnconfigure(0, weight=1)

        self.speak_chk = ttk.Checkbutton(voice, text="Speak responses (offline TTS)", variable=self.speak_var, command=self._on_tts_toggle)
        self.speak_chk.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.speak_live_chk = ttk.Checkbutton(
            voice,
            text="Speak while generating (sentence chunks)",
            variable=self.speak_live_var,
        )
        self.speak_live_chk.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        ttk.Label(voice, text="Speech rate").grid(row=2, column=0, sticky="w", padx=10)
        self.tts_rate = ttk.Scale(
            voice,
            from_=110,
            to=220,
            command=lambda v: self._on_tts_rate(v),
        )
        self.tts_rate.set(self.tts_rate_var.get())
        self.tts_rate.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        stt = ttk.LabelFrame(parent, text="Voice input")
        stt.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        stt.columnconfigure(0, weight=1)

        ttk.Label(
            stt,
            text=(
                "Mic button uses SpeechRecognition's default Google recognizer (free, online).\n"
                "If you want fully offline STT, ask and I can switch it to Vosk."
            ),
            style="Subtle.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

    def _build_about_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        box = ttk.Frame(parent)
        box.pack(fill="both", expand=True, padx=16, pady=16)

        txt = (
            "This UI is a ChatGPT-style Tkinter app:\n\n"
            "• Multiple tabs + multiple chat tabs\n"
            "• Bubble messages + typing indicator\n"
            "• Streaming generation from Ollama (free/local)\n"
            "• Offline TTS (pyttsx3)\n\n"
            "Tip: You can store Ollama models on another drive by setting OLLAMA_MODELS."
        )
        ttk.Label(box, text=txt, justify="left").pack(anchor="nw")

    # ---------------- Chat/session management ----------------
    def new_chat(self) -> None:
        self._chat_counter += 1
        title = f"Chat {self._chat_counter}"
        pane = ChatPane(self.chat_tabs, app=self, title=title)
        self._chats.append(pane)
        self.chat_tabs.add(pane, text=title)
        self.chat_tabs.select(pane)
        self._sync_controls_state()

    def current_chat(self) -> Optional[ChatPane]:
        try:
            idx = self.chat_tabs.index(self.chat_tabs.select())
        except Exception:
            return None
        if 0 <= idx < len(self._chats):
            return self._chats[idx]
        return None

    def rename_chat(self, chat: ChatPane, new_title: str) -> None:
        chat.title = new_title
        try:
            tab_id = self.chat_tabs.index(chat)
            self.chat_tabs.tab(tab_id, text=new_title)
        except Exception:
            pass

    def _on_tab_right_click(self, event: tk.Event) -> None:
        # Close tab on right click.
        try:
            tab_id = self.chat_tabs.index(f"@{event.x},{event.y}")
        except Exception:
            return
        if len(self._chats) <= 1:
            messagebox.showinfo("Chats", "You must keep at least one chat open.")
            return
        if self.state_generating or self.state_listening:
            messagebox.showinfo("Chats", "Stop the current action before closing a chat.")
            return

        pane = self._chats[tab_id]
        if messagebox.askyesno("Close chat", f"Close '{self.chat_tabs.tab(tab_id, 'text')}'?"):
            self.chat_tabs.forget(tab_id)
            try:
                self._chats.remove(pane)
            except ValueError:
                pass
            pane.destroy()
            self._sync_controls_state()

    # ---------------- Settings ----------------
    def speak_enabled(self) -> bool:
        return bool(self.speak_var.get())

    def _on_tts_toggle(self) -> None:
        self.tts.set_enabled(bool(self.speak_var.get()))
        if not self.speak_var.get():
            self.tts.interrupt()

    def _on_tts_rate(self, value: str) -> None:
        try:
            r = int(float(value))
        except Exception:
            return
        self.tts_rate_var.set(r)
        self.tts.set_rate(r)

    def _on_model_selected(self) -> None:
        self.backend.model = self.model_var.get().strip() or self.backend.model

    def _sync_controls_state(self) -> None:
        chat = self.current_chat()
        if chat is not None:
            chat.set_controls_state(self.state_generating, self.state_listening)

        # Global buttons
        try:
            self.new_chat_btn.configure(state="disabled" if (self.state_generating or self.state_listening) else "normal")
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        self.status_lbl.configure(text=text)

    def _load_settings(self) -> None:
        cfg = load_settings()
        if not isinstance(cfg, dict):
            return

        self.base_url_var.set(str(cfg.get("base_url", self.base_url_var.get())))
        self.model_var.set(str(cfg.get("model", self.model_var.get())))
        self.temperature_var.set(float(cfg.get("temperature", self.temperature_var.get())))
        self.keep_context_var.set(bool(cfg.get("keep_context", self.keep_context_var.get())))
        self.speak_var.set(bool(cfg.get("speak", self.speak_var.get())))
        self.speak_live_var.set(bool(cfg.get("speak_live", self.speak_live_var.get())))

        try:
            rate = int(cfg.get("tts_rate", self.tts_rate_var.get()))
            self.tts_rate_var.set(rate)
            self.tts.set_rate(rate)
            self.tts_rate.set(rate)
        except Exception:
            pass

        sys_prompt = cfg.get("system_prompt")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            self.system_prompt = sys_prompt.strip()

        # Apply
        self.backend.base_url = self.base_url_var.get().strip() or self.backend.base_url
        self.backend.model = self.model_var.get().strip() or self.backend.model
        self.tts.set_enabled(bool(self.speak_var.get()))

    def save_settings_ui(self) -> None:
        data = {
            "base_url": self.base_url_var.get().strip(),
            "model": self.model_var.get().strip(),
            "temperature": float(self.temperature_var.get()),
            "keep_context": bool(self.keep_context_var.get()),
            "speak": bool(self.speak_var.get()),
            "speak_live": bool(self.speak_live_var.get()),
            "tts_rate": int(self.tts_rate_var.get()),
            "system_prompt": self.system_prompt,
        }
        save_settings(data)
        chat = self.current_chat()
        if chat:
            chat.view.add_message("system", "Settings saved.")

    # ---------------- Tools ----------------
    def clear_current_chat(self) -> None:
        if self.state_generating or self.state_listening:
            return
        chat = self.current_chat()
        if not chat:
            return
        chat.messages = []
        chat.view.clear()
        chat.view.add_message("system", "Chat cleared.")

    def export_current_chat(self) -> None:
        chat = self.current_chat()
        if not chat:
            return
        text = chat.view.export_plain_text()
        if not text.strip():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def edit_system_prompt(self) -> None:
        if self.state_generating or self.state_listening:
            return

        win = tk.Toplevel(self.master)
        win.title("System Prompt")
        win.geometry("780x460")
        win.transient(self.master)
        win.grab_set()

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(1, weight=1)

        ttk.Label(
            frm,
            text=(
                "This prompt is sent as a system message for the local model.\n"
                "Keep it short and specific."
            ),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        txt = scrolledtext.ScrolledText(frm, wrap="word")
        txt.grid(row=1, column=0, sticky="nsew")
        txt.insert("1.0", self.system_prompt)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(10, 0))

        def on_apply() -> None:
            new_prompt = txt.get("1.0", "end").strip()
            if not new_prompt:
                messagebox.showwarning("System Prompt", "System prompt can't be empty.")
                return
            self.system_prompt = new_prompt
            chat = self.current_chat()
            if chat:
                chat.view.add_message("system", "System prompt updated (affects next response).")
            win.destroy()

        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Apply", command=on_apply).pack(side="right")

    # ---------------- Ollama actions ----------------
    def refresh_models(self) -> None:
        def worker() -> None:
            try:
                self.backend.base_url = self.base_url_var.get().strip() or self.backend.base_url
                models = self.backend.list_models()
                self.ui_q.put(("models", models))
            except Exception as e:
                self.ui_q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def test_backend(self) -> None:
        def worker() -> None:
            self.backend.base_url = self.base_url_var.get().strip() or self.backend.base_url
            ok = self.backend.healthcheck()
            self.ui_q.put(("health", ok))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Speech + generation orchestration ----------------
    def start_listening(self, chat: ChatPane) -> None:
        self.state_listening = True
        self._sync_controls_state()
        self._set_status("Listening…")

        def worker() -> None:
            try:
                self.speech.config = SpeechConfig(
                    language="en-US",
                    timeout_s=6,
                    phrase_time_limit_s=12,
                    ambient_adjust_s=0.5,
                )
                text = self.speech.listen_once()
                self.ui_q.put(("speech", chat, text))
            except SpeechError as e:
                self.ui_q.put(("speech_error", chat, str(e)))
            except Exception as e:
                self.ui_q.put(("speech_error", chat, f"Speech error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def start_generation(self, chat: ChatPane) -> None:
        self.state_generating = True
        self._sync_controls_state()
        self._set_status("Generating…")

        # Ensure speech doesn't drift out of sync.
        self.tts.interrupt()

        self.cancel_flag = CancelFlag()

        # Snapshot request
        msgs = list(chat.messages)
        if not self.keep_context_var.get():
            # only keep the most recent user message
            msgs = [m for m in msgs[-1:]]
        if len(msgs) > 24:
            msgs = msgs[-24:]

        temp = float(self.temperature_var.get())
        system_prompt = self.system_prompt
        backend_url = self.base_url_var.get().strip() or self.backend.base_url
        model = self.model_var.get().strip() or self.backend.model

        def worker() -> None:
            try:
                self.backend.base_url = backend_url
                self.backend.model = model

                deltas: List[str] = []
                self.ui_q.put(("assistant_begin", chat))
                for d in self.backend.chat_stream(
                    msgs,
                    temperature=temp,
                    system_prompt_override=system_prompt,
                    cancel_flag=self.cancel_flag,
                ):
                    deltas.append(d)
                    self.ui_q.put(("assistant_delta", chat, d))
                full = "".join(deltas)
                self.ui_q.put(("assistant_done", chat, full))
            except Exception as e:
                self.ui_q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def stop_everything(self) -> None:
        # Cancel generation
        if self.cancel_flag is not None:
            self.cancel_flag.cancel()

        # Stop speech
        self.tts.interrupt()

        # Listening can't always be interrupted mid-capture; unblock UI anyway.
        if self.state_listening:
            self.state_listening = False
            self._sync_controls_state()
            self._set_status("Ready")

    # ---------------- Queue / timers ----------------
    def _process_ui_queue(self) -> None:
        while True:
            try:
                item = self.ui_q.get_nowait()
            except queue.Empty:
                break

            kind = item[0]

            if kind == "models":
                models: List[str] = item[1]
                self.model_combo.configure(values=models)
                self.model_combo_top.configure(values=models)
                if models and (self.model_var.get() not in models):
                    self.model_var.set(models[0])
                    self.backend.model = models[0]
                if models:
                    self._set_status(f"Ollama: {len(models)} model(s)")
                else:
                    self._set_status("Ollama reachable, but no models found")

            elif kind == "health":
                ok: bool = bool(item[1])
                self._set_status("Ollama: connected" if ok else "Ollama: not reachable")
                if not ok:
                    chat = self.current_chat()
                    if chat:
                        chat.view.add_message(
                            "system",
                            "Couldn't reach Ollama. Install it and ensure it's running at "
                            f"{self.base_url_var.get().strip() or self.backend.base_url}.",
                        )

            elif kind == "speech":
                _, chat, text = item
                if not self.state_listening:
                    continue
                self.state_listening = False
                self._sync_controls_state()
                self._set_status("Ready")
                text = str(text).strip()
                if text:
                    chat.view.add_message("user", text)
                    chat.messages.append(ChatMessage(role="user", content=text))
                    self.start_generation(chat)

            elif kind == "speech_error":
                _, chat, err = item
                self.state_listening = False
                self._sync_controls_state()
                self._set_status("Ready")
                chat.view.add_message("system", str(err))

            elif kind == "assistant_begin":
                _, chat = item
                chat.begin_assistant_stream()

            elif kind == "assistant_delta":
                _, chat, delta = item
                chat.append_assistant_delta(str(delta))

            elif kind == "assistant_done":
                _, chat, full = item
                cancelled = bool(self.cancel_flag is not None and self.cancel_flag.cancelled)
                chat.end_assistant_stream(full_text=str(full), cancelled=cancelled)

                self.state_generating = False
                self.cancel_flag = None
                self._sync_controls_state()
                self._set_status("Ready")

            elif kind == "error":
                err = str(item[1])
                self.state_generating = False
                self.state_listening = False
                self.cancel_flag = None
                self._sync_controls_state()
                self._set_status("Ready")
                chat = self.current_chat()
                if chat:
                    chat.view.add_message("system", err)

        self.after(50, self._process_ui_queue)

    def _periodic_healthcheck(self) -> None:
        if not (self.state_generating or self.state_listening):
            ok = self.backend.healthcheck()
            self._set_status("Ollama: connected" if ok else "Ollama: not reachable")
        self.after(4000, self._periodic_healthcheck)


def main() -> None:
    root = tk.Tk()
    root.title("Voice Assistant (Local + Free)")
    root.geometry("1180x780")
    root.minsize(980, 640)

    def resource_path(rel_path: str) -> str:
        base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
        return os.path.join(base, rel_path)

    try:
        root.iconphoto(True, tk.PhotoImage(file=resource_path("chatgpt2-icon.png")))
    except Exception:
        pass

    app = ChatGPTishApp(root)

    def on_close() -> None:
        try:
            app.save_settings_ui()
        except Exception:
            pass
        try:
            app.tts.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
