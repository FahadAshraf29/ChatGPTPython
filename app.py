"""ChatGPT-style desktop UI (Tk + customtkinter) with a free local LLM backend.

Highlights
- ChatGPT-ish layout: left sidebar with chat history + main chat view
- Streaming bubbles + typing indicator
- Multiline input that always shows typed characters
- Optional live TTS while streaming (sentence chunking) to keep speech consistent with what's displayed
- Free AI agent via local Ollama server (no OpenAI credits / no API keys)

Run:
    python app.py

Requirements:
    pip install -r requirements.txt
    (and install Ollama: https://ollama.com)
"""

from __future__ import annotations

import datetime as _dt
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover
    # Fail gracefully with a useful message.
    raise SystemExit(
        "customtkinter is required for the new ChatGPT-style UI.\n"
        "Install it with: pip install customtkinter pillow\n\n"
        f"Import error: {e}"
    )

from ai_backend import AIBackendError, CancelFlag, ChatMessage, OllamaBackend
from settings_store import load_settings, save_settings
from speech import SpeechConfig, SpeechError, SpeechRecognizer
from tts import TTSConfig, TTSEngine


# ------------------------------ Theme ------------------------------


@dataclass(frozen=True)
class Theme:
    # Core
    app_bg: str
    sidebar_bg: str
    topbar_bg: str
    input_bg: str
    card_bg: str
    card_bg_2: str

    # Text
    fg: str
    muted: str

    # Accents
    accent: str
    danger: str


CHATGPT_DARK = Theme(
    app_bg="#343541",
    sidebar_bg="#202123",
    topbar_bg="#2A2B32",
    input_bg="#40414F",
    card_bg="#444654",
    card_bg_2="#343541",
    fg="#ECECF1",
    muted="#B4B4C0",
    accent="#10A37F",
    danger="#EF4444",
)


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%H:%M")


def _safe_title_from_text(text: str, fallback: str = "New chat") -> str:
    t = " ".join(text.strip().split())
    if not t:
        return fallback
    if len(t) <= 28:
        return t
    return t[:28].rstrip() + "…"


# ---------------------- Sentence chunking for live TTS ----------------------


_SENT_END = re.compile(r"[.!?]+\s+|\n{2,}")


def _pop_speakable_chunks(buffer: str) -> List[Tuple[str, str]]:
    """Return (chunk, rest) pairs to speak from a buffer.

    This keeps speech aligned with what is already displayed while streaming.
    """

    if len(buffer) < 90:
        return []

    for m in _SENT_END.finditer(buffer):
        end = m.end()
        if end >= 70:
            chunk = buffer[:end].strip()
            rest = buffer[end:]
            if len(chunk) >= 40:
                return [(chunk, rest)]

    # Safety: speak long buffers in pieces even without punctuation.
    if len(buffer) > 260:
        cut = 220
        chunk = buffer[:cut].strip()
        rest = buffer[cut:]
        return [(chunk, rest)]

    return []


# ------------------------------ Data ------------------------------


@dataclass
class ChatSession:
    session_id: str
    title: str
    messages: List[ChatMessage] = field(default_factory=list)


# ------------------------------ Widgets ------------------------------


class TypingDots:
    """A tiny helper to animate a label with · · ·"""

    def __init__(self, label: ctk.CTkLabel) -> None:
        self._label = label
        self._job: Optional[str] = None
        self._step = 0
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._step = 0
        self._tick()

    def stop(self) -> None:
        self._running = False
        if self._job is not None:
            try:
                self._label.after_cancel(self._job)
            except Exception:
                pass
        self._job = None

    def _tick(self) -> None:
        if not self._running:
            return
        dots = ["·", "··", "···"]
        self._label.configure(text=dots[self._step % 3])
        self._step += 1
        self._job = self._label.after(350, self._tick)


class MessageBlock:
    """A ChatGPT-style message row (full-width band + centered content)."""

    def __init__(
        self,
        parent: ctk.CTkScrollableFrame,
        theme: Theme,
        role: str,
        text: str,
        wrap_px: int,
    ) -> None:
        self.theme = theme
        self.role = role
        self.text = text

        is_assistant = role == "assistant"
        is_user = role == "user"
        is_system = role == "system"

        row_color = theme.card_bg if is_assistant else (theme.card_bg_2 if is_user else theme.sidebar_bg)

        self.row = ctk.CTkFrame(parent, fg_color=row_color, corner_radius=0)
        self.row.pack(fill="x", padx=0, pady=0)

        # Allow callers to update wraplength by walking the widget tree.
        # (customtkinter scroll frames expose only widgets, not our wrapper objects)
        try:
            setattr(self.row, "set_wraplength", self.set_wraplength)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Centered content container (ChatGPT max-width style)
        self.inner = ctk.CTkFrame(self.row, fg_color="transparent")
        self.inner.pack(fill="x", padx=32, pady=16)

        # Avatar + content
        self.inner.grid_columnconfigure(1, weight=1)

        if is_user:
            avatar_text = "You"
            avatar_bg = theme.accent
        elif is_assistant:
            avatar_text = "AI"
            avatar_bg = theme.card_bg
        else:
            avatar_text = "SYS"
            avatar_bg = theme.sidebar_bg

        self.avatar = ctk.CTkFrame(
            self.inner,
            width=36,
            height=36,
            corner_radius=18,
            fg_color=avatar_bg,
        )
        self.avatar.grid(row=0, column=0, sticky="nw")
        self.avatar.grid_propagate(False)

        self.avatar_lbl = ctk.CTkLabel(
            self.avatar,
            text=avatar_text,
            text_color=theme.fg,
            font=("Segoe UI", 12, "bold"),
        )
        self.avatar_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self.text_lbl = ctk.CTkLabel(
            self.inner,
            text=text,
            text_color=theme.fg,
            justify="left",
            anchor="w",
            wraplength=wrap_px,
            font=("Segoe UI", 13),
        )
        self.text_lbl.grid(row=0, column=1, sticky="ew", padx=(14, 0))

        # Optional tiny timestamp (off by default look). Keep for debugging.
        self._stamp_lbl: Optional[ctk.CTkLabel] = None

        self._typing = TypingDots(self.text_lbl)

    def set_wraplength(self, px: int) -> None:
        try:
            self.text_lbl.configure(wraplength=px)
        except Exception:
            return

    def start_typing(self) -> None:
        self.text = ""
        self.text_lbl.configure(text="")
        self._typing.start()

    def stop_typing(self) -> None:
        self._typing.stop()

    def set_text(self, text: str) -> None:
        self.text = text
        self.text_lbl.configure(text=text)

    def append(self, delta: str) -> None:
        if not delta:
            return
        # First token: stop typing.
        if self._typing is not None:
            self._typing.stop()
        self.text += delta
        self.text_lbl.configure(text=self.text)


# ------------------------------ Main App ------------------------------


class ChatGPTDesktopApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.theme = CHATGPT_DARK

        # Window
        self.title("Local Chat (ChatGPT-style)")
        self.geometry("1280x820")
        self.minsize(1040, 680)

        # Icon (best effort)
        self._set_icon_best_effort("chatgpt2-icon.png")

        # Engine / backends
        self.backend = OllamaBackend()
        self.speech = SpeechRecognizer()
        self.tts = TTSEngine(TTSConfig(enabled=True, rate=165))
        self.tts.start()

        # Runtime state
        self.state_generating = False
        self.state_listening = False
        self.cancel_flag: Optional[CancelFlag] = None
        self.ui_q: "queue.Queue[tuple]" = queue.Queue()

        # Settings (Tk vars so UI binds)
        self.base_url_var = tk.StringVar(value=self.backend.base_url)
        self.model_var = tk.StringVar(value=self.backend.model)
        self.temperature_var = tk.DoubleVar(value=0.6)
        self.keep_context_var = tk.BooleanVar(value=True)
        self.speak_var = tk.BooleanVar(value=True)
        self.speak_live_var = tk.BooleanVar(value=True)
        self.tts_rate_var = tk.IntVar(value=165)

        self.system_prompt: str = (
            "You are a helpful desktop assistant. Be concise by default, but provide detail when asked."
        )

        # Chats
        self._sessions: List[ChatSession] = []
        self._active_session_id: Optional[str] = None
        self._active_assistant_block: Optional[MessageBlock] = None
        self._tts_stream_buffer: str = ""

        # Dynamic wrap
        self._wrap_px = 860

        # UI
        ctk.set_appearance_mode("Dark")
        # Don't use a stock theme; we drive colors ourselves.

        self.configure(fg_color=self.theme.app_bg)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

        self._load_settings()

        self.new_chat(select=True)

        # Poll queue + health checks
        self.after(50, self._process_ui_queue)
        self.after(900, self._periodic_healthcheck)

        # Populate models
        self.test_backend()
        self.refresh_models()

        # Resize handling
        self.bind("<Configure>", self._on_window_resize)

        # Close hook
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI construction ----------------

    def _build_sidebar(self) -> None:
        t = self.theme

        self.sidebar = ctk.CTkFrame(self, width=290, fg_color=t.sidebar_bg, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_rowconfigure(3, weight=1)

        # New chat
        self.new_chat_btn = ctk.CTkButton(
            self.sidebar,
            text="+  New chat",
            fg_color="transparent",
            border_width=1,
            border_color="#3A3B42",
            hover_color="#2A2B32",
            command=lambda: self.new_chat(select=True),
            height=38,
        )
        self.new_chat_btn.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))

        # Search (UI only; optional)
        self.search_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="Search chats…",
            fg_color="#2A2B32",
            border_color="#3A3B42",
            text_color=t.fg,
            height=34,
        )
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda _e: self._refresh_chat_list())

        # Chats list
        self.chat_list = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            scrollbar_button_color="#3A3B42",
            scrollbar_button_hover_color="#4A4B55",
        )
        self.chat_list.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 10))

        # Bottom area
        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))
        bottom.grid_columnconfigure(0, weight=1)

        self.status_lbl = ctk.CTkLabel(bottom, text="Ollama: checking…", text_color=t.muted, anchor="w")
        self.status_lbl.grid(row=0, column=0, sticky="ew")

        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.settings_btn = ctk.CTkButton(
            btn_row,
            text="Settings",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=self.open_settings,
            height=34,
        )
        self.settings_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.voice_btn = ctk.CTkButton(
            btn_row,
            text="Voice",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=self.open_voice_settings,
            height=34,
        )
        self.voice_btn.grid(row=0, column=1, sticky="ew")

    def _build_main(self) -> None:
        t = self.theme

        self.main = ctk.CTkFrame(self, fg_color=t.app_bg, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Top bar
        self.topbar = ctk.CTkFrame(self.main, fg_color=t.topbar_bg, corner_radius=0, height=56)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(0, weight=1)

        self.chat_title_lbl = ctk.CTkLabel(
            self.topbar,
            text="",
            text_color=t.fg,
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        )
        self.chat_title_lbl.grid(row=0, column=0, sticky="ew", padx=(18, 12), pady=12)

        right = ctk.CTkFrame(self.topbar, fg_color="transparent")
        right.grid(row=0, column=1, sticky="e", padx=(0, 16), pady=10)

        self.model_menu = ctk.CTkOptionMenu(
            right,
            values=[self.model_var.get()],
            variable=self.model_var,
            fg_color="#2A2B32",
            button_color="#2A2B32",
            button_hover_color="#3A3B42",
            dropdown_fg_color="#2A2B32",
            dropdown_hover_color="#3A3B42",
            text_color=t.fg,
            command=lambda _v=None: self._on_model_selected(),
            width=210,
            height=34,
        )
        self.model_menu.pack(side="left", padx=(0, 10))

        self.refresh_btn = ctk.CTkButton(
            right,
            text="Refresh",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=self.refresh_models,
            width=92,
            height=34,
        )
        self.refresh_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            right,
            text="Stop",
            fg_color=t.danger,
            hover_color="#DC2626",
            command=self.stop_everything,
            width=76,
            height=34,
        )
        self.stop_btn.pack(side="left")

        # Chat area
        self.chat_scroll = ctk.CTkScrollableFrame(
            self.main,
            fg_color=t.app_bg,
            corner_radius=0,
            scrollbar_button_color="#3A3B42",
            scrollbar_button_hover_color="#4A4B55",
        )
        self.chat_scroll.grid(row=1, column=0, sticky="nsew")

        # Input bar
        self.input_wrap = ctk.CTkFrame(self.main, fg_color=t.app_bg, corner_radius=0)
        self.input_wrap.grid(row=2, column=0, sticky="ew")
        self.input_wrap.grid_columnconfigure(0, weight=1)

        self.input_card = ctk.CTkFrame(self.input_wrap, fg_color=t.input_bg, corner_radius=12)
        self.input_card.grid(row=0, column=0, sticky="ew", padx=20, pady=(12, 18))
        self.input_card.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            self.input_card,
            height=88,
            fg_color=t.input_bg,
            text_color=t.fg,
            border_width=0,
            corner_radius=12,
            wrap="word",
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(12, 6), pady=10)

        # Keybinds: Enter sends, Shift+Enter newline
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", self._on_shift_enter)

        btns = ctk.CTkFrame(self.input_card, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="ne", padx=(6, 12), pady=10)

        self.send_btn = ctk.CTkButton(
            btns,
            text="Send",
            fg_color=t.accent,
            hover_color="#0E8A6B",
            command=self.on_send,
            width=88,
            height=34,
        )
        self.send_btn.pack(side="top", pady=(0, 8))

        self.mic_btn = ctk.CTkButton(
            btns,
            text="🎙",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=self.on_speak,
            width=88,
            height=34,
        )
        self.mic_btn.pack(side="top")

        # Footer hint (like ChatGPT)
        self.footer_lbl = ctk.CTkLabel(
            self.input_wrap,
            text="Local model via Ollama • No OpenAI credits required",
            text_color=t.muted,
            font=("Segoe UI", 11),
        )
        self.footer_lbl.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 10))

    # ---------------- Chat list ----------------

    def _refresh_chat_list(self) -> None:
        # Clear
        for w in list(self.chat_list.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass

        q = (self.search_entry.get() or "").strip().lower()

        for s in self._sessions:
            if q and q not in s.title.lower():
                continue

            selected = s.session_id == self._active_session_id
            self._add_chat_list_item(s, selected=selected)

    def _add_chat_list_item(self, session: ChatSession, selected: bool) -> None:
        t = self.theme

        fg = "#2A2B32" if selected else "transparent"
        hover = "#2A2B32" if selected else "#2A2B32"

        btn = ctk.CTkButton(
            self.chat_list,
            text=session.title,
            fg_color=fg,
            hover_color=hover,
            text_color=t.fg,
            anchor="w",
            corner_radius=10,
            command=lambda sid=session.session_id: self.select_chat(sid),
            height=36,
        )
        btn.pack(fill="x", padx=8, pady=4)

    # ---------------- Chat rendering ----------------

    def _clear_chat_view(self) -> None:
        for w in list(self.chat_scroll.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._active_assistant_block = None

    def _render_session(self, session: ChatSession) -> None:
        self._clear_chat_view()

        # Intro system message like ChatGPT
        if not session.messages:
            MessageBlock(
                self.chat_scroll,
                theme=self.theme,
                role="assistant",
                text=(
                    "Hi! I'm running locally via Ollama (free).\n\n"
                    "• Type a message below\n"
                    "• Or use 🎙 for voice input\n\n"
                    "Tip: `ollama pull llama3.1:8b` then press Refresh if models don't show."
                ),
                wrap_px=self._wrap_px,
            )

        for m in session.messages:
            MessageBlock(self.chat_scroll, theme=self.theme, role=m.role, text=m.content, wrap_px=self._wrap_px)

        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        try:
            self.chat_scroll._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ---------------- Actions ----------------

    def new_chat(self, select: bool = False) -> ChatSession:
        sid = f"chat_{len(self._sessions) + 1}_{int(_dt.datetime.now().timestamp())}"
        session = ChatSession(session_id=sid, title=f"New chat {len(self._sessions) + 1}")
        self._sessions.insert(0, session)  # newest at top like ChatGPT

        if select:
            self.select_chat(sid)
        else:
            self._refresh_chat_list()

        return session

    def get_active_session(self) -> Optional[ChatSession]:
        if not self._active_session_id:
            return None
        for s in self._sessions:
            if s.session_id == self._active_session_id:
                return s
        return None

    def select_chat(self, session_id: str) -> None:
        if self.state_generating or self.state_listening:
            # ChatGPT prevents some navigation; keep it simple.
            return

        self._active_session_id = session_id
        session = self.get_active_session()
        if session is None:
            return

        self.chat_title_lbl.configure(text=session.title)
        self._refresh_chat_list()
        self._render_session(session)
        self._sync_controls_state()

    def on_send(self) -> None:
        if self.state_generating or self.state_listening:
            return

        session = self.get_active_session()
        if session is None:
            return

        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return

        # Clear immediately
        self.input_box.delete("1.0", "end")

        # Rename new chats based on first message
        if session.title.startswith("New chat") and not session.messages:
            session.title = _safe_title_from_text(text, fallback=session.title)
            self.chat_title_lbl.configure(text=session.title)

        session.messages.append(ChatMessage(role="user", content=text))

        # Add UI block
        MessageBlock(self.chat_scroll, theme=self.theme, role="user", text=text, wrap_px=self._wrap_px)
        self._scroll_to_bottom()

        self._refresh_chat_list()
        self.start_generation(session)

    def on_speak(self) -> None:
        if self.state_generating or self.state_listening:
            return
        session = self.get_active_session()
        if session is None:
            return
        self.start_listening(session)

    def _on_enter(self, event: tk.Event) -> str:
        self.on_send()
        return "break"

    def _on_shift_enter(self, event: tk.Event) -> None:
        # allow newline
        return

    # ---------------- Settings windows ----------------

    def open_settings(self) -> None:
        SettingsWindow(self)

    def open_voice_settings(self) -> None:
        VoiceWindow(self)

    # ---------------- Backend settings ----------------

    def _on_model_selected(self) -> None:
        self.backend.model = (self.model_var.get() or "").strip() or self.backend.model

    def _set_status(self, text: str) -> None:
        self.status_lbl.configure(text=text)

    def speak_enabled(self) -> bool:
        return bool(self.speak_var.get())

    # ---------------- Orchestration (STT + streaming) ----------------

    def start_listening(self, session: ChatSession) -> None:
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
                self.ui_q.put(("speech", session.session_id, text))
            except SpeechError as e:
                self.ui_q.put(("speech_error", session.session_id, str(e)))
            except Exception as e:
                self.ui_q.put(("speech_error", session.session_id, f"Speech error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def start_generation(self, session: ChatSession) -> None:
        self.state_generating = True
        self._sync_controls_state()
        self._set_status("Generating…")

        # Ensure speech doesn't drift out of sync
        self.tts.interrupt()
        self._tts_stream_buffer = ""

        self.cancel_flag = CancelFlag()

        # Snapshot messages
        msgs = list(session.messages)
        if not self.keep_context_var.get() and msgs:
            msgs = [msgs[-1]]
        if len(msgs) > 24:
            msgs = msgs[-24:]

        temp = float(self.temperature_var.get())
        system_prompt = self.system_prompt
        backend_url = (self.base_url_var.get() or "").strip() or self.backend.base_url
        model = (self.model_var.get() or "").strip() or self.backend.model

        def worker() -> None:
            try:
                self.backend.base_url = backend_url
                self.backend.model = model

                self.ui_q.put(("assistant_begin", session.session_id))
                deltas: List[str] = []
                for d in self.backend.chat_stream(
                    msgs,
                    temperature=temp,
                    system_prompt_override=system_prompt,
                    cancel_flag=self.cancel_flag,
                ):
                    deltas.append(d)
                    self.ui_q.put(("assistant_delta", session.session_id, d))
                full = "".join(deltas)
                self.ui_q.put(("assistant_done", session.session_id, full))
            except Exception as e:
                self.ui_q.put(("error", session.session_id, str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def stop_everything(self) -> None:
        if self.cancel_flag is not None:
            self.cancel_flag.cancel()
        self.tts.interrupt()

        # Update UI state immediately.
        if self.state_listening:
            self.state_listening = False
            self._sync_controls_state()
            self._set_status("Ready")

    # ---------------- UI queue processing ----------------

    def _process_ui_queue(self) -> None:
        while True:
            try:
                item = self.ui_q.get_nowait()
            except queue.Empty:
                break

            kind = item[0]

            if kind == "models":
                models: List[str] = item[1]
                if not models:
                    self._set_status("Ollama reachable, but no models found")
                    continue

                # Update option menu list
                self.model_menu.configure(values=models)
                if self.model_var.get() not in models:
                    self.model_var.set(models[0])
                    self.backend.model = models[0]

                self._set_status(f"Ollama: {len(models)} model(s)")

            elif kind == "health":
                ok = bool(item[1])
                self._set_status("Ollama: connected" if ok else "Ollama: not reachable")

            elif kind == "speech":
                _, sid, text = item
                self.state_listening = False
                self._sync_controls_state()
                self._set_status("Ready")

                session = self._get_session_by_id(sid)
                if session is None:
                    continue

                text = str(text).strip()
                if not text:
                    continue

                # Show user message
                session.messages.append(ChatMessage(role="user", content=text))
                if sid == self._active_session_id:
                    MessageBlock(self.chat_scroll, theme=self.theme, role="user", text=text, wrap_px=self._wrap_px)
                    self._scroll_to_bottom()

                self.start_generation(session)

            elif kind == "speech_error":
                _, sid, err = item
                self.state_listening = False
                self._sync_controls_state()
                self._set_status("Ready")

                if sid == self._active_session_id:
                    MessageBlock(self.chat_scroll, theme=self.theme, role="system", text=str(err), wrap_px=self._wrap_px)
                    self._scroll_to_bottom()

            elif kind == "assistant_begin":
                _, sid = item
                if sid == self._active_session_id:
                    # Create assistant message row with typing indicator
                    self._active_assistant_block = MessageBlock(
                        self.chat_scroll,
                        theme=self.theme,
                        role="assistant",
                        text="",
                        wrap_px=self._wrap_px,
                    )
                    self._active_assistant_block.start_typing()
                    self._scroll_to_bottom()
                else:
                    self._active_assistant_block = None

            elif kind == "assistant_delta":
                _, sid, delta = item
                delta = str(delta)

                if sid == self._active_session_id and self._active_assistant_block is not None:
                    # stop typing on first delta
                    self._active_assistant_block.append(delta)
                    self._scroll_to_bottom()

                # Live TTS (keep consistent)
                if self.speak_enabled() and self.speak_live_var.get():
                    self._tts_stream_buffer += delta
                    for chunk, rest in _pop_speakable_chunks(self._tts_stream_buffer):
                        self.tts.speak(chunk)
                        self._tts_stream_buffer = rest

            elif kind == "assistant_done":
                _, sid, full = item
                full = str(full).strip()

                cancelled = bool(self.cancel_flag is not None and self.cancel_flag.cancelled)

                session = self._get_session_by_id(sid)
                if session is not None and (not cancelled) and full:
                    session.messages.append(ChatMessage(role="assistant", content=full))

                # Finalize UI for active chat
                if sid == self._active_session_id:
                    if cancelled:
                        MessageBlock(
                            self.chat_scroll,
                            theme=self.theme,
                            role="system",
                            text="Generation cancelled.",
                            wrap_px=self._wrap_px,
                        )
                    else:
                        if self._active_assistant_block is None:
                            MessageBlock(self.chat_scroll, theme=self.theme, role="assistant", text=full, wrap_px=self._wrap_px)
                        else:
                            self._active_assistant_block.stop_typing()
                            self._active_assistant_block.set_text(full)
                    self._scroll_to_bottom()

                # Speak after completion only if not speaking live
                if (not cancelled) and full and self.speak_enabled() and (not self.speak_live_var.get()):
                    self.tts.speak(full)

                # Flush tail if speaking live
                if (not cancelled) and self.speak_enabled() and self.speak_live_var.get():
                    tail = self._tts_stream_buffer.strip()
                    if tail:
                        self.tts.speak(tail)
                self._tts_stream_buffer = ""

                self.state_generating = False
                self.cancel_flag = None
                self._sync_controls_state()
                self._set_status("Ready")
                self._refresh_chat_list()

            elif kind == "error":
                _, sid, err = item
                self.state_generating = False
                self.state_listening = False
                self.cancel_flag = None
                self._sync_controls_state()
                self._set_status("Ready")

                msg = str(err)
                if sid == self._active_session_id:
                    MessageBlock(self.chat_scroll, theme=self.theme, role="system", text=msg, wrap_px=self._wrap_px)
                    self._scroll_to_bottom()

        self.after(50, self._process_ui_queue)

    # ---------------- Model list / health ----------------

    def refresh_models(self) -> None:
        def worker() -> None:
            try:
                self.backend.base_url = (self.base_url_var.get() or "").strip() or self.backend.base_url
                models = self.backend.list_models()
                self.ui_q.put(("models", models))
            except Exception as e:
                # show in active chat
                sid = self._active_session_id or ""
                self.ui_q.put(("error", sid, str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def test_backend(self) -> None:
        def worker() -> None:
            self.backend.base_url = (self.base_url_var.get() or "").strip() or self.backend.base_url
            ok = self.backend.healthcheck()
            self.ui_q.put(("health", ok))

        threading.Thread(target=worker, daemon=True).start()

    def _periodic_healthcheck(self) -> None:
        if not (self.state_generating or self.state_listening):
            ok = self.backend.healthcheck()
            self._set_status("Ollama: connected" if ok else "Ollama: not reachable")
        self.after(4000, self._periodic_healthcheck)

    # ---------------- Settings persistence ----------------

    def _load_settings(self) -> None:
        cfg = load_settings()
        if not isinstance(cfg, dict):
            return

        self.base_url_var.set(str(cfg.get("base_url", self.base_url_var.get())))
        self.model_var.set(str(cfg.get("model", self.model_var.get())))
        try:
            self.temperature_var.set(float(cfg.get("temperature", self.temperature_var.get())))
        except Exception:
            pass
        self.keep_context_var.set(bool(cfg.get("keep_context", self.keep_context_var.get())))
        self.speak_var.set(bool(cfg.get("speak", self.speak_var.get())))
        self.speak_live_var.set(bool(cfg.get("speak_live", self.speak_live_var.get())))

        try:
            rate = int(cfg.get("tts_rate", self.tts_rate_var.get()))
            self.tts_rate_var.set(rate)
            self.tts.set_rate(rate)
        except Exception:
            pass

        sys_prompt = cfg.get("system_prompt")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            self.system_prompt = sys_prompt.strip()

        # Apply
        self.backend.base_url = (self.base_url_var.get() or "").strip() or self.backend.base_url
        self.backend.model = (self.model_var.get() or "").strip() or self.backend.model
        self.tts.set_enabled(bool(self.speak_var.get()))

    def save_settings_ui(self, toast_in_chat: bool = True) -> None:
        data = {
            "base_url": (self.base_url_var.get() or "").strip(),
            "model": (self.model_var.get() or "").strip(),
            "temperature": float(self.temperature_var.get()),
            "keep_context": bool(self.keep_context_var.get()),
            "speak": bool(self.speak_var.get()),
            "speak_live": bool(self.speak_live_var.get()),
            "tts_rate": int(self.tts_rate_var.get()),
            "system_prompt": self.system_prompt,
        }
        save_settings(data)
        if toast_in_chat and self._active_session_id:
            MessageBlock(self.chat_scroll, theme=self.theme, role="system", text="Settings saved.", wrap_px=self._wrap_px)
            self._scroll_to_bottom()

    # ---------------- Chat tools ----------------

    def export_active_chat(self) -> None:
        """Export the active chat as a plain text file."""
        session = self.get_active_session()
        if session is None:
            return
        if not session.messages:
            return

        lines: List[str] = []
        for m in session.messages:
            role = "You" if m.role == "user" else ("Assistant" if m.role == "assistant" else "System")
            lines.append(f"{role}:\n{m.content.strip()}\n")
        content = "\n".join(lines).strip() + "\n"

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
            initialfile=f"{session.title}.txt",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            if self._active_session_id == session.session_id:
                MessageBlock(self.chat_scroll, theme=self.theme, role="system", text=f"Exported to: {path}", wrap_px=self._wrap_px)
                self._scroll_to_bottom()
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def clear_active_chat(self) -> None:
        if self.state_generating or self.state_listening:
            return
        session = self.get_active_session()
        if session is None:
            return
        if not messagebox.askyesno("Clear chat", "Clear all messages in this chat?"):
            return
        session.messages = []
        self._render_session(session)
        self._refresh_chat_list()

    def delete_active_chat(self) -> None:
        if self.state_generating or self.state_listening:
            return
        session = self.get_active_session()
        if session is None:
            return
        if len(self._sessions) <= 1:
            messagebox.showinfo("Chats", "You must keep at least one chat.")
            return
        if not messagebox.askyesno("Delete chat", f"Delete '{session.title}'?"):
            return
        try:
            self._sessions.remove(session)
        except ValueError:
            return
        # Select next available
        next_sid = self._sessions[0].session_id if self._sessions else None
        self._active_session_id = next_sid
        self._refresh_chat_list()
        if next_sid:
            self.select_chat(next_sid)

    # ---------------- Helpers ----------------

    def _get_session_by_id(self, sid: str) -> Optional[ChatSession]:
        for s in self._sessions:
            if s.session_id == sid:
                return s
        return None

    def _sync_controls_state(self) -> None:
        busy = self.state_generating or self.state_listening

        # Keep input editable always, but disable send/mic during actions.
        self.send_btn.configure(state="disabled" if busy else "normal")
        self.mic_btn.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")

        self.new_chat_btn.configure(state="disabled" if busy else "normal")
        self.settings_btn.configure(state="disabled" if busy else "normal")
        self.voice_btn.configure(state="disabled" if busy else "normal")

        # Disable chat selection buttons while busy
        for w in self.chat_list.winfo_children():
            try:
                w.configure(state="disabled" if busy else "normal")
            except Exception:
                pass

        # Stop button should only be visible in topbar - keep enabled state.

    def _set_icon_best_effort(self, rel_path: str) -> None:
        try:
            base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
            path = os.path.join(base, rel_path)
            if os.path.exists(path):
                # Tk iconphoto needs PhotoImage
                self.iconphoto(True, tk.PhotoImage(file=path))
        except Exception:
            return

    def _on_window_resize(self, _e: tk.Event) -> None:
        # Update wrap based on window width.
        try:
            w = int(self.winfo_width())
        except Exception:
            return
        # Reserve sidebar width + margins.
        main_w = max(520, w - 310)
        new_wrap = max(360, min(980, int(main_w * 0.72)))
        if abs(new_wrap - self._wrap_px) < 14:
            return
        self._wrap_px = new_wrap
        for child in self.chat_scroll.winfo_children():
            try:
                if hasattr(child, "set_wraplength"):
                    child.set_wraplength(self._wrap_px)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _on_close(self) -> None:
        try:
            self.save_settings_ui(toast_in_chat=False)
        except Exception:
            pass
        try:
            self.tts.stop()
        except Exception:
            pass
        self.destroy()


# ------------------------------ Settings Windows ------------------------------


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, app: ChatGPTDesktopApp) -> None:
        super().__init__(app)
        self.app = app
        t = app.theme

        self.title("Settings")
        self.geometry("740x520")
        self.minsize(640, 480)
        self.configure(fg_color=t.app_bg)
        self.transient(app)
        self.grab_set()

        tabs = ctk.CTkTabview(self, fg_color=t.app_bg, segmented_button_fg_color=t.topbar_bg)
        tabs.pack(fill="both", expand=True, padx=16, pady=16)

        tab_model = tabs.add("Model")
        tab_gen = tabs.add("Generation")
        tab_prompt = tabs.add("System prompt")
        tab_tools = tabs.add("Tools")
        tab_about = tabs.add("About")

        # Model tab
        model_card = ctk.CTkFrame(tab_model, fg_color=t.topbar_bg, corner_radius=12)
        model_card.pack(fill="x", padx=10, pady=10)
        model_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(model_card, text="Ollama base URL", text_color=t.muted).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 6))
        url = ctk.CTkEntry(model_card, textvariable=app.base_url_var, fg_color="#2A2B32", border_color="#3A3B42")
        url.grid(row=0, column=1, sticky="ew", padx=14, pady=(14, 6))

        ctk.CTkLabel(model_card, text="Model", text_color=t.muted).grid(row=1, column=0, sticky="w", padx=14, pady=6)
        model = ctk.CTkOptionMenu(
            model_card,
            values=[app.model_var.get()],
            variable=app.model_var,
            fg_color="#2A2B32",
            button_color="#2A2B32",
            button_hover_color="#3A3B42",
            dropdown_fg_color="#2A2B32",
            dropdown_hover_color="#3A3B42",
            text_color=t.fg,
            command=lambda _v=None: app._on_model_selected(),
        )
        model.grid(row=1, column=1, sticky="ew", padx=14, pady=6)

        btn_row = ctk.CTkFrame(model_card, fg_color="transparent")
        btn_row.grid(row=2, column=1, sticky="e", padx=14, pady=(6, 14))
        ctk.CTkButton(
            btn_row,
            text="Test",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=app.test_backend,
            width=90,
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btn_row,
            text="Refresh models",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=app.refresh_models,
            width=140,
        ).pack(side="left")

        # Keep this option menu synced with main
        def sync_models() -> None:
            try:
                vals = list(app.model_menu.cget("values"))
                if vals:
                    model.configure(values=vals)
            except Exception:
                pass
            self.after(800, sync_models)

        sync_models()

        # Generation tab
        gen_card = ctk.CTkFrame(tab_gen, fg_color=t.topbar_bg, corner_radius=12)
        gen_card.pack(fill="x", padx=10, pady=10)
        gen_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(gen_card, text="Temperature", text_color=t.muted).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 6))
        temp = ctk.CTkSlider(gen_card, from_=0.0, to=1.2, number_of_steps=60, variable=app.temperature_var)
        temp.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        ctx = ctk.CTkCheckBox(
            gen_card,
            text="Keep conversation context",
            variable=app.keep_context_var,
            text_color=t.fg,
            fg_color=t.accent,
        )
        ctx.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 14))

        # System prompt tab
        prompt_card = ctk.CTkFrame(tab_prompt, fg_color=t.topbar_bg, corner_radius=12)
        prompt_card.pack(fill="both", expand=True, padx=10, pady=10)
        prompt_card.grid_rowconfigure(1, weight=1)
        prompt_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            prompt_card,
            text="System prompt (sent as a system message to the model)",
            text_color=t.muted,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))

        self.prompt_box = ctk.CTkTextbox(
            prompt_card,
            fg_color="#2A2B32",
            text_color=t.fg,
            border_color="#3A3B42",
            border_width=1,
            wrap="word",
        )
        self.prompt_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.prompt_box.insert("1.0", app.system_prompt)

        pr_btns = ctk.CTkFrame(prompt_card, fg_color="transparent")
        pr_btns.grid(row=2, column=0, sticky="e", padx=14, pady=(0, 14))

        def apply_prompt() -> None:
            val = self.prompt_box.get("1.0", "end").strip()
            if not val:
                return
            app.system_prompt = val
            app.save_settings_ui(toast_in_chat=True)
            self.destroy()

        ctk.CTkButton(
            pr_btns,
            text="Apply",
            fg_color=t.accent,
            hover_color="#0E8A6B",
            command=apply_prompt,
            width=90,
        ).pack(side="right")

        # About tab
        about_card = ctk.CTkFrame(tab_about, fg_color=t.topbar_bg, corner_radius=12)
        about_card.pack(fill="both", expand=True, padx=10, pady=10)

        txt = (
            "This app uses a free, local AI agent via Ollama.\n\n"
            "• Install Ollama and run:  ollama pull llama3.1:8b\n"
            "• Change model storage drive with OLLAMA_MODELS (Windows/mac/Linux)\n"
            "• No OpenAI credits needed\n\n"
            "UI goal: match ChatGPT layout (sidebar + chat bands) while staying in Tk."
        )
        ctk.CTkLabel(about_card, text=txt, text_color=t.fg, justify="left", anchor="nw").pack(
            fill="both", expand=True, padx=14, pady=14
        )

        # Tools tab
        tools_card = ctk.CTkFrame(tab_tools, fg_color=t.topbar_bg, corner_radius=12)
        tools_card.pack(fill="x", padx=10, pady=10)
        tools_card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            tools_card,
            text="Export current chat…",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=app.export_active_chat,
            height=38,
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))

        ctk.CTkButton(
            tools_card,
            text="Clear current chat",
            fg_color="#2A2B32",
            hover_color="#3A3B42",
            command=app.clear_active_chat,
            height=38,
        ).grid(row=0, column=1, sticky="ew", padx=14, pady=(14, 10))

        ctk.CTkButton(
            tools_card,
            text="Delete current chat",
            fg_color=app.theme.danger,
            hover_color="#DC2626",
            command=app.delete_active_chat,
            height=38,
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

        ctk.CTkButton(
            tools_card,
            text="Save settings",
            fg_color=app.theme.accent,
            hover_color="#0E8A6B",
            command=lambda: app.save_settings_ui(toast_in_chat=True),
            height=38,
        ).grid(row=1, column=1, sticky="ew", padx=14, pady=(0, 14))


class VoiceWindow(ctk.CTkToplevel):
    def __init__(self, app: ChatGPTDesktopApp) -> None:
        super().__init__(app)
        self.app = app
        t = app.theme

        self.title("Voice")
        self.geometry("620x420")
        self.minsize(560, 380)
        self.configure(fg_color=t.app_bg)
        self.transient(app)
        self.grab_set()

        card = ctk.CTkFrame(self, fg_color=t.topbar_bg, corner_radius=12)
        card.pack(fill="both", expand=True, padx=16, pady=16)
        card.grid_columnconfigure(0, weight=1)

        speak = ctk.CTkCheckBox(
            card,
            text="Speak responses (offline TTS)",
            variable=app.speak_var,
            text_color=t.fg,
            fg_color=t.accent,
            command=self._on_toggle,
        )
        speak.grid(row=0, column=0, sticky="w", padx=14, pady=(14, 8))

        live = ctk.CTkCheckBox(
            card,
            text="Speak while generating (keeps speech consistent)",
            variable=app.speak_live_var,
            text_color=t.fg,
            fg_color=t.accent,
        )
        live.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 14))

        ctk.CTkLabel(card, text="Speech rate", text_color=t.muted).grid(row=2, column=0, sticky="w", padx=14)

        rate = ctk.CTkSlider(
            card,
            from_=110,
            to=220,
            number_of_steps=110,
            variable=app.tts_rate_var,
            command=self._on_rate,
        )
        rate.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))

        hint = (
            "Mic button uses SpeechRecognition's default Google recognizer (free, online).\n"
            "If you want fully offline STT, swap to Vosk."
        )
        ctk.CTkLabel(card, text=hint, text_color=t.muted, justify="left", anchor="w").grid(
            row=4, column=0, sticky="w", padx=14, pady=(0, 14)
        )

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=5, column=0, sticky="e", padx=14, pady=(0, 14))

        def save_close() -> None:
            app.save_settings_ui(toast_in_chat=True)
            self.destroy()

        ctk.CTkButton(
            btns,
            text="Save",
            fg_color=t.accent,
            hover_color="#0E8A6B",
            command=save_close,
            width=90,
        ).pack(side="right")

    def _on_toggle(self) -> None:
        self.app.tts.set_enabled(bool(self.app.speak_var.get()))
        if not self.app.speak_var.get():
            self.app.tts.interrupt()

    def _on_rate(self, _v: float) -> None:
        try:
            r = int(self.app.tts_rate_var.get())
        except Exception:
            return
        self.app.tts.set_rate(r)


def main() -> None:
    app = ChatGPTDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
