from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSConfig:
    enabled: bool = True
    rate: int = 165
    voice_id: Optional[str] = None


class TTSEngine:
    """Background text-to-speech using pyttsx3.

    pyttsx3 is fully offline (free) but voice quality depends on the OS voices.
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False
        self._stop_event = threading.Event()

        # Best-effort interruption support. pyttsx3 isn't perfectly
        # thread-safe across all platforms, so this is defensive.
        self._engine = None
        self._engine_lock = threading.Lock()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def clear_queue(self) -> None:
        """Drop any queued utterances not yet spoken."""
        try:
            while True:
                item = self._q.get_nowait()
                if item is None:
                    # Preserve shutdown marker if present.
                    self._q.put_nowait(None)
                    break
        except queue.Empty:
            return

    def interrupt(self) -> None:
        """Best-effort stop of current speech + clear pending queue."""
        self.clear_queue()
        with self._engine_lock:
            eng = self._engine
        if eng is not None:
            try:
                eng.stop()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        if not self.config.enabled:
            return
        if not text.strip():
            return
        if not self._started:
            self.start()
        self._q.put(text)

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = bool(enabled)
        if not self.config.enabled:
            # Avoid stale speech later when user re-enables.
            self.clear_queue()

    def set_rate(self, rate: int) -> None:
        try:
            rate = int(rate)
        except Exception:
            return
        self.config.rate = max(80, min(260, rate))

    def _worker(self) -> None:
        try:
            import pyttsx3
        except Exception:
            # If pyttsx3 isn't installed, just drain the queue.
            while not self._stop_event.is_set():
                item = self._q.get()
                if item is None:
                    return
            return

        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.rate)
        if self.config.voice_id:
            try:
                engine.setProperty("voice", self.config.voice_id)
            except Exception:
                pass

        with self._engine_lock:
            self._engine = engine

        while not self._stop_event.is_set():
            text = self._q.get()
            if text is None:
                return

            # Apply latest config each utterance.
            try:
                engine.setProperty("rate", self.config.rate)
            except Exception:
                pass

            if not self.config.enabled:
                continue

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # Don't crash the app due to TTS hiccups.
                continue

        with self._engine_lock:
            self._engine = None
