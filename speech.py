from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional


class SpeechError(RuntimeError):
    pass


@dataclass
class SpeechConfig:
    language: str = "en-US"
    timeout_s: int = 6
    phrase_time_limit_s: int = 10
    ambient_adjust_s: float = 0.6


class SpeechRecognizer:
    """Speech-to-text using SpeechRecognition.

    Notes:
      - Default backend is Google's free web speech recognizer.
      - For fully offline STT you can add Vosk support later.
    """

    def __init__(self, config: Optional[SpeechConfig] = None) -> None:
        self.config = config or SpeechConfig()

    def listen_once(self) -> str:
        try:
            import speech_recognition as sr
        except Exception as e:
            raise SpeechError(
                "speech_recognition is not installed. Install requirements.txt. "
                f"Details: {e}"
            )

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            with contextlib.suppress(Exception):
                recognizer.adjust_for_ambient_noise(
                    source, duration=float(self.config.ambient_adjust_s)
                )
            try:
                audio_data = recognizer.listen(
                    source,
                    timeout=float(self.config.timeout_s),
                    phrase_time_limit=float(self.config.phrase_time_limit_s),
                )
            except sr.WaitTimeoutError as e:
                raise SpeechError("Timed out waiting for speech.") from e

        try:
            return recognizer.recognize_google(
                audio_data, language=str(self.config.language)
            )
        except sr.UnknownValueError as e:
            raise SpeechError("Sorry — I couldn't understand that.") from e
        except sr.RequestError as e:
            raise SpeechError(
                "Speech recognition request failed (network issue or blocked)."
            ) from e
