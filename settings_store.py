from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_SETTINGS_PATH = Path.home() / ".voice_assistant_settings.json"


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def save_settings(data: Dict[str, Any], path: Path = DEFAULT_SETTINGS_PATH) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Don't crash app if filesystem is read-only.
        return
