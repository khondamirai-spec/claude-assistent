"""Small JSON-file stores kept under ``data/`` (a Railway Volume in production)."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]


def ensure_private_dir(path: PathLike) -> Path:
    """Create ``path`` if needed and make it accessible to the owner only (0700)."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


class JsonFile:
    """A JSON file that is written atomically and read forgivingly."""

    def __init__(self, path: PathLike, default: Union[Callable[[], Any], Any]) -> None:
        self.path = Path(path)
        self._default = default

    def default(self) -> Any:
        return self._default() if callable(self._default) else copy.deepcopy(self._default)

    def load(self) -> Any:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return self.default()

    def save(self, data: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


class HistoryStore:
    """Recent messages per business connection *and* chat."""

    def __init__(self, path: PathLike, limit: int) -> None:
        self._file = JsonFile(path, dict)
        self.limit = limit
        data = self._file.load()
        self._data: Dict[str, List[Dict[str, Any]]] = data if isinstance(data, dict) else {}

    @staticmethod
    def key(connection_id: str, chat_id: int) -> str:
        return f"{connection_id}:{chat_id}"

    def get(self, connection_id: str, chat_id: int) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._data.get(self.key(connection_id, chat_id), [])]

    def add(
        self,
        connection_id: str,
        chat_id: int,
        *,
        role: str,
        name: str,
        text: str,
        lang: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        items = self._data.setdefault(self.key(connection_id, chat_id), [])
        items.append(
            {
                "role": role,
                "name": name,
                "text": text,
                "lang": lang,
                "ts": time.time() if ts is None else ts,
            }
        )
        del items[: -self.limit]
        self._file.save(self._data)

    def last_language(self, connection_id: str, chat_id: int) -> Optional[str]:
        """The last detected language in the chat, preferring the other person's messages."""
        items = self._data.get(self.key(connection_id, chat_id), [])
        for wanted in ("other", None):
            for item in reversed(items):
                if item.get("lang") and (wanted is None or item.get("role") == wanted):
                    return item["lang"]
        return None

    def chat_count(self) -> int:
        return len(self._data)


class SeenMessages:
    """Remembers recently handled message keys, across restarts."""

    def __init__(self, path: PathLike, capacity: int = 5000) -> None:
        self._file = JsonFile(path, list)
        self.capacity = capacity
        loaded = self._file.load()
        keys = loaded if isinstance(loaded, list) else []
        self._keys: "OrderedDict[str, None]" = OrderedDict((str(k), None) for k in keys[-capacity:])

    def first_time(self, key: str) -> bool:
        if key in self._keys:
            return False
        self._keys[key] = None
        while len(self._keys) > self.capacity:
            self._keys.popitem(last=False)
        self._file.save(list(self._keys))
        return True


class RuntimeState:
    """Owner-controlled state that survives restarts: pause flag and usage-limit pause."""

    def __init__(self, path: PathLike) -> None:
        self._file = JsonFile(path, dict)
        data = self._file.load()
        if not isinstance(data, dict):
            data = {}
        self.paused = bool(data.get("paused", False))
        limit_until = data.get("limit_until")
        self.limit_until: Optional[float] = (
            float(limit_until) if isinstance(limit_until, (int, float)) else None
        )

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self._save()

    def set_limit_until(self, until: Optional[float]) -> None:
        self.limit_until = until
        self._save()

    def _save(self) -> None:
        self._file.save({"paused": self.paused, "limit_until": self.limit_until})


def append_private_line(path: PathLike, line: str) -> None:
    """Append one line to a file that only the owner can read (0600)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (line.rstrip("\n") + "\n").encode("utf-8"))
    finally:
        os.close(fd)
