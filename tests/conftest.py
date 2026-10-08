"""Offline test harness: a real aiogram Dispatcher fed with fake updates.

Nothing here talks to the network: Telegram calls go to RecordingSession and
Claude calls go to FakeGenerator.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetBusinessConnection, SendMessage, TelegramMethod
from aiogram.types import BusinessConnection, Message, Update

from assistant.config import Settings, load_settings
from assistant.handlers import build_router
from assistant.service import Assistant, Outcome, Preview

BOT_ID = 4242
BOT_TOKEN = f"{BOT_ID}:TEST-not-a-real-token"
OWNER_ID = 1001
CONTACT_ID = 2002
STRANGER_ID = 3003
START_TS = 1_760_000_000.0

PROFILE_TEXT = """\
<!-- template comment that must not reach Claude -->
# Men haqimda
- Ism: Eldor
- Shahar: Toshkent
Secret-ish detail: plays football on Fridays
"""


def connection_payload(
    connection_id: str = "conn-1",
    *,
    user_id: int = OWNER_ID,
    enabled: bool = True,
    can_reply: bool = True,
) -> Dict[str, Any]:
    return {
        "id": connection_id,
        "user": {"id": user_id, "is_bot": False, "first_name": "Owner"},
        "user_chat_id": user_id,
        "date": 1_700_000_000,
        "is_enabled": enabled,
        # Both the current (rights) and the legacy (can_reply) shape.
        "rights": {"can_reply": can_reply},
        "can_reply": can_reply,
    }


class RecordingSession(BaseSession):
    """Records every Bot API call and answers it without any network."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: List[TelegramMethod] = []
        self.connections: Dict[str, Dict[str, Any]] = {}
        self._message_ids = itertools.count(50_000)

    async def make_request(self, bot: Bot, method: TelegramMethod, timeout: Optional[int] = None) -> Any:
        self.requests.append(method)
        if isinstance(method, GetBusinessConnection):
            payload = self.connections.get(method.business_connection_id)
            if payload is None:
                raise TelegramBadRequest(method=method, message="Bad Request: business connection not found")
            return BusinessConnection.model_validate(payload, context={"bot": bot})
        if isinstance(method, SendMessage):
            return Message.model_validate(
                {
                    "message_id": next(self._message_ids),
                    "date": int(START_TS),
                    "chat": {"id": method.chat_id, "type": "private"},
                    "from": {"id": BOT_ID, "is_bot": True, "first_name": "Assistant"},
                    "text": method.text,
                    "business_connection_id": method.business_connection_id,
                },
                context={"bot": bot},
            )
        return True

    async def close(self) -> None:
        pass

    def stream_content(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must not download anything")


class FakeGenerator:
    """Stands in for ClaudeCLI: returns (or raises) queued items and records calls."""

    def __init__(self, *responses: Any, during: Optional[Callable[[], Any]] = None) -> None:
        self.responses: List[Any] = list(responses)
        self.calls: List[Dict[str, str]] = []
        self.during = during
        self.default = "Salom! Hozir biroz bandman, keyinroq yozaman."

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        await asyncio.sleep(0)  # let parallel updates interleave
        if self.during is not None:
            hook, self.during = self.during, None
            result = hook()
            if inspect.isawaitable(result):
                await result
        item = self.responses.pop(0) if self.responses else self.default
        if isinstance(item, BaseException):
            raise item
        return item


class Clock:
    def __init__(self, start: float = START_TS) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Harness:
    def __init__(self, tmp_path: Path, *, auto_send: bool = False, **overrides: str) -> None:
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data"
        self.profile_path = tmp_path / "profile.md"
        if not self.profile_path.exists():
            self.profile_path.write_text(PROFILE_TEXT, encoding="utf-8")
        env_file = tmp_path / ".env"
        if not env_file.exists():
            env_file.write_text("", encoding="utf-8")
            env_file.chmod(0o600)

        values = {
            "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
            "OWNER_ID": str(OWNER_ID),
            "AUTO_SEND": "true" if auto_send else "false",
            "CLAUDE_BIN": str(tmp_path / "claude-must-not-run"),
            "DATA_DIR": str(self.data_dir),
            "PROFILE_PATH": str(self.profile_path),
        }
        values.update(overrides)
        self.settings: Settings = load_settings(env_file, overrides=values)

        self.session = RecordingSession()
        self.bot = Bot(BOT_TOKEN, session=self.session)
        self.generator = FakeGenerator()
        self.clock = Clock()
        self.previews: List[Preview] = []
        self._results: Dict[int, Outcome] = {}
        self._update_ids = itertools.count(1)
        self._message_ids = itertools.count(1)
        self._build()

    def _build(self) -> None:
        self.assistant = Assistant(
            self.settings, self.generator, clock=self.clock, preview_sink=self.previews.append
        )
        self.dp = Dispatcher()
        self.dp.include_router(build_router(self.assistant))

        # Remember each message's own outcome, so parallel updates can be told apart.
        handle = self.assistant.on_business_message

        async def recording(bot: Bot, message: Message) -> Outcome:
            outcome = await handle(bot, message)
            self._results[id(message)] = outcome
            return outcome

        self.assistant.on_business_message = recording  # type: ignore[method-assign]

    def restart(self) -> None:
        """A fresh Assistant and Dispatcher over the same data directory."""
        self._build()

    # ----------------------------------------------------------------- feeding

    async def feed(self, payload: Dict[str, Any]) -> Update:
        update = Update.model_validate({"update_id": next(self._update_ids), **payload}, context={"bot": self.bot})
        await self.dp.feed_update(self.bot, update)
        return update

    async def connect(self, connection_id: str = "conn-1", **kwargs: Any) -> None:
        """Telegram reports a (new or changed) business connection."""
        self.set_connection(connection_id, **kwargs)
        await self.feed({"business_connection": connection_payload(connection_id, **kwargs)})

    def set_connection(self, connection_id: str = "conn-1", **kwargs: Any) -> None:
        """Change what getBusinessConnection returns, without sending an update."""
        self.session.connections[connection_id] = connection_payload(connection_id, **kwargs)

    async def business_message(
        self,
        text: Optional[str] = "Salom, qalesan?",
        *,
        from_id: int = CONTACT_ID,
        chat_id: int = CONTACT_ID,
        message_id: Optional[int] = None,
        connection_id: Optional[str] = "conn-1",
        is_bot: bool = False,
        via_business_bot: bool = False,
        first_name: str = "Ali",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Outcome:
        message: Dict[str, Any] = {
            "message_id": message_id if message_id is not None else next(self._message_ids),
            "date": int(self.clock()),
            "chat": {"id": chat_id, "type": "private", "first_name": first_name},
            "from": {"id": from_id, "is_bot": is_bot, "first_name": first_name},
        }
        if connection_id is not None:
            message["business_connection_id"] = connection_id
        if text is not None:
            message["text"] = text
        if via_business_bot:
            message["sender_business_bot"] = {
                "id": BOT_ID,
                "is_bot": True,
                "first_name": "Assistant",
                "username": "assistant_bot",
            }
        if extra:
            message.update(extra)
        update = await self.feed({"business_message": message})
        assert id(update.business_message) in self._results, "the business message was not handled"
        return self._results.pop(id(update.business_message))

    async def direct_message(self, text: str, *, from_id: int = OWNER_ID) -> None:
        """A private message sent to the bot itself (commands get a bot_command entity)."""
        message: Dict[str, Any] = {
            "message_id": next(self._message_ids),
            "date": int(self.clock()),
            "chat": {"id": from_id, "type": "private", "first_name": "Someone"},
            "from": {"id": from_id, "is_bot": False, "first_name": "Someone"},
            "text": text,
        }
        if text.startswith("/"):
            message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
        await self.feed({"message": message})

    # ---------------------------------------------------------------- queries

    def requests_of(self, kind: Type[TelegramMethod]) -> List[Any]:
        return [r for r in self.session.requests if isinstance(r, kind)]

    def sent(self, chat_id: Optional[int] = None) -> List[SendMessage]:
        return [m for m in self.requests_of(SendMessage) if chat_id is None or m.chat_id == chat_id]

    def history(self, chat_id: int = CONTACT_ID, connection_id: str = "conn-1") -> List[Dict[str, Any]]:
        return self.assistant.history.get(connection_id, chat_id)


@pytest.fixture
def make_harness(tmp_path: Path) -> Callable[..., Harness]:
    def factory(**kwargs: Any) -> Harness:
        return Harness(tmp_path, **kwargs)

    return factory
