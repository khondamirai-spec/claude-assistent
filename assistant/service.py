"""Business message handling: decides whether to reply, drafts and (optionally) sends."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BusinessConnection, Message

from .claude_cli import ClaudeCLI, ClaudeError, ClaudeLimitError
from .config import Settings
from .language import detect_language
from .prompts import SKIP_TOKEN, build_system_prompt, build_user_prompt
from .store import (
    HistoryStore,
    RuntimeState,
    SeenMessages,
    append_private_line,
    ensure_private_dir,
)

log = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
OWNER_ALERT_INTERVAL = 600  # seconds between "Claude failed" messages to the owner

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_EMPTY_FIELD_RE = re.compile(r"^-\s*[^:]+:\s*$")


class Outcome(str, Enum):
    previewed = "previewed"
    sent = "sent"
    duplicate = "duplicate"
    ignored_bot = "ignored_bot"
    no_connection = "no_connection"
    foreign_connection = "foreign_connection"
    connection_disabled = "connection_disabled"
    unsupported = "unsupported"
    owner_message = "owner_message"
    owner_active = "owner_active"
    paused = "paused"
    limited = "limited"
    skipped = "skipped"
    permission_denied = "permission_denied"
    error = "error"


@dataclass
class Preview:
    connection_id: str
    chat_id: int
    chat_name: str
    language: Optional[str]
    incoming: str
    reply: str
    note: Optional[str] = None


def print_preview(preview: Preview) -> None:
    rule = "-" * 60
    lines = [
        "",
        f"{rule}",
        "PREVIEW, not sent",
        f"chat:     {preview.chat_name} (chat {preview.chat_id}, connection {preview.connection_id})",
        f"language: {preview.language or 'unknown'}",
        f"incoming: {preview.incoming}",
        f"reply:    {preview.reply}",
    ]
    if preview.note:
        lines.append(f"note:     {preview.note}")
    lines.append(rule)
    log.info("\n".join(lines))


def describe(message: Message) -> Optional[str]:
    """Text for a message, a short tag for media, or None for service messages."""
    if message.text:
        return message.text
    caption = (message.caption or "").strip()

    def tag(label: str) -> str:
        return f"{label} {caption}" if caption else label

    if message.photo:
        return tag("[photo]")
    if message.animation:
        return tag("[GIF]")
    if message.video:
        return tag("[video]")
    if message.video_note:
        return "[video message]"
    if message.voice:
        return tag("[voice message]")
    if message.audio:
        return tag("[audio]")
    if message.sticker:
        return f"[sticker] {message.sticker.emoji or ''}".strip()
    if message.document:
        name = message.document.file_name
        return tag(f"[file: {name}]" if name else "[file]")
    if message.contact:
        return "[contact]"
    if message.venue:
        return f"[place] {message.venue.title}"
    if message.location:
        return "[location]"
    if message.poll:
        return f"[poll] {message.poll.question}"
    if message.dice:
        return f"[dice] {message.dice.emoji}"
    if getattr(message, "story", None):
        return "[story]"
    if getattr(message, "paid_media", None):
        return tag("[paid media]")
    return None


def can_reply(connection: BusinessConnection) -> bool:
    rights = getattr(connection, "rights", None)
    if rights is not None:
        if isinstance(rights, dict):
            return bool(rights.get("can_reply"))
        return bool(getattr(rights, "can_reply", False))
    return bool(getattr(connection, "can_reply", False))


def read_profile(path: Union[str, Path]) -> str:
    """The profile without <!-- comments -->. Read on every reply, so edits apply at once."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    return _COMMENT_RE.sub("", text).strip()


def profile_is_empty(profile: str) -> bool:
    """True when the profile has only headings and empty "- Field:" lines."""
    for line in profile.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not _EMPTY_FIELD_RE.match(line):
            return False
    return True


def split_message(text: str, size: int = MAX_MESSAGE_LENGTH) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")


class Assistant:
    def __init__(
        self,
        settings: Settings,
        generator: Any,
        *,
        clock: Callable[[], float] = time.time,
        preview_sink: Callable[[Preview], None] = print_preview,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.clock = clock
        self.preview_sink = preview_sink
        self.bot_username: Optional[str] = None

        data_dir = ensure_private_dir(settings.data_dir)
        self.history = HistoryStore(data_dir / "history.json", settings.history_limit)
        self.seen = SeenMessages(data_dir / "seen.json")
        self.state = RuntimeState(data_dir / "state.json")
        self.previews_path = data_dir / "previews.jsonl"

        self.connections: Dict[str, BusinessConnection] = {}
        self.outcomes: Deque[Outcome] = deque(maxlen=100)
        self._takeover_until: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_owner_alert: Optional[float] = None

    # ------------------------------------------------------------------ updates

    def on_business_connection(self, connection: BusinessConnection) -> None:
        self.connections[connection.id] = connection
        if connection.user.id != self.settings.owner_id:
            log.warning(
                "Business connection %s belongs to another account (user=%s); it will be ignored.",
                connection.id,
                connection.user.id,
            )
            return
        log.info(
            "Business connection %s: user=%s enabled=%s can_reply=%s",
            connection.id,
            connection.user.id,
            connection.is_enabled,
            can_reply(connection),
        )

    async def on_business_message(self, bot: Bot, message: Message) -> Outcome:
        outcome = await self._handle(bot, message)
        self.outcomes.append(outcome)
        log.info("Business message %s in chat %s: %s", message.message_id, message.chat.id, outcome.value)
        return outcome

    async def _handle(self, bot: Bot, message: Message) -> Outcome:
        conn_id = message.business_connection_id
        if not conn_id:
            return Outcome.no_connection
        chat_id = message.chat.id

        # Marked before any await, so parallel deliveries of the same update are caught.
        if not self.seen.first_time(f"{conn_id}:{chat_id}:{message.message_id}"):
            return Outcome.duplicate

        sender = message.from_user
        if message.sender_business_bot or (sender is not None and sender.is_bot):
            return Outcome.ignored_bot

        connection = self.connections.get(conn_id)
        if connection is None:
            try:
                connection = await bot.get_business_connection(conn_id)
            except TelegramAPIError as exc:
                log.warning("Could not look up business connection %s: %s", conn_id, exc)
                return Outcome.no_connection
            self.connections[conn_id] = connection
        if connection.user.id != self.settings.owner_id:
            return Outcome.foreign_connection
        if not connection.is_enabled:
            return Outcome.connection_disabled

        text = describe(message)
        if text is None:
            return Outcome.unsupported

        key = HistoryStore.key(conn_id, chat_id)
        now = self.clock()
        if sender is not None and sender.id == self.settings.owner_id:
            self.history.add(conn_id, chat_id, role="owner", name="You", text=text, lang=detect_language(text), ts=now)
            if self.settings.owner_takeover_minutes > 0:
                self._takeover_until[key] = now + self.settings.owner_takeover_minutes * 60
            return Outcome.owner_message

        language = detect_language(text) or self.history.last_language(conn_id, chat_id)
        chat_name = (sender.full_name if sender else None) or message.chat.full_name or str(chat_id)
        self.history.add(conn_id, chat_id, role="other", name=chat_name, text=text, lang=language, ts=now)

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            blocked = self._blocked(key)
            if blocked is not None:
                return blocked

            reply = await self._generate(bot, conn_id, chat_id, language)
            if isinstance(reply, Outcome):
                return reply
            if not reply or SKIP_TOKEN in reply:
                log.info("Claude chose not to reply in chat %s", chat_id)
                return Outcome.skipped

            preview = Preview(conn_id, chat_id, chat_name, language, text, reply)
            if not self.settings.auto_send:
                self._show_preview(preview)
                return Outcome.previewed
            return await self._send(bot, preview)

    # --------------------------------------------------------------- internals

    def _blocked(self, key: str) -> Optional[Outcome]:
        now = self.clock()
        if self.state.paused:
            return Outcome.paused
        if self.state.limit_until is not None and self.state.limit_until > now:
            return Outcome.limited
        if self._takeover_until.get(key, 0) > now:
            return Outcome.owner_active
        return None

    async def _generate(self, bot: Bot, conn_id: str, chat_id: int, language: Optional[str]) -> Union[str, Outcome]:
        profile = read_profile(self.settings.profile_path)
        system_prompt = build_system_prompt(profile, datetime.fromtimestamp(self.clock()).astimezone())
        user_prompt = build_user_prompt(self.history.get(conn_id, chat_id), language)
        try:
            reply = await self.generator.generate(system_prompt, user_prompt)
        except ClaudeLimitError as exc:
            now = self.clock()
            until = exc.resets_at if exc.resets_at and exc.resets_at > now else now + self.settings.limit_cooldown_minutes * 60
            self.state.set_limit_until(until)
            log.warning("%s. Not calling Claude again until %s. CLI said: %s", exc, _format_time(until), exc.detail)
            if self.settings.auto_send:
                await self._notify_owner(
                    bot,
                    f"{exc}. I will not reply to anyone until {_format_time(until)}. "
                    "Send /resume to try earlier.",
                )
            return Outcome.limited
        except ClaudeError as exc:
            log.error("Claude failed for chat %s: %s", chat_id, exc)
            if self.settings.auto_send:
                now = self.clock()
                if self._last_owner_alert is None or now - self._last_owner_alert >= OWNER_ALERT_INTERVAL:
                    self._last_owner_alert = now
                    await self._notify_owner(bot, f"Could not draft a reply: {exc}")
            return Outcome.error
        return (reply or "").strip()

    def _show_preview(self, preview: Preview) -> None:
        self.preview_sink(preview)
        record = {"ts": self.clock(), **asdict(preview)}
        try:
            append_private_line(self.previews_path, json.dumps(record, ensure_ascii=False))
        except OSError as exc:
            log.error("Could not write %s: %s", self.previews_path, exc)

    async def _send(self, bot: Bot, preview: Preview) -> Outcome:
        conn_id, chat_id = preview.connection_id, preview.chat_id

        # The owner may have written (or paused) while Claude was thinking.
        blocked = self._blocked(HistoryStore.key(conn_id, chat_id))
        if blocked is not None:
            reasons = {
                Outcome.owner_active: "not sent: you wrote in this chat while the reply was being drafted",
                Outcome.paused: "not sent: the assistant was paused while the reply was being drafted",
                Outcome.limited: "not sent: Claude usage limit",
            }
            preview.note = reasons[blocked]
            self._show_preview(preview)
            return blocked

        # Re-check the connection with Telegram itself (not the cache) right before sending.
        reason = None
        try:
            connection = await bot.get_business_connection(conn_id)
        except TelegramAPIError as exc:
            reason = f"not sent: could not re-check the business connection ({exc})"
        else:
            self.connections[conn_id] = connection
            if connection.user.id != self.settings.owner_id:
                reason = "not sent: the business connection belongs to another account"
            elif not connection.is_enabled:
                reason = "not sent: the business connection is disabled"
            elif not can_reply(connection):
                reason = "not sent: the bot has no permission to reply in Telegram Business settings"
        if reason:
            log.warning("Chat %s: %s", chat_id, reason)
            preview.note = reason
            self._show_preview(preview)
            return Outcome.permission_denied

        try:
            for chunk in split_message(preview.reply):
                await bot.send_message(chat_id, chunk, business_connection_id=conn_id)
        except TelegramAPIError as exc:
            log.error("Could not send the reply to chat %s: %s", chat_id, exc)
            return Outcome.error

        self.history.add(conn_id, chat_id, role="owner", name="You", text=preview.reply, lang=preview.language, ts=self.clock())
        log.info("Sent a reply to %s (chat %s)", preview.chat_name, chat_id)
        return Outcome.sent

    async def _notify_owner(self, bot: Bot, text: str) -> None:
        try:
            await bot.send_message(self.settings.owner_id, text)
        except TelegramAPIError as exc:
            log.warning("Could not notify the owner: %s", exc)

    # ----------------------------------------------------------- owner control

    def pause(self) -> None:
        self.state.set_paused(True)
        log.info("Paused by the owner")

    def resume(self) -> None:
        self.state.set_paused(False)
        self.state.set_limit_until(None)
        self._takeover_until.clear()
        log.info("Resumed by the owner")

    def status_text(self) -> str:
        if self.settings.auto_send:
            mode = "AUTO-SEND (replies are sent from your account)"
        else:
            mode = "PREVIEW (replies are only shown in the logs)"
        lines = [
            f"Mode: {mode}",
            f"State: {'paused' if self.state.paused else 'running'}",
        ]
        limit_until = self.state.limit_until
        if limit_until is not None and limit_until > self.clock():
            lines.append(f"Claude usage limit: waiting until {_format_time(limit_until)}")
        else:
            lines.append("Claude usage limit: none")
        lines.append(f"Chats in history: {self.history.chat_count()}")
        lines.append(f"Claude auth: {self.settings.claude_auth_mode}")
        return "\n".join(lines)

    def start_text(self) -> str:
        s = self.settings
        username = f"@{self.bot_username}" if self.bot_username else "shu botning username'i"
        if s.auto_send:
            mode = "AUTO-SEND — javoblar sizning nomingizdan yuboriladi"
        else:
            mode = "PREVIEW — javoblar faqat logda ko'rinadi, hech kimga yuborilmaydi"
        state = "pauzada (/resume bilan davom ettiring)" if self.state.paused else "ishlayapti"
        if s.owner_takeover_minutes > 0:
            takeover = f"• Chatda o'zingiz yozsangiz, bot o'sha chatda {s.owner_takeover_minutes} daqiqa jim turadi."
        else:
            takeover = "• O'zingiz yozganingizda jim turish o'chirilgan (OWNER_TAKEOVER_MINUTES=0)."
        lines = [
            "Salom! Men sizning shaxsiy yordamchingizman.",
            "Sizga kimdir Telegram'da yozsa, profile.md asosida sizning uslubingizda, "
            "sizning nomingizdan javob tayyorlayman.",
            "",
            "Hozirgi holat:",
            f"• Rejim: {mode}",
            f"• Holat: {state}",
            "",
            "Qanday ulanadi (Telegram Premium kerak):",
            "1. Telegram → Settings → Telegram Business → Chatbots",
            f"2. Bot username'ini kiriting: {username}",
            "3. Bot qaysi chatlarni ko'rishini tanlang",
            "4. Javob berish (reply) ruxsatini yoqing",
            "",
            "Buyruqlar:",
            "/pause — javob berishni to'xtatish",
            "/resume — davom ettirish",
            "/status — holat",
            "",
            "Bilib qo'ying:",
            takeover,
            "• Pul, qarz, uchrashuv va rejalarga rozilik bermaydi, va'da bermaydi.",
            "• profile.md ni to'ldiring — bot sizni shu fayl orqali taniydi.",
        ]
        if not s.auto_send:
            lines.append("• Hozir preview rejimi: javoblarni haqiqatan yuborish uchun AUTO_SEND=true qiling.")
        return "\n".join(lines)


def make_claude_cli(settings: Settings) -> ClaudeCLI:
    return ClaudeCLI(
        settings.claude_bin,
        workdir=settings.claude_workdir,
        model=settings.claude_model,
        effort=settings.claude_effort,
        timeout=settings.claude_timeout,
        oauth_token=settings.claude_oauth_token,
        auth_label=settings.claude_auth_mode,
    )


def build_assistant(settings: Settings, **kwargs: Any) -> Assistant:
    return Assistant(settings, make_claude_cli(settings), **kwargs)
