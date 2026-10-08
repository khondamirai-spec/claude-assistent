"""Settings loading.

Settings come from exactly one source and are never mixed:

* local development: only the ``.env`` file (read with ``dotenv_values``; the
  shell environment is ignored on purpose);
* container (Railway): a mapping that ``bot.py --settings-from-env`` builds from
  the process environment and passes in as ``source``. ``.env`` is not read.

Nothing is ever written to ``os.environ``, so secrets do not leak into child
processes by accident.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from dotenv import dotenv_values

PathLike = Union[str, "os.PathLike[str]"]

SETTING_KEYS = (
    "TELEGRAM_BOT_TOKEN",
    "OWNER_ID",
    "AUTO_SEND",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_BIN",
    "CLAUDE_MODEL",
    "CLAUDE_EFFORT",
    "CLAUDE_TIMEOUT_SECONDS",
    "LIMIT_COOLDOWN_MINUTES",
    "OWNER_TAKEOVER_MINUTES",
    "HISTORY_LIMIT",
    "DATA_DIR",
    "PROFILE_PATH",
)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

AUTH_LOCAL_LOGIN = "this machine's claude login"
AUTH_TOKEN_DOTENV = "subscription token from .env"
AUTH_TOKEN_ENVIRONMENT = "subscription token from environment variables"

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]+$")


class ConfigError(Exception):
    """Invalid or missing settings. Messages never contain secret values."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    owner_id: Optional[int]
    auto_send: bool
    claude_bin: str
    claude_model: Optional[str]
    claude_effort: str
    claude_timeout: int
    limit_cooldown_minutes: int
    owner_takeover_minutes: int
    history_limit: int
    data_dir: Path
    profile_path: Path
    claude_oauth_token: Optional[str] = field(default=None, repr=False)
    from_environment: bool = False

    @property
    def claude_auth_mode(self) -> str:
        if not self.claude_oauth_token:
            return AUTH_LOCAL_LOGIN
        return AUTH_TOKEN_ENVIRONMENT if self.from_environment else AUTH_TOKEN_DOTENV

    @property
    def claude_workdir(self) -> Path:
        return self.data_dir / "claude-run"


def load_settings(
    env_file: PathLike,
    *,
    require_telegram: bool = True,
    overrides: Optional[Mapping[str, str]] = None,
    source: Optional[Mapping[str, Optional[str]]] = None,
) -> Settings:
    """Load settings from ``env_file`` or, when given, from ``source``.

    Relative paths (and the ``data/`` / ``profile.md`` defaults) are resolved
    against the directory of ``env_file`` in both cases.
    """
    env_path = Path(env_file)
    base_dir = env_path.resolve().parent

    if source is not None:
        raw: Dict[str, Optional[str]] = dict(source)
        where = "the environment variables"
    else:
        if not env_path.is_file():
            raise ConfigError(
                f"{env_path} not found. Create it with: "
                "cp .env.example .env && chmod 600 .env"
            )
        raw = dict(dotenv_values(env_path))
        where = str(env_path)
    if overrides:
        raw.update(overrides)

    def get(key: str) -> str:
        return (raw.get(key) or "").strip()

    bot_token = get("TELEGRAM_BOT_TOKEN")
    owner_raw = get("OWNER_ID")
    owner_id: Optional[int] = int(owner_raw) if owner_raw.isdigit() else None
    if require_telegram:
        if not bot_token:
            raise ConfigError(f"TELEGRAM_BOT_TOKEN is not set in {where} (get it from @BotFather).")
        if not _BOT_TOKEN_RE.match(bot_token):
            raise ConfigError(
                f"TELEGRAM_BOT_TOKEN in {where} does not look like a bot token "
                "(expected <digits>:<text> from @BotFather)."
            )
        if not owner_raw:
            raise ConfigError(f"OWNER_ID is not set in {where} (your numeric Telegram id, see @userinfobot).")
        if owner_id is None:
            raise ConfigError(f"OWNER_ID in {where} must be your numeric Telegram user id (digits only).")
    else:
        bot_token = ""

    auto_send_raw = get("AUTO_SEND").lower() or "false"
    if auto_send_raw in _TRUE:
        auto_send = True
    elif auto_send_raw in _FALSE:
        auto_send = False
    else:
        raise ConfigError(f"AUTO_SEND in {where} must be true or false (also 1/0, yes/no, on/off).")

    oauth_token = _check_oauth_token(get("CLAUDE_CODE_OAUTH_TOKEN"), where)

    claude_bin = get("CLAUDE_BIN")
    if claude_bin:
        claude_bin = os.path.expanduser(claude_bin)
    else:
        claude_bin = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")

    effort = get("CLAUDE_EFFORT").lower() or "low"
    if effort not in EFFORT_LEVELS:
        raise ConfigError(f"CLAUDE_EFFORT in {where} must be one of: {', '.join(EFFORT_LEVELS)}.")

    return Settings(
        bot_token=bot_token,
        owner_id=owner_id,
        auto_send=auto_send,
        claude_bin=claude_bin,
        claude_model=get("CLAUDE_MODEL") or None,
        claude_effort=effort,
        claude_timeout=_int(get, "CLAUDE_TIMEOUT_SECONDS", 120, 1, where),
        limit_cooldown_minutes=_int(get, "LIMIT_COOLDOWN_MINUTES", 30, 1, where),
        owner_takeover_minutes=_int(get, "OWNER_TAKEOVER_MINUTES", 15, 0, where),
        history_limit=_int(get, "HISTORY_LIMIT", 30, 1, where),
        data_dir=_path(get("DATA_DIR") or "data", base_dir),
        profile_path=_path(get("PROFILE_PATH") or "profile.md", base_dir),
        claude_oauth_token=oauth_token,
        from_environment=source is not None,
    )


def _check_oauth_token(token: str, where: str) -> Optional[str]:
    if not token:
        return None
    if token.startswith("sk-ant-api"):
        raise ConfigError(
            f"CLAUDE_CODE_OAUTH_TOKEN in {where} is an Anthropic API key, not a subscription token. "
            "This bot only uses your Claude subscription: run `claude setup-token` and use the "
            "sk-ant-oat... token it prints."
        )
    if any(ch.isspace() for ch in token):
        raise ConfigError(
            f"CLAUDE_CODE_OAUTH_TOKEN in {where} contains spaces or line breaks. "
            "Paste the token on a single line."
        )
    if not token.startswith("sk-ant-oat"):
        raise ConfigError(
            f"CLAUDE_CODE_OAUTH_TOKEN in {where} does not look like a `claude setup-token` token "
            "(it should start with sk-ant-oat)."
        )
    return token


def _int(get, key: str, default: int, minimum: int, where: str) -> int:
    raw = get(key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} in {where} must be a whole number.") from None
    if value < minimum:
        raise ConfigError(f"{key} in {where} must be at least {minimum}.")
    return value


def _path(value: str, base_dir: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else base_dir / path


def env_file_warning(path: PathLike) -> Optional[str]:
    """Return a warning if the .env file is readable by group or others."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None
    if mode & 0o077:
        return (
            f"{path} is readable by other users (mode {oct(mode)}); it holds secrets. "
            f"Run: chmod 600 {path}"
        )
    return None
