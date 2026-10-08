#!/usr/bin/env python3
"""Telegram Business assistant: entry point.

    python bot.py                       run the bot (settings from .env)
    python bot.py --settings-from-env   run in a container (settings from env variables)
    python bot.py --check-claude        check how the Claude CLI is set up (no Telegram)
    python bot.py --try "text"          draft a reply to a made-up message (no Telegram)

Exit codes: 0 ok, 1 Claude check/reply failed, 2 configuration or Telegram token error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

from assistant.claude_cli import ClaudeError
from assistant.config import SETTING_KEYS, ConfigError, Settings, env_file_warning, load_settings
from assistant.language import detect_language
from assistant.prompts import SKIP_TOKEN, build_system_prompt, build_user_prompt
from assistant.service import build_assistant, make_claude_cli, profile_is_empty, read_profile
from assistant.store import ensure_private_dir

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

API_KEY_SOURCES = ("ANTHROPIC_API_KEY", "apiKeyHelper", "/login managed key")

log = logging.getLogger("bot")


def settings_from_process_env(
    *, require_telegram: bool = True, environ: Optional[MutableMapping[str, str]] = None
) -> Settings:
    """Container mode: read the settings from the environment, then remove them from it.

    They are removed even when the settings are invalid, so the ``claude`` child
    process can never inherit the bot token.
    """
    environ = os.environ if environ is None else environ
    try:
        source = {key: environ[key] for key in SETTING_KEYS if key in environ}
        return load_settings(ENV_FILE, require_telegram=require_telegram, source=source)
    finally:
        for key in SETTING_KEYS:
            environ.pop(key, None)


def check_report(report: Dict[str, Any]) -> List[str]:
    """Problems found in the ``inspect()`` report; an empty list means all good."""
    init = report.get("init")
    if not init:
        return ["the CLI did not send an init event"]
    problems = []
    for key in ("tools", "mcp_servers", "plugins", "skills"):
        if init.get(key):
            problems.append(f"{key} should be empty but is {json.dumps(init.get(key))}")
    source = init.get("apiKeySource")
    if source in API_KEY_SOURCES:
        problems.append(f"Claude is using an API key ({source}), not the subscription")
    provider = init.get("apiProvider")
    if provider not in ("firstParty", None):
        problems.append(f"Claude is using provider {provider!r}, expected firstParty")
    return problems


async def check_claude(settings: Settings) -> int:
    cli = make_claude_cli(settings)
    try:
        report = await cli.inspect()
    except ClaudeError as exc:
        print(json.dumps({"auth": cli.auth_mode, "error": str(exc)}, indent=2, ensure_ascii=False))
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    problems = check_report(report)
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"OK: Claude answered using {report['auth']}; no tools, MCP servers, plugins or skills.")
    return 0


async def try_reply(settings: Settings, text: str) -> int:
    language = detect_language(text)
    system_prompt = build_system_prompt(read_profile(settings.profile_path), datetime.now().astimezone())
    user_prompt = build_user_prompt([{"role": "other", "name": "Test contact", "text": text}], language)
    cli = make_claude_cli(settings)
    try:
        reply = await cli.generate(system_prompt, user_prompt)
    except ClaudeError as exc:
        print(f"Claude error: {exc}", file=sys.stderr)
        return 1
    print(f"Language hint: {language or 'unknown'}")
    print(f"Message:       {text}")
    if SKIP_TOKEN in reply:
        print(f"Reply:         (none, Claude answered {SKIP_TOKEN})")
    else:
        print(f"Reply:         {reply}")
    return 0


async def run_bot(settings: Settings) -> int:
    from aiogram import Bot, Dispatcher
    from aiogram.exceptions import TelegramUnauthorizedError

    from assistant.handlers import ALLOWED_UPDATES, build_router

    bot = Bot(settings.bot_token)
    try:
        try:
            me = await bot.get_me()
        except TelegramUnauthorizedError:
            log.error("Telegram rejected TELEGRAM_BOT_TOKEN. Check it with @BotFather (or /revoke and use the new one).")
            return 2

        assistant = build_assistant(settings)
        assistant.bot_username = me.username
        dispatcher = Dispatcher()
        dispatcher.include_router(build_router(assistant))

        log.info("Bot @%s is running", me.username)
        if settings.auto_send:
            log.warning("Mode: AUTO-SEND - replies are sent to people from your account")
        else:
            log.info("Mode: PREVIEW - replies are only shown in the logs, nothing is sent")
        log.info("Claude auth: %s", settings.claude_auth_mode)
        log.info("Claude CLI: %s", settings.claude_bin)
        if not os.access(settings.claude_bin, os.X_OK):
            log.warning("Claude CLI not found or not executable at %s; set CLAUDE_BIN", settings.claude_bin)
        if assistant.state.paused:
            log.warning("The assistant is paused; send /resume to the bot to start replying")
        if profile_is_empty(read_profile(settings.profile_path)):
            log.warning("%s is not filled in yet; replies will be generic", settings.profile_path.name)

        await dispatcher.start_polling(bot, allowed_updates=ALLOWED_UPDATES)
    finally:
        await bot.session.close()
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram Business assistant powered by the Claude Code CLI.")
    parser.add_argument(
        "--settings-from-env",
        action="store_true",
        help="read settings from environment variables (container/Railway mode) instead of .env",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-claude", action="store_true", help="check the Claude CLI setup and exit")
    action.add_argument("--try", dest="try_text", metavar="TEXT", help="draft a reply to TEXT and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    require_telegram = not (args.check_claude or args.try_text is not None)

    try:
        if args.settings_from_env:
            settings = settings_from_process_env(require_telegram=require_telegram)
        else:
            settings = load_settings(ENV_FILE, require_telegram=require_telegram)
            warning = env_file_warning(ENV_FILE)
            if warning:
                log.warning(warning)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    ensure_private_dir(settings.data_dir)
    ensure_private_dir(settings.claude_workdir)

    if args.check_claude:
        return asyncio.run(check_claude(settings))
    if args.try_text is not None:
        return asyncio.run(try_reply(settings, args.try_text))
    try:
        return asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
