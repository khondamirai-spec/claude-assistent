"""Runs the Claude Code CLI (``claude -p``) once per reply.

Replies are generated with the owner's Claude *subscription*: either a
``claude setup-token`` token (``CLAUDE_CODE_OAUTH_TOKEN``) or, for local tests,
the machine's own ``claude`` login. An Anthropic API key is never used.

Safety rules:

* no shell; argv is a fixed list and the conversation goes only through stdin;
* the system prompt (with the profile) is written to a 0600 temp file, so it
  never shows up in ``ps``;
* every customisation (tools, MCP, hooks, plugins, skills, CLAUDE.md, memory)
  is switched off;
* the child environment is scrubbed of API keys and of the bot token;
* the subscription token is redacted from all CLI output before parsing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .store import ensure_private_dir

PathLike = Union[str, "os.PathLike[str]"]

# Built-in plugins shipped inside the CLI (checked with Claude Code 2.1.291 via
# the "plugins" field of the stream-json init event). All of them are disabled.
BUILTIN_PLUGINS = (
    "cc-plugin-agents-md@builtin",
    "cc-plugin-telemetry@builtin",
    "cc-plugin-plugin-authoring@builtin",
)

ISOLATION_SETTINGS: Dict[str, Any] = {
    "disableAllHooks": True,
    "enabledPlugins": {name: False for name in BUILTIN_PLUGINS},
}

# Removed from the child environment (plus everything starting with ANTHROPIC_).
STRIPPED_ENV = (
    "TELEGRAM_BOT_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    # Set by --bare; it makes the CLI ignore the OAuth login.
    "CLAUDE_CODE_SIMPLE",
)

ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}

INIT_KEYS = ("tools", "mcp_servers", "plugins", "skills", "apiKeySource", "apiProvider", "model")

REDACTED = "[REDACTED]"
_ANY_TOKEN_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")

_LIMIT_RE = re.compile(
    r"usage limit|hit your (usage )?limit|limit reached|rate[ _-]?limit|out of (extra )?usage|quota",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"not logged in|/login|invalid api key|authentication|unauthori[sz]ed|oauth token",
    re.IGNORECASE,
)
_EPOCH_RE = re.compile(r"\|\s*(\d{9,11})\b")
_RESET_TEXT_RE = re.compile(r"\bresets\s+([^\n|]+)", re.IGNORECASE)


class ClaudeError(Exception):
    """A failure that is safe to show to the owner (never contains the token)."""


class ClaudeTimeout(ClaudeError):
    pass


class ClaudeAuthError(ClaudeError):
    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class ClaudeLimitError(ClaudeError):
    def __init__(
        self,
        detail: str = "",
        resets_at: Optional[float] = None,
        reset_text: Optional[str] = None,
    ) -> None:
        self.detail = detail
        self.resets_at = resets_at
        self.reset_text = reset_text
        if reset_text:
            when = reset_text
        elif resets_at:
            when = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(resets_at))
        else:
            when = "time unknown"
        super().__init__(f"Claude subscription usage limit reached (resets {when})")


def build_env(oauth_token: Optional[str] = None, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Environment for the ``claude`` child process."""
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith("ANTHROPIC_") or key in STRIPPED_ENV:
            del env[key]
    env.update(ISOLATION_ENV)
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


def classify_failure(message: str, status: Optional[int] = None) -> ClaudeError:
    """Turn CLI error text into the matching exception (returned, not raised)."""
    text = (message or "").strip()
    if status == 429 or _LIMIT_RE.search(text):
        match = _EPOCH_RE.search(text)
        resets_at = float(match.group(1)) if match else None
        detail = _EPOCH_RE.sub("", text).strip()
        reset_match = _RESET_TEXT_RE.search(detail)
        reset_text = reset_match.group(1).strip().rstrip(".").strip()[:80] if reset_match else None
        return ClaudeLimitError(detail[:500], resets_at, reset_text or None)
    if status in (401, 403) or _AUTH_RE.search(text):
        return ClaudeAuthError(
            "Claude CLI is not logged in or the login was refused. "
            "Run `claude` on this machine and use /login.",
            detail=text[:300],
        )
    return ClaudeError(f"Claude CLI failed: {text[:500]}" if text else "Claude CLI failed")


def parse_result(returncode: Optional[int], stdout: str, stderr: str) -> str:
    """Extract the reply from ``--output-format json`` / ``stream-json`` output."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    data: Any = None
    if lines:
        try:
            data = json.loads(lines[-1])
        except ValueError:
            data = None
    if not isinstance(data, dict):
        raise classify_failure(stderr.strip() or stdout.strip() or f"exit code {returncode}")

    if data.get("is_error") or data.get("subtype") != "success":
        parts = [str(data.get("result") or "")]
        errors = data.get("errors")
        if isinstance(errors, list):
            parts.extend(str(e) for e in errors)
        parts.append(stderr.strip())
        message = " ".join(p for p in parts if p).strip()
        status = data.get("api_error_status")
        raise classify_failure(
            message or str(data.get("subtype") or f"exit code {returncode}"),
            status if isinstance(status, int) else None,
        )

    result = data.get("result")
    result = result.strip() if isinstance(result, str) else ""
    if not result:
        raise ClaudeError("Claude returned an empty reply")
    return result


class ClaudeCLI:
    def __init__(
        self,
        claude_bin: str,
        *,
        workdir: PathLike,
        model: Optional[str] = None,
        effort: str = "low",
        timeout: float = 120,
        oauth_token: Optional[str] = None,
        auth_label: Optional[str] = None,
    ) -> None:
        self.claude_bin = claude_bin
        self.workdir = Path(workdir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.oauth_token = oauth_token
        self.auth_label = auth_label

    def __repr__(self) -> str:
        return (
            f"ClaudeCLI(claude_bin={self.claude_bin!r}, model={self.model!r}, "
            f"effort={self.effort!r}, timeout={self.timeout!r}, auth={self.auth_mode!r})"
        )

    @property
    def auth_mode(self) -> str:
        if self.auth_label:
            return self.auth_label
        return "subscription token" if self.oauth_token else "this machine's claude login"

    def build_command(self, system_prompt_file: str, output_format: str = "json") -> List[str]:
        cmd = [
            self.claude_bin, "-p",
            "--safe-mode",
            "--settings", json.dumps(ISOLATION_SETTINGS),
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt-file", system_prompt_file,
            "--output-format", output_format,
            "--effort", self.effort,
        ]
        if output_format == "stream-json":
            cmd.append("--verbose")
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        returncode, stdout, stderr = await self._run(system_prompt, user_prompt, "json")
        return self._parse(returncode, stdout, stderr)

    async def inspect(self) -> Dict[str, Any]:
        """Send a tiny request and report how the CLI was set up (auth, tools, plugins...)."""
        returncode, stdout, stderr = await self._run("Reply with exactly: OK", "ping", "stream-json")
        init: Optional[Dict[str, Any]] = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "system" and event.get("subtype") == "init":
                init = {key: event.get(key) for key in INIT_KEYS}
                break
        reply = self._parse(returncode, stdout, stderr)
        return {"auth": self.auth_mode, "init": init, "reply": reply}

    def _parse(self, returncode: Optional[int], stdout: str, stderr: str) -> str:
        try:
            return parse_result(returncode, stdout, stderr)
        except ClaudeAuthError as exc:
            if not self.oauth_token:
                raise
            raise ClaudeAuthError(
                "Claude rejected the subscription token or it has expired: run `claude setup-token` "
                "on a computer with a browser and update CLAUDE_CODE_OAUTH_TOKEN. "
                f"CLI said: {exc.detail or '(no details)'}",
                exc.detail,
            ) from None

    def _redact(self, text: str) -> str:
        if self.oauth_token:
            text = text.replace(self.oauth_token, REDACTED)
        return _ANY_TOKEN_RE.sub(REDACTED, text)

    async def _run(self, system_prompt: str, user_prompt: str, output_format: str) -> Tuple[Optional[int], str, str]:
        workdir = ensure_private_dir(self.workdir)
        fd, prompt_path = tempfile.mkstemp(prefix="system-", suffix=".txt", dir=workdir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(system_prompt)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self.build_command(prompt_path, output_format),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workdir),
                    env=build_env(self.oauth_token),
                    start_new_session=True,
                )
            except FileNotFoundError:
                raise ClaudeError(
                    f"Claude CLI not found at {self.claude_bin}. Install Claude Code or set CLAUDE_BIN."
                ) from None
            except PermissionError:
                raise ClaudeError(f"Claude CLI at {self.claude_bin} is not executable.") from None

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(user_prompt.encode("utf-8")), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                _kill_group(proc)
                await proc.wait()
                raise ClaudeTimeout(f"Claude CLI did not answer within {self.timeout} seconds") from None
            except asyncio.CancelledError:
                _kill_group(proc)
                raise

            return (
                proc.returncode,
                self._redact(stdout.decode("utf-8", "replace")),
                self._redact(stderr.decode("utf-8", "replace")),
            )
        finally:
            try:
                os.unlink(prompt_path)
            except FileNotFoundError:
                pass


def _kill_group(proc: "asyncio.subprocess.Process") -> None:
    """Kill the CLI and everything it started (it runs in its own session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
