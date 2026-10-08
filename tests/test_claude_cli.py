from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from assistant.claude_cli import (
    ISOLATION_SETTINGS,
    ClaudeAuthError,
    ClaudeCLI,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
    build_env,
    classify_failure,
    parse_result,
)

TOKEN = "sk-ant-oat01-" + "T" * 60  # fake, shaped like a setup-token token
PARENT_TOKEN = "sk-ant-oat01-" + "P" * 60

FAKE_CLAUDE = r'''#!{python}
import json, os, subprocess, sys, time

args = sys.argv[1:]
record = {
    "argv": args,
    "stdin": sys.stdin.read(),
    "env": dict(os.environ),
    "cwd": os.getcwd(),
    "cwd_mode": oct(os.stat(".").st_mode & 0o777),
    "listdir": sorted(os.listdir(".")),
    "pid": os.getpid(),
}
if "--system-prompt-file" in args:
    path = args[args.index("--system-prompt-file") + 1]
    record["system_prompt_path"] = path
    record["system_prompt_mode"] = oct(os.stat(path).st_mode & 0o777)
    with open(path) as fh:
        record["system_prompt"] = fh.read()

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record["grandchild_pid"] = child.pid

with open(os.environ["FAKE_RECORD"], "w") as fh:
    json.dump(record, fh)

def emit(obj):
    print(json.dumps(obj), flush=True)

def result(text, **extra):
    data = {"type": "result", "subtype": "success", "is_error": False, "result": text}
    data.update(extra)
    emit(data)

stream = "stream-json" in args
if mode == "ok":
    if stream:
        emit({"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "plugins": [],
              "skills": [], "apiKeySource": "none", "apiProvider": "firstParty", "model": "claude-test"})
        result("OK")
    else:
        result("  Salom!  \n")
elif mode == "sleep":
    time.sleep(60)
elif mode == "limit_epoch":
    result("Claude AI usage limit reached|1760003600", is_error=True)
elif mode == "limit_text":
    result("You've hit your limit · resets 3pm (Asia/Tashkent)", is_error=True)
elif mode == "auth":
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    sys.stderr.write("OAuth token " + token + " was rejected\n")
    result("Invalid API key · Please run /login", is_error=True, api_error_status=401)
    sys.exit(1)
elif mode == "crash":
    sys.stderr.write("Traceback (most recent call last): boom\n")
    sys.exit(3)
elif mode == "empty":
    result("")
elif mode == "echo_token":
    result("token=" + os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "none"))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Returns a factory for ClaudeCLI objects that run a fake `claude` script."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(FAKE_CLAUDE.replace("{python}", sys.executable), encoding="utf-8")
    script.chmod(0o755)
    record_path = tmp_path / "record.json"
    workdir = tmp_path / "data" / "claude-run"
    monkeypatch.setenv("FAKE_RECORD", str(record_path))
    monkeypatch.setenv("FAKE_MODE", "ok")

    class Fake:
        path = script
        record_file = record_path

        def cli(self, **kwargs):
            kwargs.setdefault("workdir", workdir)
            kwargs.setdefault("timeout", 30)
            return ClaudeCLI(str(script), **kwargs)

        def mode(self, value):
            monkeypatch.setenv("FAKE_MODE", value)

        def record(self):
            return json.loads(record_path.read_text(encoding="utf-8"))

    return Fake()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


# ----------------------------------------------------------------- happy path


async def test_result_is_stripped(fake):
    assert await fake.cli().generate("system", "user") == "Salom!"


async def test_message_only_on_stdin_and_profile_only_in_private_file(fake):
    cli = fake.cli()
    await cli.generate("PROFILE-MARKER system prompt", "CHAT-MARKER hello")
    record = fake.record()

    assert record["stdin"] == "CHAT-MARKER hello"
    argv = " ".join(record["argv"])
    assert "CHAT-MARKER" not in argv
    assert "PROFILE-MARKER" not in argv
    assert record["system_prompt"] == "PROFILE-MARKER system prompt"
    assert record["system_prompt_mode"] == "0o600"
    assert not Path(record["system_prompt_path"]).exists()


async def test_temp_file_is_removed_on_error(fake):
    fake.mode("crash")
    with pytest.raises(ClaudeError):
        await fake.cli().generate("system", "user")
    assert not Path(fake.record()["system_prompt_path"]).exists()


async def test_no_shell_is_used(fake, tmp_path):
    marker = tmp_path / "pwned"
    evil = f"$(touch {marker}) `touch {marker}`; touch {marker} && echo hi"
    await fake.cli().generate(evil, evil)
    assert not marker.exists()
    assert fake.record()["stdin"] == evil


async def test_isolation_flags_in_order(fake):
    await fake.cli(effort="medium").generate("system", "user")
    argv = fake.record()["argv"]
    prompt_file = argv[argv.index("--system-prompt-file") + 1]
    assert argv == [
        "-p",
        "--safe-mode",
        "--settings", json.dumps(ISOLATION_SETTINGS),
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt-file", prompt_file,
        "--output-format", "json",
        "--effort", "medium",
    ]
    assert "--mcp-config" not in argv
    assert "--bare" not in argv
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings["disableAllHooks"] is True
    assert settings["enabledPlugins"]
    assert all(value is False for value in settings["enabledPlugins"].values())
    assert "cc-plugin-telemetry@builtin" in settings["enabledPlugins"]


async def test_model_flag_is_added_when_set(fake):
    await fake.cli(model="sonnet").generate("system", "user")
    argv = fake.record()["argv"]
    assert argv[-2:] == ["--model", "sonnet"]


async def test_child_environment_is_scrubbed(fake, monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SIMPLE",
    ):
        monkeypatch.setenv(key, "parent-value")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", PARENT_TOKEN)
    monkeypatch.setenv("HOME", "/home/fake-home")

    await fake.cli().generate("system", "user")
    env = fake.record()["env"]

    assert not [key for key in env if key.startswith("ANTHROPIC_")]
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SIMPLE",
    ):
        assert key not in env, key
    assert PARENT_TOKEN not in json.dumps(env)
    assert env["HOME"] == "/home/fake-home"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


async def test_configured_token_goes_only_to_child_env(fake, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", PARENT_TOKEN)
    cli = fake.cli(oauth_token=TOKEN, auth_label="subscription token from .env")
    await cli.generate("system prompt", "user prompt")
    record = fake.record()

    assert record["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert TOKEN not in json.dumps(record["argv"])
    assert TOKEN not in record["stdin"]
    assert TOKEN not in record["system_prompt"]
    assert TOKEN not in repr(cli)


def test_build_env_without_token_has_no_token():
    env = build_env(None, base={"CLAUDE_CODE_OAUTH_TOKEN": PARENT_TOKEN, "PATH": "/bin"})
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["PATH"] == "/bin"


async def test_token_echoed_by_cli_is_redacted(fake):
    fake.mode("echo_token")
    reply = await fake.cli(oauth_token=TOKEN).generate("system", "user")
    assert reply == "token=[REDACTED]"


async def test_rejected_token_message_hides_token_and_shows_cli_text(fake):
    fake.mode("auth")
    with pytest.raises(ClaudeAuthError) as caught:
        await fake.cli(oauth_token=TOKEN).generate("system", "user")
    message = str(caught.value)
    assert "claude setup-token" in message
    assert "CLAUDE_CODE_OAUTH_TOKEN" in message
    assert "CLI said: " in message
    assert "[REDACTED]" in message
    assert TOKEN not in message
    assert TOKEN not in caught.value.detail


async def test_auth_error_without_token_suggests_login(fake):
    fake.mode("auth")
    with pytest.raises(ClaudeAuthError) as caught:
        await fake.cli().generate("system", "user")
    assert "/login" in str(caught.value)
    assert "setup-token" not in str(caught.value)


async def test_inspect_reports_auth_and_init(fake):
    report = await fake.cli(oauth_token=TOKEN, auth_label="subscription token from .env").inspect()
    assert report["auth"] == "subscription token from .env"
    assert report["reply"] == "OK"
    assert report["init"]["plugins"] == []
    assert report["init"]["apiProvider"] == "firstParty"
    record = fake.record()
    assert record["stdin"] == "ping"
    assert record["system_prompt"] == "Reply with exactly: OK"
    assert "--verbose" in record["argv"]
    assert record["argv"][record["argv"].index("--output-format") + 1] == "stream-json"

    report = await fake.cli().inspect()
    assert report["auth"] == "this machine's claude login"


async def test_runs_in_its_own_private_empty_dir(fake, tmp_path):
    await fake.cli().generate("system", "user")
    record = fake.record()
    workdir = tmp_path / "data" / "claude-run"
    assert Path(record["cwd"]).resolve() == workdir.resolve()
    assert record["cwd_mode"] == "0o700"
    assert record["listdir"] == [Path(record["system_prompt_path"]).name]


# ------------------------------------------------------------------ failures


async def test_timeout_kills_the_whole_process_group(fake):
    fake.mode("sleep")
    started = time.monotonic()
    with pytest.raises(ClaudeTimeout):
        await fake.cli(timeout=1).generate("system", "user")
    assert time.monotonic() - started < 5
    record = fake.record()
    assert _wait_dead(record["pid"])
    assert _wait_dead(record["grandchild_pid"])


async def test_cancel_kills_the_process(fake):
    fake.mode("sleep")
    task = asyncio.ensure_future(fake.cli(timeout=30).generate("system", "user"))
    record = None
    for _ in range(200):
        await asyncio.sleep(0.05)
        try:
            record = fake.record()
            break
        except (OSError, ValueError):
            continue
    assert record is not None, "the fake claude never started"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _wait_dead(record["pid"])
    assert _wait_dead(record["grandchild_pid"])


async def test_limit_with_epoch(fake):
    fake.mode("limit_epoch")
    with pytest.raises(ClaudeLimitError) as caught:
        await fake.cli().generate("system", "user")
    assert caught.value.resets_at == 1760003600
    assert "|" not in caught.value.detail
    assert str(caught.value).startswith("Claude subscription usage limit reached (resets ")


async def test_limit_with_text(fake):
    fake.mode("limit_text")
    with pytest.raises(ClaudeLimitError) as caught:
        await fake.cli().generate("system", "user")
    assert caught.value.resets_at is None
    assert caught.value.reset_text == "3pm (Asia/Tashkent)"
    assert str(caught.value) == "Claude subscription usage limit reached (resets 3pm (Asia/Tashkent))"


async def test_crash_is_a_claude_error(fake):
    fake.mode("crash")
    with pytest.raises(ClaudeError) as caught:
        await fake.cli().generate("system", "user")
    assert type(caught.value) is ClaudeError
    assert "boom" in str(caught.value)


async def test_empty_reply_is_an_error(fake):
    fake.mode("empty")
    with pytest.raises(ClaudeError, match="empty"):
        await fake.cli().generate("system", "user")


async def test_missing_binary(tmp_path):
    cli = ClaudeCLI(str(tmp_path / "no-such-claude"), workdir=tmp_path / "run")
    with pytest.raises(ClaudeError, match="not found"):
        await cli.generate("system", "user")


# ------------------------------------------------------- classify / parse


@pytest.mark.parametrize(
    "message, status",
    [
        ("anything", 429),
        ("Claude AI usage limit reached|1760000000", None),
        ("You've hit your usage limit", None),
        ("rate_limit_error: slow down", None),
        ("You are out of extra usage", None),
        ("Quota exceeded", None),
    ],
)
def test_classify_limit(message, status):
    assert isinstance(classify_failure(message, status), ClaudeLimitError)


def test_classify_limit_details():
    error = classify_failure("Claude AI usage limit reached|1760000000")
    assert error.resets_at == 1760000000
    assert error.detail == "Claude AI usage limit reached"
    error = classify_failure("You've hit your limit · resets 3pm (Europe/Berlin)")
    assert error.reset_text == "3pm (Europe/Berlin)"


@pytest.mark.parametrize(
    "message, status",
    [
        ("whatever", 401),
        ("whatever", 403),
        ("Not logged in · Please run /login", None),
        ("Invalid API key", None),
        ("Authentication failed", None),
        ("401 Unauthorized", None),
        ("OAuth token has expired", None),
    ],
)
def test_classify_auth(message, status):
    error = classify_failure(message, status)
    assert isinstance(error, ClaudeAuthError)
    assert "/login" in str(error)
    assert error.detail == message[:300]


def test_classify_other_is_truncated():
    error = classify_failure("x" * 2000)
    assert type(error) is ClaudeError
    assert len(str(error)) < 600


def test_parse_result_success_uses_last_line():
    stdout = '{"type": "system", "subtype": "init"}\n\n{"type": "result", "subtype": "success", "is_error": false, "result": " hi "}\n'
    assert parse_result(0, stdout, "") == "hi"


def test_parse_result_error_flag_and_subtype():
    with pytest.raises(ClaudeLimitError):
        parse_result(1, json.dumps({"subtype": "success", "is_error": True, "result": "usage limit reached"}), "")
    with pytest.raises(ClaudeError, match="error_max_turns"):
        parse_result(1, json.dumps({"subtype": "error_max_turns", "is_error": False}), "")
    with pytest.raises(ClaudeAuthError):
        parse_result(1, json.dumps({"subtype": "success", "is_error": True, "result": "x", "api_error_status": 401}), "")


def test_parse_result_empty_and_non_json():
    with pytest.raises(ClaudeError, match="empty"):
        parse_result(0, json.dumps({"subtype": "success", "is_error": False, "result": "   "}), "")
    with pytest.raises(ClaudeError, match="something broke"):
        parse_result(1, "not json", "something broke")
    with pytest.raises(ClaudeError, match="not json"):
        parse_result(1, "not json", "")
    with pytest.raises(ClaudeError, match="exit code 7"):
        parse_result(7, "", "")
