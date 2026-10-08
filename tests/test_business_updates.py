from __future__ import annotations

import asyncio
import json
import stat

from aiogram.methods import GetBusinessConnection, SendMessage

from assistant.claude_cli import ClaudeAuthError, ClaudeLimitError, ClaudeTimeout
from assistant.service import MAX_MESSAGE_LENGTH, Outcome

from .conftest import CONTACT_ID, OWNER_ID, STRANGER_ID

# ------------------------------------------------------------------ preview


async def test_preview_mode_shows_reply_and_sends_nothing(make_harness):
    h = make_harness()
    h.generator.responses = ["Yaxshi, o'zingchi?"]
    await h.connect()

    outcome = await h.business_message("Salom, qalesan?")

    assert outcome == Outcome.previewed
    assert [p.reply for p in h.previews] == ["Yaxshi, o'zingchi?"]
    assert h.previews[0].incoming == "Salom, qalesan?"
    assert h.sent() == []

    previews_file = h.data_dir / "previews.jsonl"
    assert stat.S_IMODE(previews_file.stat().st_mode) == 0o600
    record = json.loads(previews_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["reply"] == "Yaxshi, o'zingchi?"
    assert record["chat_id"] == CONTACT_ID

    # The unsent reply must not pretend to be something the owner said.
    assert [item["role"] for item in h.history()] == ["other"]


async def test_preview_with_known_connection_makes_no_telegram_requests(make_harness):
    h = make_harness()
    await h.connect()
    h.session.requests.clear()

    assert await h.business_message() == Outcome.previewed
    assert h.session.requests == []


async def test_unknown_connection_is_looked_up_once_then_cached(make_harness):
    h = make_harness()
    h.set_connection("conn-1")

    assert await h.business_message("Salom") == Outcome.previewed
    assert await h.business_message("Qalesan?") == Outcome.previewed

    assert len(h.requests_of(GetBusinessConnection)) == 1
    assert len(h.session.requests) == 1


async def test_failed_connection_lookup_is_no_connection(make_harness):
    h = make_harness()
    assert await h.business_message() == Outcome.no_connection
    assert h.generator.calls == []


async def test_message_without_connection_id(make_harness):
    h = make_harness()
    assert await h.business_message(connection_id=None) == Outcome.no_connection


async def test_foreign_account_is_ignored(make_harness):
    h = make_harness()
    await h.connect(user_id=999_999)
    assert await h.business_message() == Outcome.foreign_connection
    assert h.generator.calls == []


async def test_disabled_connection_is_ignored(make_harness):
    h = make_harness()
    await h.connect(enabled=False)
    assert await h.business_message() == Outcome.connection_disabled
    assert h.generator.calls == []


async def test_any_chat_gets_a_reply_without_an_allowlist(make_harness):
    h = make_harness()
    await h.connect()
    for chat_id in (11, 222_222, 987_654_321):
        assert await h.business_message("Salom", from_id=chat_id, chat_id=chat_id) == Outcome.previewed
    assert len(h.generator.calls) == 3


# -------------------------------------------------------------- owner / bots


async def test_owner_message_is_recorded_and_never_answered(make_harness):
    h = make_harness()
    await h.connect()

    outcome = await h.business_message("Hozir kelaman", from_id=OWNER_ID, chat_id=CONTACT_ID)

    assert outcome == Outcome.owner_message
    assert h.generator.calls == []
    assert h.history() == [
        {"role": "owner", "name": "You", "text": "Hozir kelaman", "lang": "uz", "ts": h.clock()}
    ]


async def test_owner_takeover_silences_chat_then_expires(make_harness):
    h = make_harness()
    await h.connect()
    await h.business_message("Men o'zim yozaman", from_id=OWNER_ID)

    assert await h.business_message("Qachon kelasan?") == Outcome.owner_active
    assert h.generator.calls == []
    # Other chats are not affected.
    assert await h.business_message("Salom", from_id=77, chat_id=77) == Outcome.previewed

    h.clock.advance(15 * 60 + 1)
    assert await h.business_message("Javob ber") == Outcome.previewed
    # The owner's message is part of the conversation Claude sees.
    assert "[You]: Men o'zim yozaman" in h.generator.calls[-1]["user"]


async def test_takeover_can_be_turned_off(make_harness):
    h = make_harness(OWNER_TAKEOVER_MINUTES="0")
    await h.connect()
    await h.business_message("Salom", from_id=OWNER_ID)
    assert await h.business_message("Salom") == Outcome.previewed


async def test_messages_from_business_bot_and_bots_are_ignored(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Our own reply", from_id=OWNER_ID, via_business_bot=True) == Outcome.ignored_bot
    assert await h.business_message("I am a bot", from_id=555, chat_id=555, is_bot=True) == Outcome.ignored_bot
    assert h.generator.calls == []


# ---------------------------------------------------------------- duplicates


async def test_parallel_duplicates_are_handled_once(make_harness):
    h = make_harness()
    await h.connect()
    outcomes = await asyncio.gather(
        h.business_message("Salom", message_id=7),
        h.business_message("Salom", message_id=7),
    )
    assert sorted(o.value for o in outcomes) == ["duplicate", "previewed"]
    assert len(h.generator.calls) == 1


async def test_sequential_duplicates_and_restart(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Salom", message_id=8) == Outcome.previewed
    assert await h.business_message("Salom", message_id=8) == Outcome.duplicate

    h.restart()
    await h.connect()
    assert await h.business_message("Salom", message_id=8) == Outcome.duplicate
    assert len(h.generator.calls) == 1


async def test_same_message_id_in_another_chat_is_not_a_duplicate(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Salom", message_id=9) == Outcome.previewed
    assert await h.business_message("Salom", message_id=9, from_id=31, chat_id=31) == Outcome.previewed


# ------------------------------------------------------------ message kinds


async def test_service_message_is_unsupported(make_harness):
    h = make_harness()
    await h.connect()
    outcome = await h.business_message(text=None, extra={"new_chat_title": "New title"})
    assert outcome == Outcome.unsupported
    assert h.generator.calls == []


async def test_sticker_and_voice_are_described(make_harness):
    h = make_harness()
    await h.connect()
    sticker = {
        "file_id": "s1",
        "file_unique_id": "su1",
        "type": "regular",
        "width": 512,
        "height": 512,
        "is_animated": False,
        "is_video": False,
        "emoji": "😀",
    }
    voice = {"file_id": "v1", "file_unique_id": "vu1", "duration": 3}

    assert await h.business_message(text=None, extra={"sticker": sticker}) == Outcome.previewed
    assert "[Ali]: [sticker] 😀" in h.generator.calls[-1]["user"]

    assert await h.business_message(text=None, extra={"voice": voice}) == Outcome.previewed
    assert "[Ali]: [voice message]" in h.generator.calls[-1]["user"]


# ---------------------------------------------------------- history / prompts


async def test_history_is_separate_per_connection_and_chat(make_harness):
    h = make_harness()
    await h.connect("conn-1")
    await h.connect("conn-2")

    await h.business_message("alpha in chat A", chat_id=101, from_id=101, connection_id="conn-1")
    await h.business_message("beta in chat B", chat_id=202, from_id=202, connection_id="conn-1")
    await h.business_message("gamma in chat A, other connection", chat_id=101, from_id=101, connection_id="conn-2")
    await h.business_message("delta in chat A", chat_id=101, from_id=101, connection_id="conn-1")

    last_prompt = h.generator.calls[-1]["user"]
    assert "alpha in chat A" in last_prompt
    assert "delta in chat A" in last_prompt
    assert "beta" not in last_prompt
    assert "gamma" not in last_prompt
    assert len(h.history(101, "conn-2")) == 1


async def test_system_prompt_has_profile_without_comments(make_harness):
    h = make_harness()
    await h.connect()
    await h.business_message()
    system = h.generator.calls[-1]["system"]
    assert "<profile>" in system and "Toshkent" in system
    assert "template comment" not in system


async def test_language_hints(make_harness):
    h = make_harness()
    await h.connect()

    await h.business_message("Привет, как дела?")
    assert h.generator.calls[-1]["user"].endswith("Their latest message is in Russian.")

    await h.business_message("👍")
    assert h.generator.calls[-1]["user"].endswith("Their latest message is in Russian.")

    await h.business_message("Hi, are you free tonight?", from_id=40, chat_id=40)
    assert h.generator.calls[-1]["user"].endswith("Their latest message is in English.")

    await h.business_message("Bugun bo'shmisan?", from_id=41, chat_id=41)
    assert h.generator.calls[-1]["user"].endswith("Their latest message is in Uzbek.")


async def test_skip_is_not_shown(make_harness):
    h = make_harness()
    h.generator.responses = ["[SKIP]"]
    await h.connect()
    assert await h.business_message("ok") == Outcome.skipped
    assert h.previews == []
    assert not (h.data_dir / "previews.jsonl").exists()


# ---------------------------------------------------------------- auto-send


async def test_auto_send_rechecks_connection_then_sends(make_harness):
    h = make_harness(auto_send=True)
    h.generator.responses = ["Yaxshi, rahmat!"]
    await h.connect()
    h.session.requests.clear()

    assert await h.business_message("Salom, qalesan?") == Outcome.sent

    assert [type(r) for r in h.session.requests] == [GetBusinessConnection, SendMessage]
    sent = h.requests_of(SendMessage)[0]
    assert sent.chat_id == CONTACT_ID
    assert sent.text == "Yaxshi, rahmat!"
    assert sent.business_connection_id == "conn-1"
    assert h.history()[-1]["role"] == "owner"
    assert h.history()[-1]["text"] == "Yaxshi, rahmat!"
    assert h.previews == []


async def test_auto_send_splits_long_replies(make_harness):
    h = make_harness(auto_send=True)
    long_reply = "".join(str(i % 10) for i in range(9000))
    h.generator.responses = [long_reply]
    await h.connect()

    assert await h.business_message() == Outcome.sent
    chunks = [m.text for m in h.sent(CONTACT_ID)]
    assert len(chunks) == 3
    assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
    assert "".join(chunks) == long_reply
    assert all(m.business_connection_id == "conn-1" for m in h.sent(CONTACT_ID))


async def test_auto_send_without_reply_right_does_not_send(make_harness):
    h = make_harness(auto_send=True)
    await h.connect()
    h.set_connection(can_reply=False)  # changed in Telegram, no update yet

    assert await h.business_message() == Outcome.permission_denied
    assert h.sent() == []
    assert "permission to reply" in h.previews[-1].note
    assert all(item["role"] == "other" for item in h.history())


async def test_auto_send_with_disabled_connection_does_not_send(make_harness):
    h = make_harness(auto_send=True)
    await h.connect()
    h.set_connection(enabled=False)

    assert await h.business_message() == Outcome.permission_denied
    assert h.sent() == []
    assert "disabled" in h.previews[-1].note


async def test_auto_send_stops_if_owner_writes_during_generation(make_harness):
    h = make_harness(auto_send=True)
    await h.connect()

    async def owner_writes():
        assert await h.business_message("Men o'zim javob beraman", from_id=OWNER_ID) == Outcome.owner_message

    h.generator.during = owner_writes
    assert await h.business_message("Salom") == Outcome.owner_active
    assert h.sent() == []
    assert "you wrote in this chat" in h.previews[-1].note


# ------------------------------------------------------------ owner commands


async def test_pause_and_resume_are_owner_only_and_survive_restart(make_harness):
    h = make_harness()
    await h.connect()

    await h.direct_message("/pause", from_id=STRANGER_ID)
    assert h.assistant.state.paused is False

    await h.direct_message("/pause", from_id=OWNER_ID)
    assert h.assistant.state.paused is True
    assert len(h.sent(OWNER_ID)) == 1
    assert await h.business_message() == Outcome.paused
    assert h.generator.calls == []

    h.restart()
    await h.connect()
    assert await h.business_message() == Outcome.paused

    await h.direct_message("/resume", from_id=STRANGER_ID)
    assert h.assistant.state.paused is True

    await h.direct_message("/resume", from_id=OWNER_ID)
    assert await h.business_message() == Outcome.previewed
    assert h.sent(STRANGER_ID) == []


async def test_strangers_get_nothing_for_commands_and_messages(make_harness):
    h = make_harness()
    for text in ("/pause", "/resume", "/status", "salom, kimsan?", "/help"):
        await h.direct_message(text, from_id=STRANGER_ID)
    assert h.sent() == []
    assert h.generator.calls == []


async def test_status_for_owner(make_harness):
    h = make_harness()
    await h.direct_message("/status", from_id=OWNER_ID)
    text = h.sent(OWNER_ID)[-1].text
    assert "PREVIEW" in text and "replies are only shown in the logs" in text
    assert "running" in text


async def test_start_for_owner_is_an_uzbek_guide(make_harness):
    h = make_harness()
    await h.direct_message("/start", from_id=OWNER_ID)
    text = h.sent(OWNER_ID)[-1].text
    for expected in ("PREVIEW", "ishlayapti", "Telegram Business", "Chatbots", "/pause", "/resume", "/status",
                     "15 daqiqa", "AUTO_SEND=true", "profile.md"):
        assert expected in text, expected
    assert h.generator.calls == []


async def test_start_shows_auto_send_and_paused_state(make_harness):
    h = make_harness(auto_send=True)
    h.assistant.pause()
    await h.direct_message("/start", from_id=OWNER_ID)
    text = h.sent(OWNER_ID)[-1].text
    assert "AUTO-SEND" in text
    assert "pauzada" in text
    assert "AUTO_SEND=true" not in text


async def test_start_for_stranger_is_short_and_reveals_nothing(make_harness):
    h = make_harness()
    await h.direct_message("/start", from_id=STRANGER_ID)
    sent = h.sent(STRANGER_ID)
    assert len(sent) == 1
    text = sent[0].text
    assert "egasi" in text
    for private in ("Eldor", "Toshkent", "football", "/pause", "PREVIEW", "profile"):
        assert private not in text
    assert h.generator.calls == []


# ------------------------------------------------------------------- limits


async def test_limit_waits_until_reset_time(make_harness):
    h = make_harness()
    resets_at = h.clock() + 3600
    h.generator.responses = [ClaudeLimitError("usage limit reached", resets_at=resets_at)]
    await h.connect()

    assert await h.business_message("one") == Outcome.limited
    assert h.assistant.state.limit_until == resets_at
    assert await h.business_message("two") == Outcome.limited
    assert len(h.generator.calls) == 1
    assert h.sent() == []  # preview mode: the owner is not messaged

    h.clock.advance(3601)
    assert await h.business_message("three") == Outcome.previewed


async def test_limit_without_reset_time_uses_cooldown_and_survives_restart(make_harness):
    h = make_harness(LIMIT_COOLDOWN_MINUTES="30")
    h.generator.responses = [ClaudeLimitError("rate limit")]
    await h.connect()

    assert await h.business_message("one") == Outcome.limited
    assert h.assistant.state.limit_until == h.clock() + 30 * 60

    h.restart()
    await h.connect()
    h.clock.advance(29 * 60)
    assert await h.business_message("two") == Outcome.limited
    h.clock.advance(2 * 60)
    assert await h.business_message("three") == Outcome.previewed


async def test_limit_in_auto_send_notifies_owner_and_resume_clears_it(make_harness):
    h = make_harness(auto_send=True)
    h.generator.responses = [ClaudeLimitError("usage limit reached", resets_at=h.clock() + 7200)]
    await h.connect()

    assert await h.business_message() == Outcome.limited
    assert h.sent(CONTACT_ID) == []
    owner_messages = h.sent(OWNER_ID)
    assert len(owner_messages) == 1
    assert "usage limit" in owner_messages[0].text

    await h.direct_message("/resume", from_id=OWNER_ID)
    assert h.assistant.state.limit_until is None
    assert await h.business_message("again") == Outcome.sent


# ------------------------------------------------------------------- errors


async def test_timeout_and_auth_errors_send_nothing_to_the_chat_and_alert_once(make_harness):
    h = make_harness(auto_send=True)
    h.generator.responses = [
        ClaudeTimeout("Claude CLI did not answer within 120 seconds"),
        ClaudeAuthError("Claude CLI is not logged in", detail="Not logged in"),
        ClaudeTimeout("again"),
    ]
    await h.connect()

    assert await h.business_message("one") == Outcome.error
    assert await h.business_message("two") == Outcome.error
    assert h.sent(CONTACT_ID) == []
    assert len(h.sent(OWNER_ID)) == 1  # not repeated within 10 minutes

    h.clock.advance(11 * 60)
    assert await h.business_message("three") == Outcome.error
    assert len(h.sent(OWNER_ID)) == 2
    assert h.sent(CONTACT_ID) == []


async def test_errors_in_preview_mode_do_not_message_the_owner(make_harness):
    h = make_harness()
    h.generator.responses = [ClaudeTimeout("slow")]
    await h.connect()
    assert await h.business_message() == Outcome.error
    assert h.sent() == []
