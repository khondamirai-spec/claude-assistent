"""aiogram router: business updates go to the Assistant; owner commands in the bot's own chat."""

from __future__ import annotations

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BusinessConnection, Message

from .service import Assistant

ALLOWED_UPDATES = ["message", "business_connection", "business_message"]

STRANGER_START_TEXT = (
    "Salom! Bu shaxsiy yordamchi bot, u faqat o'z egasi uchun ishlaydi.\n"
    "Bu yerda suhbatlashib bo'lmaydi — iltimos, egasiga to'g'ridan-to'g'ri yozing."
)


def build_router(assistant: Assistant) -> Router:
    router = Router(name="assistant")
    owner_id = assistant.settings.owner_id
    private = F.chat.type == "private"
    from_owner = F.from_user.id == owner_id

    @router.business_connection()
    async def on_business_connection(connection: BusinessConnection) -> None:
        assistant.on_business_connection(connection)

    @router.business_message()
    async def on_business_message(message: Message, bot: Bot) -> None:
        await assistant.on_business_message(bot, message)

    @router.message(CommandStart(), private)
    async def on_start(message: Message) -> None:
        if message.from_user is not None and message.from_user.id == owner_id:
            await message.answer(assistant.start_text())
        else:
            await message.answer(STRANGER_START_TEXT)

    @router.message(Command("pause"), private, from_owner)
    async def on_pause(message: Message) -> None:
        assistant.pause()
        await message.answer("Paused. I will not reply to anyone until you send /resume.")

    @router.message(Command("resume"), private, from_owner)
    async def on_resume(message: Message) -> None:
        assistant.resume()
        await message.answer("Resumed.\n\n" + assistant.status_text())

    @router.message(Command("status"), private, from_owner)
    async def on_status(message: Message) -> None:
        await message.answer(assistant.status_text())

    # Anything else people send to the bot directly is ignored on purpose.
    return router
