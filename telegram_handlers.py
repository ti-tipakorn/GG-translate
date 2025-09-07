"""Telegram handlers and message utilities."""

import asyncio
import io
import re
import logging
from typing import Dict, Optional, Tuple, List

from telegram import (
    Update,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from gpt_processing import (
    process_text,
    detect_text_from_image,
    translate_text,
    PROJECT_ID,
    LOCATION,
    GEM_MODEL,
)
from tts_utils import (
    tts_mp3_en,
    tts_mp3,
    prefer_voice,
    JA_VOICE,
    ZH_CN_VOICE,
    ZH_TW_VOICE,
    DE_VOICE,
    EN_GB_VOICE,
    FI_VOICE,
)


logger = logging.getLogger(__name__)


LAST_POLISHED: Dict[int, str] = {}


# ---------------------------------------------------------------------------
# Language options and helpers
# ---------------------------------------------------------------------------

LANG_OPTS = {
    "ja": {
        "label": "🇯🇵 Japanese",
        "tgt": "ja",
        "tts_lang": "ja-JP",
        "voice": lambda: prefer_voice(
            "ja-JP",
            JA_VOICE,
            "ja-JP-Neural2-D",
            "ja-JP-Wavenet-D",
            "ja-JP-Standard-D",
        ),
    },
    "zh": {
        "label": "🇨🇳 Chinese (Mainland)",
        "tgt": "zh-CN",
        "tts_lang": "cmn-CN",
        "voice": lambda: prefer_voice(
            "cmn-CN",
            ZH_CN_VOICE,
            "cmn-CN-Wavenet-A",
            "cmn-CN-Standard-A",
        ),
    },
    "zhTW": {
        "label": "🇹🇼 Chinese (Taiwan)",
        "tgt": "zh-TW",
        "tts_lang": "cmn-TW",
        "voice": lambda: prefer_voice(
            "cmn-TW",
            ZH_TW_VOICE,
            "cmn-TW-Wavenet-A",
            "cmn-TW-Standard-A",
        ),
    },
    "de": {
        "label": "🇩🇪 German",
        "tgt": "de",
        "tts_lang": "de-DE",
        "voice": lambda: prefer_voice(
            "de-DE",
            DE_VOICE,
            "de-DE-Neural2-C",
            "de-DE-Wavenet-A",
            "de-DE-Standard-A",
        ),
    },
    "enGB": {
        "label": "🇬🇧 English (UK)",
        "tgt": "en",
        "tts_lang": "en-GB",
        "voice": lambda: prefer_voice(
            "en-GB",
            EN_GB_VOICE,
            "en-GB-Neural2-D",
            "en-GB-Wavenet-D",
            "en-GB-Standard-D",
        ),
    },
    "fi": {
        "label": "🇫🇮 Finnish",
        "tgt": "fi",
        "tts_lang": "fi-FI",
        "voice": lambda: prefer_voice(
            "fi-FI",
            FI_VOICE,
            "fi-FI-Wavenet-A",
            "fi-FI-Standard-A",
        ),
    },
}


def build_lang_keyboard() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    order_keys = ["ja"] + sorted(
        [k for k in LANG_OPTS.keys() if k != "ja"],
        key=lambda k: LANG_OPTS[k]["label"],
    )
    for key in order_keys:
        row.append(
            InlineKeyboardButton(LANG_OPTS[key]["label"], callback_data=f"tr:{key}")
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def translate_for_buttons(text_en: str, key: str) -> Tuple[str, bytes, str]:
    """Return (translated_text, mp3_bytes, human_label)."""
    opt = LANG_OPTS[key]
    tgt = opt["tgt"]
    label = opt["label"]

    if tgt.split("-")[0].lower() == "en":
        translated = text_en
    else:
        translated = translate_text(text_en, target=tgt, source="en")

    mp3 = tts_mp3(
        translated, language_code=opt["tts_lang"], voice_name=opt["voice"]()
    )
    return translated, mp3, label


async def send_long_message(
    message, text: str, parse_mode: Optional[str] = "HTML"
):
    """Send ``text`` in chunks within Telegram's message length limit."""

    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await message.reply_text(
            text, parse_mode=parse_mode, disable_web_page_preview=True
        )
        return

    lines = text.split("\n")
    buf = ""
    for line in lines:
        extra = (1 if buf else 0) + len(line)
        if len(buf) + extra <= MAX_LEN:
            buf = (buf + "\n" + line) if buf else line
        else:
            if buf:
                await message.reply_text(
                    buf, parse_mode=parse_mode, disable_web_page_preview=True
                )
            if len(line) > MAX_LEN:
                for i in range(0, len(line), MAX_LEN):
                    part = line[i : i + MAX_LEN]
                    if i == 0:
                        buf = part
                    else:
                        await message.reply_text(
                            buf, parse_mode=parse_mode, disable_web_page_preview=True
                        )
                        buf = part
            else:
                buf = line
    if buf:
        await message.reply_text(
            buf, parse_mode=parse_mode, disable_web_page_preview=True
        )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error", exc_info=context.error)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg_html = (
        "Send me a <b>sentence</b>, a <b>photo of text</b>, or a <b>.txt</b> file.\n"
        "I'll reply with:\n"
        "• Polished (native) rewrite (Gemini)\n"
        "• Concise rewrite\n"
        "• Formal/Friendly variants\n"
        "• IPA and an MP3 voice clip\n\n"
        "Then tap a button to <b>Translate + Pronounce</b> (JA, ZH-CN, ZH-TW, DE, EN-GB, FI)."
    )
    await update.message.reply_text(
        msg_html, parse_mode="HTML", disable_web_page_preview=True
    )
    await update.message.reply_text("Quick actions:", reply_markup=build_lang_keyboard())


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    res = process_text(text)
    await send_long_message(update.message, res["reply_text"], parse_mode="HTML")
    chat_id = update.effective_chat.id
    LAST_POLISHED[chat_id] = res["polished"]
    try:
        audio = tts_mp3_en(res["polished"])
        if audio:
            buf = io.BytesIO(audio)
            buf.name = "polished_en.mp3"
            caption = "🎧 Pronunciation: Polished (EN)"
            await update.message.reply_audio(
                audio=InputFile(buf, filename="polished_en.mp3"), caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text(
        "Translate & pronounce:", reply_markup=build_lang_keyboard()
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    if doc.file_size and doc.file_size > 2_000_000:
        await update.message.reply_text(
            "File is too large. Please send a smaller .txt file (≤2 MB)."
        )
        return
    if not (doc.mime_type or "").startswith("text") and not (doc.file_name or "").lower().endswith((".txt", ".md", ".csv")):
        await update.message.reply_text(
            "Please send a plain text file (.txt/.md)."
        )
        return

    file = await doc.get_file()
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)

    raw = bio.getvalue()
    text = None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore")
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="ignore")

    res = process_text(text)
    await send_long_message(update.message, res["reply_text"], parse_mode="HTML")
    chat_id = update.effective_chat.id
    LAST_POLISHED[chat_id] = res["polished"]
    try:
        audio = tts_mp3_en(res["polished"])
        if audio:
            buf = io.BytesIO(audio)
            buf.name = "polished_en.mp3"
            caption = "🎧 Pronunciation: Polished (EN)"
            await update.message.reply_audio(
                audio=InputFile(buf, filename="polished_en.mp3"), caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text(
        "Translate & pronounce:", reply_markup=build_lang_keyboard()
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return

    file = await update.message.photo[-1].get_file()
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    image_bytes = bio.getvalue()
    await update.message.reply_text("🔎 Scanning the image for text…")
    try:
        text = await asyncio.to_thread(detect_text_from_image, image_bytes)
    except Exception as e:
        await update.message.reply_text(
            f"Sorry, OCR failed: {type(e).__name__}: {str(e)[:160]}"
        )
        return
    if not text:
        await update.message.reply_text("I couldn't find any text in that image.")
        return

    text = re.sub(r"\s+", " ", text).strip()
    preview = (text[:220] + "…") if len(text) > 240 else text
    await update.message.reply_text(f"🖼️ OCR text (preview):\n{preview}")

    res = process_text(text)
    await send_long_message(update.message, res["reply_text"], parse_mode="HTML")
    chat_id = update.effective_chat.id
    LAST_POLISHED[chat_id] = res["polished"]
    try:
        audio = tts_mp3_en(res["polished"])
        if audio:
            buf = io.BytesIO(audio)
            buf.name = "polished_en.mp3"
            caption = "🎧 Pronunciation: Polished (EN)"
            await update.message.reply_audio(
                audio=InputFile(buf, filename="polished_en.mp3"), caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text(
        "Translate & pronounce:", reply_markup=build_lang_keyboard()
    )


async def on_translate_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles inline keyboard presses like 'tr:ja', 'tr:zh', etc."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat.id
    data = query.data or ""
    try:
        _, key = data.split(":")
    except ValueError:
        return

    base = LAST_POLISHED.get(chat_id)
    if not base:
        await query.edit_message_text(
            "Send me a sentence first, then tap the buttons."
        )
        return

    try:
        translated, mp3, label = translate_for_buttons(base, key)
        await send_long_message(
            query.message, f"{label}\n\n{translated}", parse_mode=None
        )
        if mp3:
            buf = io.BytesIO(mp3)
            buf.name = f"pronounce_{key}.mp3"
            await query.message.reply_audio(
                audio=InputFile(buf, filename=f"pronounce_{key}.mp3"),
                caption=f"🎧 Pronunciation: {label}",
            )
    except Exception as e:
        await query.message.reply_text(
            f"Sorry, translation/TTS failed: {type(e).__name__}: {str(e)[:180]}"
        )


async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import sys, google, google.genai as ggen

    msg = (
        f"SDKs OK\n"
        f"- python={sys.version.split()[0]}\n"
        f"- google-genai={getattr(ggen, '__version__', 'unknown')}\n"
        f"- project={PROJECT_ID}\n"
        f"- location={LOCATION}\n"
        f"- model={GEM_MODEL}\n"
    )
    try:
        client = ggen.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        cfg = ggen.types.GenerateContentConfig(system_instruction="Reply 'ok'")
        client.models.generate_content(model=GEM_MODEL, contents="ping", config=cfg)
        msg += "✅ Gemini reachable"
    except Exception as e:
        msg += f"❌ Gemini error: {type(e).__name__}: {str(e)[:200]}"
    await update.message.reply_text(msg)


__all__ = [
    "start",
    "handle_text",
    "handle_document",
    "handle_photo",
    "on_translate_button",
    "send_long_message",
    "diag",
    "on_error",
]

