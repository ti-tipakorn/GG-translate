# daily_english_pic.py
# Telegram bot: EN rewrites + IPA + MP3, Translate+Pronounce buttons (JA/ZH-CN/ZH-TW/DE/EN-GB/FI) + Photo OCR
# Requires: python-telegram-bot 21.*, google-cloud-vision, google-cloud-texttospeech, google-cloud-translate,
#           google-genai, python-dotenv, eng_to_ipa

import os, re, io, json, difflib, asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from functools import lru_cache

# Telegram
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Google Cloud (TTS + Translate + Vision OCR)
from google.cloud import texttospeech as tts
from google.cloud import translate
from google.cloud import vision

# Gemini (Google Gen AI SDK)
from google import genai
from google.genai import types as genai_types

import inflect
_p = inflect.engine()

# .env
from dotenv import load_dotenv

# HTML escape
from html import escape as h

# Top-level singletons
tts_client = tts.TextToSpeechClient()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load env from both places (script folder and C:\english-daily) ---
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# --- Style system instruction (neutral everyday English; no extra meaning) ---
SYS_INSTRUCTION = (
    "You are an English writing coach.\n"
    "TASK: Rewrite the user's text in ENGLISH ONLY, preserving meaning exactly.\n"
    "- Do NOT add reasons, options, or new information.\n"
    "- Keep 'polished' neutral, everyday, natural English (CEFR B2), not business/formal.\n"
    "- Keep 'concise' shorter but equally clear.\n"
    "- 'formal' = professional but plain; avoid jargon.\n"
    "- 'friendly' = casual and warm; avoid slangy words unless present (e.g., no 'excursion', 'reminiscent').\n"
    "- Preserve numbers, times, dates, names, and specific details as written (e.g., '2 PM', 'September 1st').\n"
    "- Prefer: start/begin → 'start'; feasible → 'possible'; consume → 'eat'; recollections → 'memories'.\n"
    "Return ONLY the requested fields."
)

# ---------- Config ----------
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
PROJECT_ID  = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
LOCATION    = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEM_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")  # or gemini-1.5-pro

# Optional voice overrides
JA_VOICE    = os.getenv("JA_VOICE",    "ja-JP-Neural2-B")
ZH_CN_VOICE = os.getenv("ZH_CN_VOICE", "cmn-CN-Wavenet-A")
ZH_TW_VOICE = os.getenv("ZH_TW_VOICE", "cmn-TW-Wavenet-A")
DE_VOICE    = os.getenv("DE_VOICE",    "de-DE-Neural2-C")
EN_GB_VOICE = os.getenv("EN_GB_VOICE", "en-GB-Neural2-D")
FI_VOICE    = os.getenv("FI_VOICE",    "fi-FI-Standard-A")

if not BOT_TOKEN:  raise SystemExit("Set TELEGRAM_BOT_TOKEN in your .env")
if not PROJECT_ID: raise SystemExit("Set GCP_PROJECT_ID in your .env")

# Clients
translate_client = translate.TranslationServiceClient()
vision_client    = vision.ImageAnnotatorClient()
TRANS_PARENT = f"projects/{PROJECT_ID}/locations/global"

genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Store last polished text per chat so buttons can act on it
LAST_POLISHED: Dict[int, str] = {}

# ---------- OCR (Vision) ----------
def detect_text_from_image(image_bytes: bytes) -> str:
    """OCR via Google Cloud Vision. Tries to be robust for blocks of text."""
    image = vision.Image(content=image_bytes)
    ctx = vision.ImageContext(language_hints=["en", "th", "ja", "zh"])  # tweak for your use
    response = vision_client.document_text_detection(image=image, image_context=ctx)

    if response.error and response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    text = response.full_text_annotation.text if response.full_text_annotation else ""
    return (text or "").strip()

# ---------- Helpers: language-aware cleanups for EN ----------

def match_casing(dst: str, src: str) -> str:
    """Match 'dst' to the case pattern of 'src' (UPPER, lower, Title, As-Is)."""
    if src.isupper():
        return dst.upper()
    if src.islower():
        return dst.lower()
    if src.istitle():
        # Title-case each token to mirror e.g. 'Doctor' -> 'Patients'
        return " ".join(w.capitalize() if w else w for w in re.split(r"(\W+)", dst))
    # default: leave as written
    return dst

def _plural(word: str) -> str:
    # inflect returns False when it can't pluralize; fall back to naive 's'
    return _p.plural_noun(word) or (word + "s")

def _capitalize_first_word(s: str) -> str:
    """
    Capitalize the first alphabetic character (skipping leading quotes/brackets).
    """
    def up(m):
        return (m.group(1) or "") + m.group(2).upper()
    return re.sub(r'^([\s"\(\[\{]*)([a-z])', up, s)

# Emoji & punctuation endings we consider as “end punctuation”
_END_OK = re.compile(r'[\.\!\?\u2026"”’\)\]]\s*$')  # includes ellipsis …

# --- English micro-fixers used by finalize_sentence() ---
def fix_indefinite_articles_en(s: str) -> str:
    """
    Minimal 'a/an' fixer (safe heuristics).
    """
    def repl(m):
        word = m.group(2)
        # Vowel starts or vowel-sound exceptions
        if re.match(r'(?i)([aeiou])', word) or re.match(r'(?i)(honest|hour|heir|FDA|MRI)', word):
            return f"an {word}"
        # Common 'u' consonant sound / 'yoo' and similar cases
        if re.match(r'(?i)(uni([^- ]+)?|useful|user|ubiquitous|euro|one(?!\w))', word):
            return f"a {word}"
        # Generic fallback
        return ("an " if word[:1].lower() in "aeiou" else "a ") + word
    return re.sub(r'(?i)\b(a|an)\s+([A-Za-z][\w-]*)', repl, s)

def fix_quantifiers_en(s: str) -> str:
    """
    Very conservative quantifier fixes.
    """
    # 'no any' -> 'no'
    s = re.sub(r'(?i)\bno\s+any\b', 'no', s)
    # Positive 'have any' -> 'have some' (avoid touching "don't/doesn't/didn't have any")
    s = re.sub(r'(?i)\b(?<!not\s)(?<!don\'t\s)(?<!doesn\'t\s)(?<!didn\'t\s)(have|has|there\s+is|there\s+are)\s+any\b', r'\1 some', s)
    return s

def finalize_sentence(s: str, lang: str = "en") -> str:
    """
    Finalize a single sentence for English:
      - trims
      - fixes quantifiers (no/any) & a/an
      - capitalizes the first letter (skipping quotes)
      - adds a period only if there’s no end punctuation
    For non-English, it returns the string unchanged.
    """
    s = (s or "").strip()
    if not s:
        return s
    if lang != "en":
        return s  # don’t enforce English punctuation on other languages

    s = fix_quantifiers_en(s)
    s = fix_indefinite_articles_en(s)
    s = _capitalize_first_word(s)
    if not _END_OK.search(s):
        s += "."
    return s

# ---------- Fallback paraphrase (if Gemini fails) ----------
PIVOTS = ["es","fr","de","it","pt","nl","sv","no","da","pl","cs","ro"]

def same_lang(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.split("-")[0].lower() == b.split("-")[0].lower()

def translate_text(text: str, target: str, source: Optional[str] = None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Guard: avoid EN->EN, DE->DE, etc.
    if same_lang(source or "", target):
        return text

    req = {
        "parent": TRANS_PARENT,
        "contents": [text],
        "mime_type": "text/plain",
        "target_language_code": target
    }
    # Only set source if different from target base language
    if source and not same_lang(source, target):
        req["source_language_code"] = source

    resp = translate_client.translate_text(request=req)
    return resp.translations[0].translated_text

def roundtrip(text: str, pivot: str) -> str:
    try:
        return translate_text(translate_text(text, pivot, "en"), "en", pivot).strip()
    except Exception:
        return ""

def pick_variant(original: str, candidates: List[str], mode: str) -> str:
    if not candidates: return original
    scored = [(c, difflib.SequenceMatcher(None, original, c).ratio(), len(c.split())) for c in candidates]
    if mode == "polish":
        pool = [s for s in scored if 0.80 <= s[1] <= 0.95] or scored
        orig_len = len(original.split())
        pool.sort(key=lambda t: (-t[1], abs(t[2]-orig_len)))
        return pool[0][0]
    else:
        pool = [s for s in scored if s[1] >= 0.70] or scored
        pool.sort(key=lambda t: (t[2], -t[1]))
        return pool[0][0]

# ---------- Gemini response shaping ----------
REWRITE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "polished": genai_types.Schema(type=genai_types.Type.STRING),
        "concise":  genai_types.Schema(type=genai_types.Type.STRING),
        "formal":   genai_types.Schema(type=genai_types.Type.STRING),
        "friendly": genai_types.Schema(type=genai_types.Type.STRING),
        "notes":    genai_types.Schema(type=genai_types.Type.STRING),
    },
    required=["polished","concise","formal","friendly"]
)

DELIMS = ["<<<POLISHED>>>","<<<CONCISE>>>","<<<FORMAL>>>","<<<FRIENDLY>>>","<<<NOTES>>>"]
DELIMITED_INSTRUCTION = (
    "You are an English writing coach. Rewrite the user's text in ENGLISH ONLY. "
    "Preserve meaning, fix grammar and phrasing. If JSON is not possible, output EXACTLY these 5 sections, "
    "each starting with a delimiter on its own line, no extra text:\n"
    "<<<POLISHED>>>\n"
    "<<<CONCISE>>>\n"
    "<<<FORMAL>>>\n"
    "<<<FRIENDLY>>>\n"
    "<<<NOTES>>>\n"
    "Do not add any other text or markers outside these sections."
)

def parse_delimited(raw: str) -> dict:
    raw = (raw or "").strip()
    out = { "polished":"", "concise":"", "formal":"", "friendly":"", "notes":"" }
    union = '|'.join(map(re.escape, DELIMS))
    def grab(tag):
        patt = rf"{re.escape(tag)}\s*(.*?)(?=(?:{union})|\Z)"
        m = re.search(patt, raw, flags=re.S)
        return m.group(1).strip() if m else ""
    out["polished"] = grab("<<<POLISHED>>>")
    out["concise"]  = grab("<<<CONCISE>>>")
    out["formal"]   = grab("<<<FORMAL>>>")
    out["friendly"] = grab("<<<FRIENDLY>>>")
    out["notes"]    = grab("<<<NOTES>>>")
    if not (out["polished"] and out["concise"] and out["formal"] and out["friendly"]):
        raise ValueError("Delimited output missing required fields")
    return out

def _case_match2(dst: str, src: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper() and src[1:].islower():
        return dst.capitalize()
    return dst

def keep_time_tokens(original: str, rewritten: str) -> str:
    tokens = [m.group(0) for m in re.finditer(r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b", original)]
    out = rewritten
    for tok in tokens:
        if tok and tok not in out:
            out = re.sub(r"(\?|\.|!)(\s|$)", f" at {tok}. ", out, count=1)
    return out

def gemini_rewrite(text: str) -> dict:
    contents = [ genai_types.Content(role="user", parts=[genai_types.Part(text=text)]) ]
    cfg_json = genai_types.GenerateContentConfig(
        system_instruction=SYS_INSTRUCTION + " Return JSON with keys: polished, concise, formal, friendly, notes.",
        temperature=0.1,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=REWRITE_SCHEMA,
    )
    try:
        resp = genai_client.models.generate_content(model=GEM_MODEL, contents=contents, config=cfg_json)
        if getattr(resp, "parsed", None):
            data = resp.parsed
        else:
            data = json.loads(resp.text or "")
        for k in ["polished","concise","formal","friendly","notes"]:
            data[k] = (data.get(k) or "").strip()
        return data
    except Exception as e_json:
        cfg_tag = genai_types.GenerateContentConfig(
            system_instruction=SYS_INSTRUCTION + "\n" + DELIMITED_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="text/plain",
        )
        try:
            resp2 = genai_client.models.generate_content(model=GEM_MODEL, contents=contents, config=cfg_tag)
            data = parse_delimited(resp2.text or "")
            for k in ["polished","concise","formal","friendly","notes"]:
                data[k] = (data.get(k) or "").strip()
            return data
        except Exception as e_tag:
            raise RuntimeError(f"Gemini JSON+delimited failed: {type(e_json).__name__} / {type(e_tag).__name__}")

# ---------- Style tweak ----------
def apply_style(text: str, style: str) -> str:
    if style == "formal":
        repl = {"can't":"cannot","won't":"will not","don't":"do not","I'm":"I am","it's":"it is",
                "we're":"we are","gonna":"going to","wanna":"want to","a lot":"a great deal"}
    elif style == "friendly":
        repl = {"do not":"don't","cannot":"can't","will not":"won't","it is":"it's",
                "we are":"we're","I am":"I'm","going to":"gonna","want to":"wanna"}
    else:
        return text
    out = text
    for k,v in repl.items(): out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    return out

# ---------- IPA (English only) ----------
def build_ipa(text: str) -> str:
    try:
        import eng_to_ipa as ipa
        return ipa.convert(text)
    except Exception:
        return ""

# ---------- TTS ----------

def tts_mp3_en(text: str, voice_name: str = "en-US-Neural2-D", rate: float = 1.0, pitch: float = 0.0) -> bytes:
    if not text.strip(): return b""
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3, speaking_rate=rate, pitch=pitch)
    return tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config).audio_content

def tts_mp3(text: str, language_code: str, voice_name: str) -> bytes:
    if not text.strip(): return b""
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    return tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config).audio_content

@lru_cache(maxsize=None)
def _voices_for(lang_code: str) -> List[str]:
    return [v.name for v in tts_client.list_voices(language_code=lang_code).voices]

def prefer_voice(lang_code: str, desired: Optional[str], *candidates: str) -> str:
    available = set(_voices_for(lang_code))
    if desired and desired in available:
        return desired
    for cand in candidates:
        if cand in available:
            return cand
    return next(iter(available)) if available else ""

# ---------- Text processing ----------
def process_text(input_text: str) -> Dict[str, str]:
    """Processes input text via Gemini API and prepares the response dictionary (HTML)."""
    text = (input_text or "").strip()
    if not text:
        text = "I will focus on small improvements every day to become more confident in English."
    if len(text) > 4000:
        text = text[:4000]

    try:
        data = gemini_rewrite(text)
        polished = finalize_sentence(data.get("polished", ""))
        concise = finalize_sentence(data.get("concise", ""))
        formal = finalize_sentence(data.get("formal", ""))
        friendly = finalize_sentence(data.get("friendly", ""))
        notes = (data.get("notes", "") or "").strip()

    except Exception as e:
        # Fallback to a simple error message if the API call fails
        polished = "Sorry, I couldn't rewrite this sentence."
        concise = polished
        formal = polished
        friendly = polished
        notes = f"Gemini error: {type(e).__name__}: {str(e)[:180]}"

    ipa_pol = build_ipa(polished)
    ipa_con = build_ipa(concise)

    parts = []
    parts.append("📝 <b>Native Polish</b>")
    parts.append(f"• <b>Polished:</b> {h(polished)}")
    parts.append("\n✂️ <b>Concise</b>")
    parts.append(h(concise))
    parts.append("\n🎭 <b>Style Variants</b>")
    parts.append(f"• <b>Formal:</b> {h(formal)}")
    parts.append(f"• <b>Friendly:</b> {h(friendly)}")
    if notes:
        parts.append("\n💡 <b>Notes</b>\n" + h(notes))
    if ipa_pol or ipa_con:
        parts.append("\n🔉 <b>Pronunciation (IPA)</b>")
    if ipa_pol:
        parts.append(f"• <b>Polished:</b> {h(ipa_pol)}")
    if ipa_con:
        parts.append(f"• <b>Concise:</b> {h(ipa_con)}")

    return {"reply_text": "\n".join(parts), "polished": polished}

# ---------- Translate + Pronounce buttons ----------
LANG_OPTS = {
    "ja":   {"label":"🇯🇵 Japanese", "tgt":"ja",     "tts_lang":"ja-JP",
             "voice": lambda: prefer_voice("ja-JP", JA_VOICE,
                                            "ja-JP-Neural2-D","ja-JP-Wavenet-D","ja-JP-Standard-D")},
    "zh":   {"label":"🇨🇳 Chinese (Mainland)", "tgt":"zh-CN", "tts_lang":"cmn-CN",
             "voice": lambda: prefer_voice("cmn-CN", ZH_CN_VOICE,
                                            "cmn-CN-Wavenet-A","cmn-CN-Standard-A")},
    "zhTW": {"label":"🇹🇼 Chinese (Taiwan)",   "tgt":"zh-TW", "tts_lang":"cmn-TW",
             "voice": lambda: prefer_voice("cmn-TW", ZH_TW_VOICE,
                                            "cmn-TW-Wavenet-A","cmn-TW-Standard-A")},
    "de":   {"label":"🇩🇪 German",   "tgt":"de",  "tts_lang":"de-DE",
             "voice": lambda: prefer_voice("de-DE", DE_VOICE,
                                            "de-DE-Neural2-C","de-DE-Wavenet-A","de-DE-Standard-A")},
    "enGB": {"label":"🇬🇧 English (UK)", "tgt":"en",  "tts_lang":"en-GB",
             "voice": lambda: prefer_voice("en-GB", EN_GB_VOICE,
                                            "en-GB-Neural2-D","en-GB-Wavenet-D","en-GB-Standard-D")},
    "fi":   {"label":"🇫🇮 Finnish",  "tgt":"fi",  "tts_lang":"fi-FI",
             "voice": lambda: prefer_voice("fi-FI", FI_VOICE,
                                            "fi-FI-Wavenet-A","fi-FI-Standard-A")},
}

def build_lang_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    order_keys = ["ja"] + sorted([k for k in LANG_OPTS.keys() if k != "ja"], key=lambda k: LANG_OPTS[k]["label"])
    for key in order_keys:
        row.append(InlineKeyboardButton(LANG_OPTS[key]["label"], callback_data=f"tr:{key}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

def translate_for_buttons(text_en: str, key: str) -> Tuple[str, bytes, str]:
    """Return (translated_text, mp3_bytes, human_label)."""
    opt = LANG_OPTS[key]
    tgt = opt["tgt"]
    label = opt["label"]

    # No-op translate for EN-GB (still English text, UK voice)
    if tgt.split('-')[0].lower() == 'en':
        translated = text_en
    else:
        translated = translate_text(text_en, target=tgt, source="en")

    mp3 = tts_mp3(translated, language_code=opt["tts_lang"], voice_name=opt["voice"]())
    return translated, mp3, label

# ---------- Helpers: safe long message sender ----------
async def send_long_message(message, text: str, parse_mode: Optional[str] = "HTML"):
    """
    Sends `text` in chunks within Telegram's 4096-char limit.
    Tries to split on lines first to avoid breaking HTML tags mid-token.
    """
    MAX_LEN = 4000  # keep some headroom under 4096
    if len(text) <= MAX_LEN:
        await message.reply_text(text, parse_mode=parse_mode, disable_web_page_preview=True)
        return

    lines = text.split("\n")
    buf = ""
    for line in lines:
        # +1 for the newline if buffer not empty
        extra = (1 if buf else 0) + len(line)
        if len(buf) + extra <= MAX_LEN:
            buf = (buf + "\n" + line) if buf else line
        else:
            if buf:
                await message.reply_text(buf, parse_mode=parse_mode, disable_web_page_preview=True)
            # If the single line is too long, hard-split it
            if len(line) > MAX_LEN:
                for i in range(0, len(line), MAX_LEN):
                    part = line[i:i+MAX_LEN]
                    if i == 0:
                        buf = part
                    else:
                        await message.reply_text(buf, parse_mode=parse_mode, disable_web_page_preview=True)
                        buf = part
            else:
                buf = line
    if buf:
        await message.reply_text(buf, parse_mode=parse_mode, disable_web_page_preview=True)

# ---------- Handlers ----------
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
    await update.message.reply_text(msg_html, parse_mode="HTML", disable_web_page_preview=True)
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
            buf = io.BytesIO(audio); buf.name = "polished_en.mp3"
            caption = "🎧 Pronunciation: Polished (EN)"
            await update.message.reply_audio(
                audio=InputFile(buf, filename="polished_en.mp3"),
                caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text("Translate & pronounce:", reply_markup=build_lang_keyboard())

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    if doc.file_size and doc.file_size > 2_000_000:
        await update.message.reply_text("File is too large. Please send a smaller .txt file (≤2 MB).")
        return
    if not (doc.mime_type or "").startswith("text") and not (doc.file_name or "").lower().endswith((".txt", ".md", ".csv")):
        await update.message.reply_text("Please send a plain text file (.txt/.md).")
        return

    file = await doc.get_file()
    bio = io.BytesIO()
    
    # Use the modern, non-blocking method
    await file.download_to_memory(out=bio)
    
    raw = bio.getvalue()
    text = None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore")
            break
        except Exception:
            continue
    if text is None:  # ultimate fallback
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
                audio=InputFile(buf, filename="polished_en.mp3"),
                caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text("Translate & pronounce:", reply_markup=build_lang_keyboard())

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
    
    file = await update.message.photo[-1].get_file()

    bio = io.BytesIO()
    
    # Use the modern, non-blocking method
    await file.download_to_memory(out=bio)
    
    image_bytes = bio.getvalue()
    await update.message.reply_text("🔎 Scanning the image for text…")
    try:
        text = await asyncio.to_thread(detect_text_from_image, image_bytes)
    except Exception as e:
        await update.message.reply_text(f"Sorry, OCR failed: {type(e).__name__}: {str(e)[:160]}")
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
                audio=InputFile(buf, filename="polished_en.mp3"),
                caption=caption
            )
    except Exception:
        pass
    await update.message.reply_text("Translate & pronounce:", reply_markup=build_lang_keyboard())

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
        await query.edit_message_text("Send me a sentence first, then tap the buttons.")
        return

    try:
        translated, mp3, label = translate_for_buttons(base, key)
        # Text
        await send_long_message(query.message, f"{label}\n\n{translated}", parse_mode=None)
        # Audio
        if mp3:
            buf = io.BytesIO(mp3); buf.name = f"pronounce_{key}.mp3"
            await query.message.reply_audio(
                audio=InputFile(buf, filename=f"pronounce_{key}.mp3"),
                caption=f"🎧 Pronunciation: {label}"
            )
    except Exception as e:
        await query.message.reply_text(f"Sorry, translation/TTS failed: {type(e).__name__}: {str(e)[:180]}")

# ---------- Diagnostics ----------
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
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        cfg = genai_types.GenerateContentConfig(system_instruction="Reply 'ok'")
        r = client.models.generate_content(model=GEM_MODEL, contents="ping", config=cfg)
        msg += "✅ Gemini reachable"
    except Exception as e:
        msg += f"❌ Gemini error: {type(e).__name__}: {str(e)[:200]}"
    await update.message.reply_text(msg)

# ---------- Main ----------
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CallbackQueryHandler(on_translate_button, pattern=r"^tr:"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
