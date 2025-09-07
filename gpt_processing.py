"""Gemini rewrite and general text processing utilities."""

import os
import re
import json
import difflib
from typing import List, Dict, Optional

from html import escape as h

from google.cloud import translate, vision
from google import genai
from google.genai import types as genai_types

import inflect


_p = inflect.engine()


# ---------------------------------------------------------------------------
# Environment / clients
# ---------------------------------------------------------------------------

PROJECT_ID = (
    os.getenv("GCP_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCLOUD_PROJECT")
)
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

if not PROJECT_ID:
    raise SystemExit("Set GCP_PROJECT_ID in your .env")

translate_client = translate.TranslationServiceClient()
vision_client = vision.ImageAnnotatorClient()
TRANS_PARENT = f"projects/{PROJECT_ID}/locations/global"

genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ---------------------------------------------------------------------------
# Gemini rewrite configuration
# ---------------------------------------------------------------------------

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


PIVOTS = ["es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "pl", "cs", "ro"]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def detect_text_from_image(image_bytes: bytes) -> str:
    """OCR via Google Cloud Vision. Tries to be robust for blocks of text."""

    image = vision.Image(content=image_bytes)
    ctx = vision.ImageContext(language_hints=["en", "th", "ja", "zh"])
    response = vision_client.document_text_detection(image=image, image_context=ctx)

    if response.error and response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    text = response.full_text_annotation.text if response.full_text_annotation else ""
    return (text or "").strip()


# ---------------------------------------------------------------------------
# English sentence helpers
# ---------------------------------------------------------------------------

def match_casing(dst: str, src: str) -> str:
    """Match ``dst`` to the case pattern of ``src`` (UPPER/lower/Title/As-Is)."""
    if src.isupper():
        return dst.upper()
    if src.islower():
        return dst.lower()
    if src.istitle():
        return " ".join(w.capitalize() if w else w for w in re.split(r"(\W+)", dst))
    return dst


def _plural(word: str) -> str:
    return _p.plural_noun(word) or (word + "s")


def _capitalize_first_word(s: str) -> str:
    def up(m):
        return (m.group(1) or "") + m.group(2).upper()

    return re.sub(r'^([\s"\(\[\{]*)([a-z])', up, s)


_END_OK = re.compile(r'[\.\!\?\u2026"”’\)\]]\s*$')


def fix_indefinite_articles_en(s: str) -> str:
    def repl(m):
        word = m.group(2)
        if re.match(r"(?i)([aeiou])", word) or re.match(r"(?i)(honest|hour|heir|FDA|MRI)", word):
            return f"an {word}"
        if re.match(r"(?i)(uni([^- ]+)?|useful|user|ubiquitous|euro|one(?!\w))", word):
            return f"a {word}"
        return ("an " if word[:1].lower() in "aeiou" else "a ") + word

    return re.sub(r"(?i)\b(a|an)\s+([A-Za-z][\w-]*)", repl, s)


def fix_quantifiers_en(s: str) -> str:
    s = re.sub(r"(?i)\bno\s+any\b", "no", s)
    s = re.sub(
        r"(?i)\b(?<!not\s)(?<!don\'t\s)(?<!doesn\'t\s)(?<!didn\'t\s)(have|has|there\s+is|there\s+are)\s+any\b",
        r"\1 some",
        s,
    )
    return s


def finalize_sentence(s: str, lang: str = "en") -> str:
    s = (s or "").strip()
    if not s or lang != "en":
        return s
    s = fix_quantifiers_en(s)
    s = fix_indefinite_articles_en(s)
    s = _capitalize_first_word(s)
    if not _END_OK.search(s):
        s += "."
    return s


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def same_lang(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.split("-")[0].lower() == b.split("-")[0].lower()


def translate_text(text: str, target: str, source: Optional[str] = None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if same_lang(source or "", target):
        return text

    req: Dict[str, object] = {
        "parent": TRANS_PARENT,
        "contents": [text],
        "mime_type": "text/plain",
        "target_language_code": target,
    }
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
    if not candidates:
        return original
    scored = [
        (c, difflib.SequenceMatcher(None, original, c).ratio(), len(c.split()))
        for c in candidates
    ]
    if mode == "polish":
        pool = [s for s in scored if 0.80 <= s[1] <= 0.95] or scored
        orig_len = len(original.split())
        pool.sort(key=lambda t: (-t[1], abs(t[2] - orig_len)))
        return pool[0][0]
    else:
        pool = [s for s in scored if s[1] >= 0.70] or scored
        pool.sort(key=lambda t: (t[2], -t[1]))
        return pool[0][0]


# ---------------------------------------------------------------------------
# Gemini response parsing
# ---------------------------------------------------------------------------

REWRITE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "polished": genai_types.Schema(type=genai_types.Type.STRING),
        "concise": genai_types.Schema(type=genai_types.Type.STRING),
        "formal": genai_types.Schema(type=genai_types.Type.STRING),
        "friendly": genai_types.Schema(type=genai_types.Type.STRING),
        "notes": genai_types.Schema(type=genai_types.Type.STRING),
    },
    required=["polished", "concise", "formal", "friendly"],
)


DELIMS = [
    "<<<POLISHED>>>",
    "<<<CONCISE>>>",
    "<<<FORMAL>>>",
    "<<<FRIENDLY>>>",
    "<<<NOTES>>>",
]

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
    out = {"polished": "", "concise": "", "formal": "", "friendly": "", "notes": ""}
    union = "|".join(map(re.escape, DELIMS))

    def grab(tag: str) -> str:
        patt = rf"{re.escape(tag)}\s*(.*?)(?=(?:{union})|\Z)"
        m = re.search(patt, raw, flags=re.S)
        return m.group(1).strip() if m else ""

    out["polished"] = grab("<<<POLISHED>>>")
    out["concise"] = grab("<<<CONCISE>>>")
    out["formal"] = grab("<<<FORMAL>>>")
    out["friendly"] = grab("<<<FRIENDLY>>>")
    out["notes"] = grab("<<<NOTES>>>")
    if not (
        out["polished"] and out["concise"] and out["formal"] and out["friendly"]
    ):
        raise ValueError("Delimited output missing required fields")
    return out


def _case_match2(dst: str, src: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper() and src[1:].islower():
        return dst.capitalize()
    return dst


def keep_time_tokens(original: str, rewritten: str) -> str:
    tokens = [
        m.group(0)
        for m in re.finditer(r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b", original)
    ]
    out = rewritten
    for tok in tokens:
        if tok and tok not in out:
            out = re.sub(r"(\?|\.|!)(\s|$)", f" at {tok}. ", out, count=1)
    return out


def gemini_rewrite(text: str) -> dict:
    contents = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
    ]
    cfg_json = genai_types.GenerateContentConfig(
        system_instruction=
        SYS_INSTRUCTION
        + " Return JSON with keys: polished, concise, formal, friendly, notes.",
        temperature=0.1,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=REWRITE_SCHEMA,
    )
    try:
        resp = genai_client.models.generate_content(
            model=GEM_MODEL, contents=contents, config=cfg_json
        )
        data = resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text or "")
        for k in ["polished", "concise", "formal", "friendly", "notes"]:
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
            resp2 = genai_client.models.generate_content(
                model=GEM_MODEL, contents=contents, config=cfg_tag
            )
            data = parse_delimited(resp2.text or "")
            for k in ["polished", "concise", "formal", "friendly", "notes"]:
                data[k] = (data.get(k) or "").strip()
            return data
        except Exception as e_tag:
            raise RuntimeError(
                f"Gemini JSON+delimited failed: {type(e_json).__name__} / {type(e_tag).__name__}"
            )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def apply_style(text: str, style: str) -> str:
    if style == "formal":
        repl = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "I'm": "I am",
            "it's": "it is",
            "we're": "we are",
            "gonna": "going to",
            "wanna": "want to",
            "a lot": "a great deal",
        }
    elif style == "friendly":
        repl = {
            "do not": "don't",
            "cannot": "can't",
            "will not": "won't",
            "it is": "it's",
            "we are": "we're",
            "I am": "I'm",
            "going to": "gonna",
            "want to": "wanna",
        }
    else:
        return text

    out = text
    for k, v in repl.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    return out


def build_ipa(text: str) -> str:
    try:
        import eng_to_ipa as ipa

        return ipa.convert(text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# High-level text processing
# ---------------------------------------------------------------------------


def process_text(input_text: str) -> Dict[str, str]:
    """Processes input text via Gemini API and prepares the response dictionary."""

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
        polished = "Sorry, I couldn't rewrite this sentence."
        concise = polished
        formal = polished
        friendly = polished
        notes = f"Gemini error: {type(e).__name__}: {str(e)[:180]}"

    ipa_pol = build_ipa(polished)
    ipa_con = build_ipa(concise)

    parts: List[str] = []
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


__all__ = [
    "process_text",
    "detect_text_from_image",
    "translate_text",
    "PROJECT_ID",
    "LOCATION",
    "GEM_MODEL",
]

