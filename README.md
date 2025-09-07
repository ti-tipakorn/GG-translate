# GG-Translate — Telegram English Coach (Gemini + Google Cloud)

This Telegram bot, called GG-teacher, uses Gemini and Google Cloud to help you with multiple languages. You can send text or a picture to the bot.

- Polishes your sentence to **natural, native-like English** (Gemini on Vertex AI)
- Gives **Concise** + **Style variants** (Formal / Friendly)
- Shows **English IPA**
- Sends an **MP3** pronunciation (Google Text-to-Speech)
- Lets you **Translate + Pronounce** into many languages with one tap:
  - 🇯🇵 Japanese, 🇨🇳 Chinese (Mainland), 🇹🇼 Chinese (Taiwan), 🇩🇪 German,
    🇬🇧 English (UK voice), 🇫🇮 Finnish, 🇰🇷 Korean, 🇹🇭 Thai, 🇻🇳 Vietnamese, 🇺🇸 Spanish (US)

> Runs locally or as a Windows service. Uses Google Cloud **Text-to-Speech**, **Translation v3**, and **Vertex AI (Gemini)**.  
> The bot is English-only for *rewrites*, then you can optionally translate/voice into other languages.

---

## Demo (text-only)
```
📝 Native Polish
• Polished: I used to eat this with porridge when I was a child. Thanks for the memories.
✂️ Concise
I ate this with porridge as a kid.
🎭 Style Variants
• Formal: …
• Friendly: …
🔉 Pronunciation (IPA)
• Polished: …
```
Tap a flag button (🇯🇵 / 🇨🇳 / 🇹🇼 / 🇩🇪 / 🇬🇧 / 🇫🇮 / 🇰🇷  / 🇺k ) to get **Translate + MP3** in that language/voice.

---

## Features
- **Gemini rewrite** with strict formatting and robust fallbacks
- **English IPA** via `eng_to_ipa`
- **Google Cloud TTS** MP3 output (multi-voice, per language)
- **Google Cloud Translation v3** (with guard against EN→EN errors)
- **Inline keyboard** to trigger multiple languages
- `/diag` command to sanity-check Gemini access

---

## Requirements
- Python **3.10+** (tested with 3.13)
- A Telegram **bot token** from **@BotFather**
- Google Cloud project with APIs enabled:
  - **Vertex AI API**
  - **Cloud Translation API**
  - **Text-to-Speech API**
- Local credentials:
  - EITHER a **Service Account JSON** (for local/dev)
  - OR **Workload Identity Federation** (recommended for prod; skip the JSON key)

> **Security tip**: Never commit your key JSON or `.env` to Git.

---

## Setup

### 1) Clone & install
```bash
git clone https://github.com/ti-tipakorn/GG-translate
cd GG-translate
python -m pip install -r requirements.txt
```

### 2) Telegram token
Talk to **@BotFather** → create a bot → copy the token.

### 3) Google Cloud
- Create/choose a **project** (note its **Project ID**).
- Enable APIs: **Vertex AI**, **Cloud Translation**, **Text-to-Speech**.
- Create a **Service Account**, grant permissions (Vertex AI User, Cloud Translation User, Text-to-Speech User are typical), and download the key JSON for **local** development.
- Save it as `C:\keys\translate-sa.json` (or adjust the path in `.env`).

**  You have to pay to use the Google Translate API. **

> Prefer **Workload Identity Federation** in production to avoid storing long-lived keys.

### 4) Environment file
Create `C:\keys\.env` with:
```ini
# Core
TELEGRAM_BOT_TOKEN=PASTE_TELEGRAM_TOKEN
GCP_PROJECT_ID="you google project ID from goolge cloud"
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash-001
GOOGLE_APPLICATION_CREDENTIALS=C:\keys\translate-sa.json

# Voice overrides (optional, can change anytime)
JA_VOICE=ja-JP-Neural2-D
ZH_CN_VOICE=cmn-CN-Wavenet-A
ZH_TW_VOICE=cmn-TW-Wavenet-A
KO_VOICE=ko-KR-Neural2-B
TH_VOICE=th-TH-Neural2-C
VI_VOICE=vi-VN-Neural2-A
DE_VOICE=de-DE-Neural2-C
EN_GB_VOICE=en-GB-Neural2-D
FI_VOICE=fi-FI-Standard-A
ES_US_VOICE=es-US-Neural2-B
```
> The bot also looks for `C:\english-daily\.env` (optional). It loads both.

---

## Run locally
```bash
python C:\keys\daily_english.py
```
- Open Telegram and message your bot
- Use `/start` then send any English sentence
- Tap the language buttons for Translate + MP3

## Usage
- `/start` — help and language buttons
- Send **text** or a small **.txt** (≤ 2 MB)
- Bot replies with:
  - **Polished / Concise / Formal / Friendly**
  - **English IPA**
  - **English MP3**
- Tap a flag button to **Translate + Pronounce** in that language

### Languages/Voices
- 🇯🇵 Japanese — `ja-JP`
- 🇨🇳 Chinese (Mainland) — `zh-CN` + `cmn-CN`
- 🇹🇼 Chinese (Taiwan) — `zh-TW` + `cmn-TW`
- 🇩🇪 German — `de-DE`
- 🇬🇧 English (UK voice) — keeps EN text, speaks with `en-GB`
- 🇫🇮 Finnish — `fi-FI`
- 🇰🇷 Korean — `ko-KR`
- 🇹🇭 Thai — `th-TH`
- 🇻🇳 Vietnamese — `vi-VN`
- 🇺🇸 Spanish (US) — `es-US`

To change a **voice**, edit the corresponding `*_VOICE` in `.env` and restart the service.

---

## Add another language 
1) Add a line to `LANG_OPTS`:
```python
"fr": {"label":"🇫🇷 French", "tgt":"fr", "tts_lang":"fr-FR",
       "voice": lambda: prefer_voice("fr-FR", os.getenv("FR_VOICE"),
                                     "fr-FR-Neural2-D","fr-FR-Wavenet-D","fr-FR-Standard-D")},
```
2) (Optional) Add an env override:
```ini
FR_VOICE=fr-FR-Neural2-D
```
3) Restart the bot — the keyboard is auto-built from `LANG_OPTS`.

---

## Troubleshooting

**“Set TELEGRAM_BOT_TOKEN in your .env”**  
→ Put your token in `C:\keys\.env`, verify the path, restart.

**Gemini error / fallback used**  
→ Check Vertex AI is enabled; verify `GCP_PROJECT_ID`, `GOOGLE_CLOUD_LOCATION`, and `GEMINI_MODEL`. Try `/diag` in the bot.

**400 “Target language can't be equal to source language”**  
→ The code guards against this by skipping translation when base languages match (e.g., EN→EN for 🇬🇧). If you added a custom button, ensure `tgt` isn’t the same base language as source — or keep the guard.

**Invalid TTS voice**  
→ Use an available name. Quick check:
```python
from google.cloud import texttospeech as tts
c = tts.TextToSpeechClient()
for lc in ("ja-JP","cmn-CN","cmn-TW","de-DE","en-GB","fi-FI","ko-KR","th-TH","vi-VN","es-US"):
    print(lc, [v.name for v in c.list_voices(language_code=lc).voices])
```
Pick one from the printed list and set it in `.env` (e.g., `JA_VOICE=ja-JP-Neural2-D`).

**Service stuck PAUSED**  
- Confirm `run_bot.bat` paths.
- Set `AppDirectory` to an existing folder (e.g., `C:\english-bot`).
- Ensure Python path & libs are installed for the **same user** the service runs as.

**No IPA shown**  
→ `eng_to_ipa` isn’t installed or failed. It’s optional; the bot still works.

**Large files**  
→ .txt must be ≤ 2 MB.

---

## Project structure (suggested)
```
C:\GG-tranlate\
  daily_english.py
  .env
  translate-sa.json     

---
👨‍💻 Author
Coded by ChatGPT (OpenAI).
Customized, improved, and explained for practical use.

## License
This project is for personal use only. MIT-licensed.
