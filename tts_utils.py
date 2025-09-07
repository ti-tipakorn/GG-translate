 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/tts_utils.py
index 0000000000000000000000000000000000000000..84549cfc472ee66452ebbb06ea8d8cda82f01b44 100644
--- a//dev/null
+++ b/tts_utils.py
@@ -0,0 +1,84 @@
+"""Text-to-Speech helpers and utilities."""
+
+import os
+from functools import lru_cache
+from typing import List, Optional
+
+from google.cloud import texttospeech as tts
+
+
+# Client
+tts_client = tts.TextToSpeechClient()
+
+
+# Optional voice overrides via environment variables
+JA_VOICE = os.getenv("JA_VOICE", "ja-JP-Neural2-B")
+ZH_CN_VOICE = os.getenv("ZH_CN_VOICE", "cmn-CN-Wavenet-A")
+ZH_TW_VOICE = os.getenv("ZH_TW_VOICE", "cmn-TW-Wavenet-A")
+DE_VOICE = os.getenv("DE_VOICE", "de-DE-Neural2-C")
+EN_GB_VOICE = os.getenv("EN_GB_VOICE", "en-GB-Neural2-D")
+FI_VOICE = os.getenv("FI_VOICE", "fi-FI-Standard-A")
+
+
+def tts_mp3_en(
+    text: str,
+    voice_name: str = "en-US-Neural2-D",
+    rate: float = 1.0,
+    pitch: float = 0.0,
+) -> bytes:
+    """Synthesize English text to MP3."""
+    if not text.strip():
+        return b""
+    synthesis_input = tts.SynthesisInput(text=text)
+    voice = tts.VoiceSelectionParams(language_code="en-US", name=voice_name)
+    audio_config = tts.AudioConfig(
+        audio_encoding=tts.AudioEncoding.MP3, speaking_rate=rate, pitch=pitch
+    )
+    response = tts_client.synthesize_speech(
+        input=synthesis_input, voice=voice, audio_config=audio_config
+    )
+    return response.audio_content
+
+
+def tts_mp3(text: str, language_code: str, voice_name: str) -> bytes:
+    """Generic TTS helper returning MP3 bytes."""
+    if not text.strip():
+        return b""
+    synthesis_input = tts.SynthesisInput(text=text)
+    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
+    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
+    response = tts_client.synthesize_speech(
+        input=synthesis_input, voice=voice, audio_config=audio_config
+    )
+    return response.audio_content
+
+
+@lru_cache(maxsize=None)
+def _voices_for(lang_code: str) -> List[str]:
+    """Return a list of available voice names for ``lang_code``."""
+    return [v.name for v in tts_client.list_voices(language_code=lang_code).voices]
+
+
+def prefer_voice(lang_code: str, desired: Optional[str], *candidates: str) -> str:
+    """Pick the first available voice from candidates, preferring ``desired`` if set."""
+    available = set(_voices_for(lang_code))
+    if desired and desired in available:
+        return desired
+    for cand in candidates:
+        if cand in available:
+            return cand
+    return next(iter(available)) if available else ""
+
+
+__all__ = [
+    "tts_mp3_en",
+    "tts_mp3",
+    "prefer_voice",
+    "JA_VOICE",
+    "ZH_CN_VOICE",
+    "ZH_TW_VOICE",
+    "DE_VOICE",
+    "EN_GB_VOICE",
+    "FI_VOICE",
+]
+
 
EOF
)
