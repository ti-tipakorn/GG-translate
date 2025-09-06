from google.cloud import texttospeech as tts
c = tts.TextToSpeechClient()
voices = c.list_voices(language_code="en-US").voices
print(len(voices), "voices")