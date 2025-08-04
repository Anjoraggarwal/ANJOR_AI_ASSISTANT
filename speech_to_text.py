import os
from groq import Groq

def transcribe_with_groq(audio_filepath):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
    stt_model = "whisper-large-v3"
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text
