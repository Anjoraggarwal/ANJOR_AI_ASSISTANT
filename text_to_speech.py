import os

from elevenlabs.client import ElevenLabs
from elevenlabs import save

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """
    Tries to generate speech using ElevenLabs.
    If quota/credits are exceeded or another error occurs, falls back to Google gTTS.
    Returns the path to the generated mp3.
    """
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.text_to_speech.convert(
            text=input_text,
            voice_id="ZF6FPAbjXT4488VcRRnw",  # Replace with your actual voice_id if needed
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
        )
        save(audio, output_filepath)
        return output_filepath
    except Exception as e:
        print(f"ElevenLabs error: {e} -- falling back to Google TTS")
        return text_to_speech_with_gtts(input_text, output_filepath)

from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Generates speech using Google gTTS and saves it as an MP3 file.
    """
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    return output_filepath

if __name__ == "__main__":
    input_text = "Hi, I am doing fine, how are you? This is a test for AI with Anjor"
    output_filepath = "test_text_to_speech.mp3"
    print("Testing ElevenLabs (with Google TTS fallback)...")
    try:
        text_to_speech_with_elevenlabs(input_text, output_filepath)
        print(f"Generated: {output_filepath}")
    except Exception as e:
        print(f"TTS failed: {e}")


