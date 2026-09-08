import os
import uuid
from gtts import gTTS
from config import Config
from services.telugu_nlp import detect_language

class VoiceService:
    """Provides Text-to-Speech and speech utilities for Telugu and English."""

    @staticmethod
    def synthesize_speech(text: str, language: str = 'auto') -> str:
        """
        Converts text to an MP3 speech file using gTTS.
        Supports Telugu ('te') and English ('en').
        """
        if not text or not text.strip():
            return ""

        # Clean text for speech (strip markdown symbols)
        clean_text = text.replace('*', '').replace('#', '').replace('`', '').replace('_', '').strip()
        # Truncate for audio length if very long
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."

        # Determine language code (supporting English, Telugu, Hindi, Tamil, Kannada, Malayalam)
        supported_langs = {'te', 'hi', 'ta', 'kn', 'ml', 'en'}
        if language == 'auto':
            detected = detect_language(clean_text)
            lang_code = 'te' if detected in ['te', 'tanglish'] else 'en'
        elif language in supported_langs:
            lang_code = language
        else:
            lang_code = 'en'

        filename = f"tts_{uuid.uuid4().hex[:12]}.mp3"
        save_path = os.path.join(Config.AUDIO_FOLDER, filename)

        try:
            tts = gTTS(text=clean_text, lang=lang_code, slow=False)
            tts.save(save_path)
            return f"/uploads/audio/{filename}"
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return ""
