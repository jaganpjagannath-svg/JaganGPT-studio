import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'jagangpt-super-secret-key-2026-multimodal')
    DATABASE_PATH = os.path.join(BASE_DIR, 'jagangpt.db')
    
    # Upload directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DOCS_FOLDER = os.path.join(UPLOAD_FOLDER, 'docs')
    IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
    VIDEOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos')
    AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')
    
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max file upload
    
    # API Keys
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    IMAGE_API_KEY = os.environ.get('IMAGE_API_KEY', '')
    VIDEO_API_KEY = os.environ.get('VIDEO_API_KEY', '')
    HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN', '')
    
    # AI Models
    DEFAULT_MODEL = 'jagangpt-smart'
    MODELS = {
        'jagangpt-smart': {
            'name': 'JaganGpt Smart',
            'desc': 'Deep reasoning, multimodal vision, document analysis & Telugu intelligence',
            'badge': 'Gemini 2.5'
        },
        'jagangpt-fast': {
            'name': 'JaganGpt Fast',
            'desc': 'Ultra-fast streaming for quick conversations and code generation',
            'badge': 'Speed'
        },
        'jagangpt-vision': {
            'name': 'JaganGpt Vision',
            'desc': 'Specialized in chart analysis, diagrams, OCR and image understanding',
            'badge': 'Vision'
        },
        'jagangpt-voice': {
            'name': 'JaganGpt Voice',
            'desc': 'Conversational audio intelligence optimized for real-time speech',
            'badge': 'Speech'
        }
    }

for folder in [Config.UPLOAD_FOLDER, Config.DOCS_FOLDER, Config.IMAGES_FOLDER, Config.VIDEOS_FOLDER, Config.AUDIO_FOLDER]:
    os.makedirs(folder, exist_ok=True)
