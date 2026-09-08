import re

TELUGU_UNICODE_REGEX = re.compile(r'[\u0C00-\u0C7F]')

TANGLISH_KEYWORDS = {
    'enti', 'ela', 'cheyyi', 'cheppandi', 'cheppu', 'undi', 'undhi', 'unna', 'unnaru',
    'kuda', 'gurinchi', 'chudu', 'kavali', 'ivvandi', 'kadu', 'ledu', 'chesuko',
    'avuthundi', 'avuddi', 'cheyali', 'emiti', 'ekkada', 'epudu', 'eppudu', 'enduku',
    'meeru', 'nenu', 'manaki', 'manamu', 'telugu', 'telugulo', 'bagundi', 'bagundhi',
    'chala', 'konchem', 'inka', 'appudu', 'ippudu', 'cheppava', 'cheyavachha'
}

def detect_language(text: str) -> str:
    """
    Detects whether the input text is:
    - 'te' (pure Telugu script)
    - 'tanglish' (Telugu in Latin alphabet)
    - 'en' (standard English)
    """
    if not text or not text.strip():
        return 'en'

    # Check for Telugu Unicode characters
    if TELUGU_UNICODE_REGEX.search(text):
        return 'te'

    # Check for Tanglish keyword matches
    words = re.findall(r'[a-zA-Z]+', text.lower())
    tanglish_matches = sum(1 for w in words if w in TANGLISH_KEYWORDS)
    
    if tanglish_matches >= 1:
        return 'tanglish'

    return 'en'

def normalize_media_prompt(prompt: str) -> str:
    """
    Translates or enriches Telugu/Tanglish media generation prompts into
    rich English descriptive prompts suitable for image/video generative models.
    """
    p_lower = prompt.lower()
    
    # Common Telugu/Tanglish translations & replacements for media generation
    replacements = [
        (r'ఒక\s+', 'a '),
        (r'రూపొందించు|చేయి|generate\s+cheyyi|create\s+cheyyi', 'generate'),
        (r'చూపించే|చూపించు', 'showing'),
        (r'చిత్రం|photo|image|bomma', 'photo'),
        (r'వీడియో|video', 'cinematic video'),
        (r'futuristic\s+hyderabad', 'futuristic cyberpunk Hyderabad city with glowing neon lights, advanced skyscrapers, flying vehicles, 8k resolution'),
        (r'futuristic\s+chennai', 'futuristic cyberpunk Chennai coastal city, high-tech illuminated Marina beach, futuristic architecture, 8k cinematic'),
        (r'charminar', 'historic Charminar illuminated with modern holographic energy rings'),
        (r'robot\s+walking', 'detailed humanoid robot walking gracefully through urban streets, cinematic lighting, 4k')
    ]

    normalized = prompt
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)

    # Clean up common Tanglish request suffixes
    normalized = re.sub(r'\b(ni|lo|la|ki|ga|tho|gurinchi)\b', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

TELUGU_TO_LATIN_MAP = {
    '\u0C05': 'a', '\u0C06': 'aa', '\u0C07': 'i', '\u0C08': 'ee', '\u0C09': 'u', '\u0C0A': 'oo', '\u0C0B': 'ru',
    '\u0C0E': 'e', '\u0C0F': 'ee', '\u0C10': 'ai', '\u0C12': 'o', '\u0C13': 'oo', '\u0C14': 'au',
    '\u0C02': 'm', '\u0C03': 'h',
    '\u0C15': 'k', '\u0C16': 'kh', '\u0C17': 'g', '\u0C18': 'gh', '\u0C19': 'ng',
    '\u0C1A': 'ch', '\u0C1B': 'chh', '\u0C1C': 'j', '\u0C1D': 'jh', '\u0C1E': 'ny',
    '\u0C1F': 't', '\u0C20': 'th', '\u0C21': 'd', '\u0C22': 'dh', '\u0C23': 'n',
    '\u0C24': 'th', '\u0C25': 'th', '\u0C26': 'd', '\u0C27': 'dh', '\u0C28': 'n',
    '\u0C2A': 'p', '\u0C2B': 'ph', '\u0C2C': 'b', '\u0C2D': 'bh', '\u0C2E': 'm',
    '\u0C2F': 'y', '\u0C30': 'r', '\u0C31': 'r', '\u0C32': 'l', '\u0C33': 'l', '\u0C35': 'v',
    '\u0C36': 'sh', '\u0C37': 'sh', '\u0C38': 's', '\u0C39': 'h',
    '\u0C3E': 'aa', '\u0C3F': 'i', '\u0C40': 'ee', '\u0C41': 'u', '\u0C42': 'oo', '\u0C43': 'ru',
    '\u0C46': 'e', '\u0C47': 'ee', '\u0C48': 'ai', '\u0C4A': 'o', '\u0C4B': 'oo', '\u0C4C': 'au',
    '\u0C4D': '', # Virama
}

COMMON_TELUGU_WORDS = [
    ('నమస్కారం', 'Namaskaram'),
    ('స్వాగతం', 'Swagatham'),
    ('ధన్యవాదాలు', 'Dhanyavadhalu'),
    ('జగన్ జీపీటీ', 'JaganGpt'),
    ('జగన్', 'Jagan'),
    ('జీపీటీ', 'GPT'),
    ('చిత్రం', 'Image'),
    ('బొమ్మ', 'Picture'),
    ('వీడియో', 'Video'),
    ('ఎలా ఉన్నారు', 'Ela unnaru'),
    ('ఎలా', 'Ela'),
    ('ఉన్నారు', 'Unnaru'),
    ('సహాయం', 'Sahayam'),
    ('చేయగలరు', 'Cheyagalaru'),
    ('మీరు', 'Meeru'),
    ('నేను', 'Nenu'),
]

def sanitize_visual_text(text: str) -> str:
    """
    CRITICAL RESTRICTION ENFORCEMENT:
    Ensures that NEVER does any Telugu Unicode character [\\u0C00-\\u0C7F]
    get returned to the user interface, chat messages, or visual displays.
    Translates or transliterates any Telugu Unicode text into clean Latin/English.
    """
    if not text:
        return text

    if not TELUGU_UNICODE_REGEX.search(text):
        return text

    # Step 1: Replace known common Telugu phrases with clear English/Tanglish
    sanitized = text
    for telugu_w, latin_w in COMMON_TELUGU_WORDS:
        sanitized = sanitized.replace(telugu_w, latin_w)

    # Step 2: Character-by-character transliteration for any remaining Telugu glyphs
    char_list = []
    for char in sanitized:
        if '\u0C00' <= char <= '\u0C7F':
            char_list.append(TELUGU_TO_LATIN_MAP.get(char, ''))
        else:
            char_list.append(char)
    sanitized = ''.join(char_list)

    # Step 3: Absolute guarantee - strip any residual Telugu Unicode
    sanitized = TELUGU_UNICODE_REGEX.sub('', sanitized)
    return sanitized

def get_system_prompt(preferred_language: str = 'auto') -> str:
    """
    Generates the master multilingual system prompt for JaganGpt.
    Enforces multilingual voice understanding with STRICT English-only visual UI.
    """
    return """You are **JaganGpt**, an advanced, friendly, and highly intelligent multimodal AI assistant.
Tagline: "Think. Create. Talk. Transform."
Created with passion to empower creators, developers, students, and professionals with world-class AI.

CRITICAL LANGUAGE & VISUAL TEXT RESTRICTION:
1. **Multilingual Understanding**:
   - You can understand user queries in all languages: English, Telugu, Hindi, Tamil, Kannada, Malayalam, etc.
   - You understand spoken Telugu and written Telugu naturally.

2. **VISUAL TEXT RESTRICTION (STRICT RULE)**:
   - NEVER display Telugu Unicode script in chat messages, UI text, labels, or generated visual output.
   - For all visible chat messages and text responses, use **ENGLISH ONLY** (or Latin-alphabet Tanglish if transliterating a phrase).
   - If the user writes or speaks in Telugu, understand their meaning completely and respond in clear, helpful **ENGLISH** in the chat display.
   - If the user asks for code, provide clean code in markdown blocks with English explanations.

3. **Capabilities**:
   - You analyze documents (PDF, Word, Excel, CSV, text, code, images).
   - High-fidelity identity-preserving image and video generation.
   - Clean formatting, bullet points, and accurate code.
   - Always be polite, professional, and helpful.
"""
