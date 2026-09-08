/**
 * JaganGpt - UI Localization & Language Configuration Engine
 * Enforces English-only visual UI while enabling multilingual voice conversations.
 */

const I18N_DATA = {
    en: {
        app_name: "JaganGpt",
        tagline: "Think. Create. Talk. Transform.",
        new_chat: "New Chat",
        search_chats: "Search chats...",
        recent_chats: "Recent Chats",
        settings: "Settings",
        logout: "Log Out",
        voice_mode: "Voice Mode",
        model_smart: "JaganGpt Smart",
        model_fast: "JaganGpt Fast",
        model_vision: "JaganGpt Vision",
        model_voice: "JaganGpt Voice",
        welcome_title: "How can JaganGpt help you today?",
        welcome_sub: "Ask a question, upload a document, dictate in voice, or generate images & videos.",
        card_1_title: "Document Analysis",
        card_1_desc: "Upload a PDF, Excel sheet, or Word doc and ask for summaries or insights.",
        card_2_title: "AI Image Generation",
        card_2_desc: "Describe what you want to create or condition on your reference photo.",
        card_3_title: "Voice Conversation",
        card_3_desc: "Speak naturally in Telugu or English and listen to real-time AI voice.",
        card_4_title: "Code & Logic",
        card_4_desc: "Debug Python, write modern JavaScript, or solve algorithmic problems.",
        input_placeholder: "Message JaganGpt... (English, Tanglish, or Voice)",
        attach_tooltip: "Attach Document, Image, or Code",
        image_gen_btn: "Generate Image",
        video_gen_btn: "Generate Video",
        mic_tooltip: "Speak in English, Telugu, or other languages",
        send_tooltip: "Send message",
        settings_title: "JaganGpt Settings",
        gemini_api_key_label: "Google Gemini API Key",
        gemini_api_key_help: "Get your free key from Google AI Studio. Enables Gemini 2.5 Flash.",
        language_pref_label: "Language Preference",
        lang_auto: "Auto Detect (Multilingual Voice / English UI)",
        lang_en: "English",
        lang_te: "Telugu (Voice Mode)",
        save_settings: "Save Settings",
        voice_listening: "Listening to you...",
        voice_thinking: "JaganGpt is thinking...",
        voice_speaking: "JaganGpt is speaking...",
        voice_end: "End Voice Session"
    },
    te: {
        app_name: "JaganGpt",
        tagline: "Think. Create. Talk. Transform.",
        new_chat: "New Chat",
        search_chats: "Search chats...",
        recent_chats: "Recent Chats",
        settings: "Settings",
        logout: "Log Out",
        voice_mode: "Voice Mode",
        model_smart: "JaganGpt Smart",
        model_fast: "JaganGpt Fast",
        model_vision: "JaganGpt Vision",
        model_voice: "JaganGpt Voice",
        welcome_title: "How can JaganGpt help you today?",
        welcome_sub: "Ask a question, upload a document, dictate in voice, or generate images & videos.",
        card_1_title: "Document Analysis",
        card_1_desc: "Upload a PDF, Excel sheet, or Word doc and ask for summaries or insights.",
        card_2_title: "AI Image Generation",
        card_2_desc: "Describe what you want to create or condition on your reference photo.",
        card_3_title: "Voice Conversation",
        card_3_desc: "Speak naturally in Telugu or English and listen to real-time AI voice.",
        card_4_title: "Code & Logic",
        card_4_desc: "Debug Python, write modern JavaScript, or solve algorithmic problems.",
        input_placeholder: "Message JaganGpt... (English, Tanglish, or Voice)",
        attach_tooltip: "Attach Document, Image, or Code",
        image_gen_btn: "Generate Image",
        video_gen_btn: "Generate Video",
        mic_tooltip: "Speak in English, Telugu, or other languages",
        send_tooltip: "Send message",
        settings_title: "JaganGpt Settings",
        gemini_api_key_label: "Google Gemini API Key",
        gemini_api_key_help: "Get your free key from Google AI Studio. Enables Gemini 2.5 Flash.",
        language_pref_label: "Language Preference",
        lang_auto: "Auto Detect (Multilingual Voice / English UI)",
        lang_en: "English",
        lang_te: "Telugu (Voice Mode)",
        save_settings: "Save Settings",
        voice_listening: "Listening to you (Telugu)...",
        voice_thinking: "JaganGpt is thinking...",
        voice_speaking: "JaganGpt is speaking...",
        voice_end: "End Voice Session"
    }
};

let currentLanguage = localStorage.getItem('jagangpt_lang') || 'en';

function setLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('jagangpt_lang', lang);

    const dict = I18N_DATA[lang] || I18N_DATA['en'];

    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key]) {
            el.textContent = dict[key];
        }
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (dict[key]) {
            el.placeholder = dict[key];
        }
    });

    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        if (dict[key]) {
            el.title = dict[key];
        }
    });

    // Update active button state
    const enBtn = document.getElementById('lang-btn-en');
    const teBtn = document.getElementById('lang-btn-te');
    if (enBtn && teBtn) {
        if (lang === 'te') {
            teBtn.classList.add('active');
            enBtn.classList.remove('active');
        } else {
            enBtn.classList.add('active');
            teBtn.classList.remove('active');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setLanguage(currentLanguage);
});
