# JaganGpt 🤖✨

**JaganGpt** is a modern, production-ready multimodal AI assistant web application with native support for **Telugu + English + Tanglish**, real-time **Voice Mode**, **document & spreadsheet analysis**, **AI Image generation**, and **AI Video generation**.

> **Tagline**: *"Think. Create. Talk. Transform."*

---

## 🌟 Key Features

1. **Multimodal AI with Google Gemini**:
   - Powered by Google Gemini 2.5 Flash / Pro models.
   - Vision & image understanding, document comprehension, and complex reasoning.
   - Seamless fallback routing so JaganGpt is always available.

2. **Telugu + English Multilingual Intelligence**:
   - Automatic language detection (English ↔ తెలుగు ↔ Tanglish).
   - Proper Telugu Unicode script generation and natural conversational Tanglish (e.g., *"Python lo functions ela work avuthayo explain cheyyi"*).
   - Complete UI localization toggle (English 🇬🇧 ↔ తెలుగు 🇮🇳).

3. **Live Voice Mode**:
   - Dedicated full-screen voice conversation overlay with dynamic glowing voice orb.
   - Real-time speech recognition in Telugu (`te-IN`) and English (`en-US`).
   - Text-to-speech voice synthesis playing AI responses aloud.

4. **File & Document Intelligence**:
   - Drag & drop or attach files directly in conversations.
   - Supports: **PDF**, **Word (DOCX)**, **Excel (XLSX)**, **CSV**, **Code files**, and **Images**.
   - Ask JaganGpt to summarize documents, inspect data tables, or debug code.

5. **AI Image & Video Generation**:
   - **Image Gen**: Generates high-resolution images from natural prompts (including Telugu prompts).
   - **Video Gen**: Asynchronous video generation pipeline with job status tracking and preview player.

6. **ChatGPT-Style Modern UI**:
   - Dark navy/black glassmorphic interface with sidebar conversation history.
   - Code blocks with syntax highlighting and instant "Copy Code" button.
   - Thumbs up/down feedback ratings.
   - In-app Settings modal to configure your Google Gemini API key and preferences.

---

## 🚀 Quick Setup & Run Instructions

### Step 1: Open Terminal and Navigate to Project
```powershell
cd c:\Users\jagan\OneDrive\Desktop\JaganGPT
```

### Step 2: (Optional) Add your Google Gemini API Key
Get a free Gemini API key in 30 seconds from [Google AI Studio](https://aistudio.google.com).
Add it into `.env`:
```text
GEMINI_API_KEY=AIzaSy...your_key_here
```
*(You can also paste it directly inside the app's **Settings ⚙️** modal at any time!)*

### Step 3: Run the Application
```powershell
python app.py
```

### Step 4: Open in Browser
Visit:
```text
http://127.0.0.1:5001
```

---

## 📁 Project Architecture

```text
JaganGPT/
├── app.py                     # Main Flask application and API endpoints
├── config.py                  # App configuration and model settings
├── database.py                # SQLite schema (conversations, messages, media)
├── requirements.txt           # Python dependencies
├── .env.example               # API keys template
│
├── services/
│   ├── ai_router.py           # Multimodal AI Router & Google Gemini dispatcher
│   ├── telugu_nlp.py          # Telugu script & Tanglish detection and normalizer
│   ├── file_analyzer.py       # PDF, Word, Excel, CSV, and code parser
│   ├── image_service.py       # AI Image generation service
│   ├── video_service.py       # Asynchronous AI Video generation service
│   └── voice_service.py       # Multilingual Text-to-Speech service
│
├── static/
│   ├── css/style.css          # Dark glassmorphic responsive design
│   ├── js/
│   │   ├── chat.js            # Markdown, code blocks, copy, attachments
│   │   ├── voice.js           # Live Voice Mode visualizer orb and speech
│   │   ├── media_gen.js       # Image and video modals
│   │   └── i18n.js            # English ↔ Telugu localization dictionary
│   └── images/logo.svg        # JaganGpt spark vector emblem
│
└── templates/
    ├── index.html             # Main ChatGPT-style multimodal workspace
    ├── login.html             # Login screen
    └── register.html          # Signup screen
```
