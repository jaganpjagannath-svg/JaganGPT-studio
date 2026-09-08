import os
import re
import json
import requests
from config import Config
from services.telugu_nlp import detect_language, get_system_prompt, sanitize_visual_text
from services.image_service import ImageService
from services.video_service import VideoService
import database

class AIRouter:
    """
    Central Multimodal AI Router.
    Dispatches to Google Gemini API when configured, with seamless open/free fallback,
    and automatic routing for Image and Video generation requests.
    """

    IMAGE_INTENT_REGEX = re.compile(
        r'\b(generate|create|make|draw|paint)\b.*?\b(image|picture|photo|illustration|logo|wallpaper|చిత్రం|బొమ్మ)\b|\b(image|photo|చిత్రం|బొమ్మ)\b.*?\b(generate|create|రూపొందించు|చేయి)\b',
        re.IGNORECASE
    )

    VIDEO_INTENT_REGEX = re.compile(
        r'\b(generate|create|make)\b.*?\b(video|clip|animation|cinematic|వీడియో)\b|\b(video|వీడియో)\b.*?\b(generate|create|రూపొందించు|చేయి)\b',
        re.IGNORECASE
    )

    @classmethod
    def get_effective_gemini_key(cls, user_id: int = None) -> str:
        """Retrieves Gemini API key from user settings, environment, or config."""
        if user_id:
            settings = database.query_db("SELECT gemini_api_key FROM user_settings WHERE user_id = ?", (user_id,), one=True)
            if settings and settings['gemini_api_key']:
                return settings['gemini_api_key'].strip()
        
        return Config.GEMINI_API_KEY.strip()

    @classmethod
    def process_query(cls, user_message: str, user_id: int, conversation_id: int, 
                      model_choice: str = 'jagangpt-smart', attachments: list = None,
                      is_guest: bool = False) -> dict:
        """
        Main query processing pipeline.
        Enforces: Guests can only use text (no images, no videos).
        """
        lang = detect_language(user_message)
        attachments = attachments or []

        # Enforce Guest Restrictions on Media Generation
        if is_guest and (cls.IMAGE_INTENT_REGEX.search(user_message) or cls.VIDEO_INTENT_REGEX.search(user_message)):
            reply_text = "🔒 **Guest Mode Restriction (Text Only)**\n\nGuest mode allows text conversations only. To create AI Images and Videos, please [Log In](/login) or [Create an Account](/register)!"

            return {
                'reply': sanitize_visual_text(reply_text),
                'media_type': 'text',
                'media_url': None,
                'language_detected': lang,
                'model_used': 'JaganGpt Access Controller'
            }

        # 1. Check for Image Generation Intent
        if cls.IMAGE_INTENT_REGEX.search(user_message):
            # Extract generation prompt
            clean_prompt = re.sub(r'^(please\s+|can\s+you\s+)?(generate|create|make|draw)\s+(an?\s+)?(image|photo|picture)\s+(of|showing)?\s*', '', user_message, flags=re.IGNORECASE).strip()
            if not clean_prompt:
                clean_prompt = user_message
            
            img_result = ImageService.generate_image(clean_prompt, user_id, conversation_id)
            reply_text = f"✨ **Image Generated with JaganGpt**\n\nPrompt: *\"{img_result['prompt']}\"*\n\n![Generated Image]({img_result['image_url']})\n\n[⬇️ Download Image]({img_result['image_url']})"

            return {
                'reply': sanitize_visual_text(reply_text),
                'media_type': 'image',
                'media_url': img_result['image_url'],
                'language_detected': lang,
                'model_used': 'JaganGpt Image Engine'
            }

        # 2. Check for Video Generation Intent
        if cls.VIDEO_INTENT_REGEX.search(user_message):
            clean_prompt = re.sub(r'^(please\s+|can\s+you\s+)?(generate|create|make)\s+(an?\s+)?(video|clip)\s+(of|showing)?\s*', '', user_message, flags=re.IGNORECASE).strip()
            if not clean_prompt:
                clean_prompt = user_message

            v_result = VideoService.create_video_job(clean_prompt, user_id, conversation_id)
            reply_text = f"🎬 **Video Generation Initiated**\n\nPrompt: *\"{clean_prompt}\"*\n\nJob ID: `{v_result['job_id']}`\n\nRendering cinematic video frames... Check back in a few seconds!"

            return {
                'reply': sanitize_visual_text(reply_text),
                'media_type': 'video',
                'media_url': None,
                'job_id': v_result['job_id'],
                'language_detected': lang,
                'model_used': 'JaganGpt Video Engine'
            }

        # 3. Assemble Multimodal Context from Attachments
        context_blocks = []
        for att in attachments:
            fname = att.get('original_name', 'File')
            extracted = att.get('extracted_text', '')
            if extracted:
                context_blocks.append(f"### [Attached Document: {fname}]\n```\n{extracted[:12000]}\n```")

        full_prompt = user_message
        if context_blocks:
            full_prompt = f"The user has uploaded the following document context:\n\n" + "\n\n".join(context_blocks) + f"\n\nUser Question/Request: {user_message}"

        # 4. Try Google Gemini API
        candidate_keys = []
        if user_id:
            user_set = database.query_db("SELECT gemini_api_key FROM user_settings WHERE user_id = ?", (user_id,), one=True)
            if user_set and user_set['gemini_api_key'] and user_set['gemini_api_key'].strip():
                candidate_keys.append(user_set['gemini_api_key'].strip())

        system_key = Config.GEMINI_API_KEY.strip() if Config.GEMINI_API_KEY else ''
        if system_key and system_key not in candidate_keys:
            candidate_keys.append(system_key)

        for gemini_key in candidate_keys:
            try:
                gemini_reply, used_model = cls._call_gemini_api(full_prompt, gemini_key, model_choice)
                if gemini_reply:
                    return {
                        'reply': sanitize_visual_text(gemini_reply),
                        'media_type': 'text',
                        'media_url': None,
                        'language_detected': lang,
                        'model_used': f'Google Gemini ({used_model})'
                    }
            except Exception as e:
                print(f"Gemini API error with key: {e}. Trying next key...")

        # 5. Free / Zero-Key Multimodal Fallback Router
        fallback_reply = cls._call_open_ai_router(full_prompt, model_choice, lang)
        return {
            'reply': sanitize_visual_text(fallback_reply),
            'media_type': 'text',
            'media_url': None,
            'language_detected': lang,
            'model_used': 'JaganGpt Multimodal Engine'
        }

    @classmethod
    def _call_gemini_api(cls, prompt: str, api_key: str, model_choice: str) -> tuple:
        """
        Calls Google Gemini API using official google.genai SDK.
        Dynamically selects the best available Gemini model (gemini-3.6-flash, gemini-flash-latest,
        gemini-3.7-flash, gemini-3.1-flash-lite, gemini-2.5-flash) with automated fallback on quota limits.
        """
        import truststore
        truststore.inject_into_ssl()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        system_instr = get_system_prompt()
        config = types.GenerateContentConfig(
            system_instruction=system_instr,
            temperature=0.7
        )

        candidate_models = [
            'gemini-3.6-flash',
            'gemini-flash-latest',
            'gemini-3.7-flash',
            'gemini-3.1-flash-lite',
            'gemini-2.5-flash'
        ]

        last_error = None
        for model in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                if resp and resp.text and resp.text.strip():
                    return resp.text.strip(), model
            except Exception as e:
                print(f"[AIRouter] Gemini model {model} error: {e}. Trying next candidate...")
                last_error = e

        raise Exception(f"All Gemini models exhausted: {last_error}")

    @classmethod
    def _call_open_ai_router(cls, prompt: str, model_choice: str, lang: str) -> str:
        """
        High-performance open AI router ensuring JaganGpt is 100% operational
        even before a user pastes their personal Gemini key.
        """
        system_instruction = get_system_prompt()
        
        # Pollinations text API or free open model endpoint
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            url = "https://text.pollinations.ai/"
            payload = {
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "model": "openai",
                "seed": 42
            }
            res = requests.post(url, json=payload, timeout=25, verify=False)
            if res.status_code == 200 and res.text.strip():
                return res.text
        except Exception as e:
            print(f"Open AI router error: {e}")

        # Built-in multilingual fallback response generator
        if lang == 'te':
            return f"Hello! I am **JaganGpt**. I received your query: \"{prompt[:100]}\".\n\nTo unlock full Google Gemini 2.5 Flash capabilities, please configure your **Gemini API Key** in **Settings**.\n\nI can assist you with coding, document analysis, and high-fidelity image & video generation!"
        elif lang == 'tanglish':
            return f"Namaskaram! Nenu **JaganGpt**. Meeru adigina query: \"{prompt[:100]}\".\n\nFull power Google Gemini models kosam **Settings** lo mee **Gemini API Key** save cheyyandi.\n\nNenu meeku code generation, document summarization, photos & videos create cheyyadam lo help chesthanu!"
        else:
            return f"Hello! I am **JaganGpt**, your multimodal AI assistant.\n\nI have received your request:\n> {prompt[:150]}...\n\n*Note: To unlock the full power of Google Gemini 2.5 Flash, add your **Gemini API Key** in the Settings modal!*"
