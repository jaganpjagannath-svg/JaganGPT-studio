"""
JaganGpt - Prompt Disentanglement & Scene Conditioning Engine
Separates user instructions into:
  IDENTITY        -> Reference person's physical facial identity
  CLOTHING        -> Desired garments (e.g. black shirt, black pants)
  ACCESSORIES     -> Wearables (e.g. black sunglasses, watch, hat)
  BODY            -> Physical build (e.g. muscular, athletic)
  POSE            -> Body posture (e.g. confident hero pose, standing)
  BACKGROUND      -> Scene backdrop (e.g. pure black background)
  LIGHTING        -> Illumination (e.g. dramatic movie lighting, rim lights)
  CAMERA          -> Framing (e.g. full-body portrait, 35mm lens)
  ENVIRONMENT     -> Location setting
  ACTION          -> Dynamic behavior
  STYLE           -> Aesthetic (e.g. photorealistic movie photography)
  EXPRESSION      -> Facial mood
"""

import os
import re
import json
import requests
from config import Config

class PromptParser:
    """
    Disentangles prompt instructions to ensure the model reconstructs the
    real reference person in the new scene without copying original clothes/background.
    """

    @classmethod
    def parse_prompt(cls, raw_prompt: str, user_id: int = None) -> dict:
        """
        Parses raw user prompt into 12 structured condition categories.
        Uses Google Gemini 2.5 Flash when available, with rule-based NLP fallback.
        """
        cleaned = raw_prompt.strip()

        # 1. Try Gemini 2.5 Flash for deep semantic disentanglement
        gemini_key = None
        if user_id:
            try:
                import database
                settings = database.query_db(
                    "SELECT gemini_api_key FROM user_settings WHERE user_id = ?",
                    (user_id,), one=True
                )
                if settings and settings['gemini_api_key']:
                    gemini_key = settings['gemini_api_key'].strip()
            except Exception:
                pass

        if not gemini_key:
            gemini_key = Config.GEMINI_API_KEY.strip()

        if gemini_key:
            try:
                parsed = cls._parse_with_gemini(cleaned, gemini_key)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"[PromptParser] Gemini NLP parse failed, using rule-based extractor: {e}")

        # 2. Rule-based NLP extraction fallback
        return cls._parse_with_rules(cleaned)

    @classmethod
    def _parse_with_gemini(cls, prompt: str, api_key: str) -> dict:
        """Calls Gemini 2.5 Flash via REST with unverified SSL to parse prompt into JSON."""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        system_instruction = (
            "You are an expert AI Media Prompt Disentanglement Engine. "
            "Your task is to analyze user prompts for AI image/video generation and extract "
            "exact visual attributes into a clean JSON structure.\n"
            "Separate WHO the person is (identity) from WHAT the person looks like / does in the scene.\n"
            "Return ONLY valid JSON with keys:\n"
            "{\n"
            "  \"clothing\": \"exact garments requested (e.g. premium black shirt, black pants)\",\n"
            "  \"accessories\": \"wearables (e.g. black sunglasses)\",\n"
            "  \"body\": \"build requested (e.g. natural muscular athletic appearance)\",\n"
            "  \"pose\": \"posture (e.g. confident hero pose, standing upright)\",\n"
            "  \"background\": \"backdrop (e.g. pure solid black background)\",\n"
            "  \"lighting\": \"illumination (e.g. dramatic cinematic movie lighting, rim lighting)\",\n"
            "  \"camera\": \"framing (e.g. full-body portrait shot, 35mm lens)\",\n"
            "  \"style\": \"aesthetic (e.g. photorealistic movie photography)\",\n"
            "  \"expression\": \"mood (e.g. confident hero expression)\",\n"
            "  \"has_sunglasses\": true/false,\n"
            "  \"has_dark_clothing\": true/false,\n"
            "  \"has_dark_background\": true/false\n"
            "}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": f"User Prompt: {prompt}"}]}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        resp = requests.post(url, json=payload, timeout=8, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get('candidates', [])
            if candidates:
                raw_text = candidates[0]['content']['parts'][0]['text']
                parsed = json.loads(raw_text)
                parsed['raw_prompt'] = prompt
                parsed['identity'] = 'Reference person identity'
                return parsed
        return None

    @classmethod
    def _parse_with_rules(cls, prompt: str) -> dict:
        """Robust rule-based regex NLP parser for clothing, accessories, pose, and background."""
        p_lower = prompt.lower()

        # Clothing detection
        clothing_items = []
        if re.search(r'\bblack\s+shirt\b|\bblack\s+t-?shirt\b|\bblack\s+button-?down\b', p_lower):
            clothing_items.append("premium black shirt")
        elif re.search(r'\b(white|blue|red|dark)\s+shirt\b', p_lower):
            m = re.search(r'\b(white|blue|red|dark)\s+shirt\b', p_lower)
            clothing_items.append(m.group(0))
        elif 'shirt' in p_lower:
            clothing_items.append("stylish shirt")

        if re.search(r'\bblack\s+pants\b|\bblack\s+trousers\b|\bblack\s+jeans\b', p_lower):
            clothing_items.append("black pants")
        elif re.search(r'\b(pants|trousers|jeans)\b', p_lower):
            m = re.search(r'\b(\w+)\s+(pants|trousers|jeans)\b', p_lower)
            clothing_items.append(m.group(0) if m else "pants")

        if 'jacket' in p_lower:
            m = re.search(r'\b(\w+\s+)?jacket\b', p_lower)
            if m: clothing_items.append(m.group(0))

        if 'suit' in p_lower or 'tuxedo' in p_lower:
            clothing_items.append("tailored black suit")

        clothing_str = ", ".join(clothing_items) if clothing_items else "stylish modern outfit"

        # Accessories detection
        accessories_items = []
        has_sunglasses = bool(re.search(r'\b(sunglasses|sunglass|shades|black\s+glasses|dark\s+glasses)\b', p_lower))
        if has_sunglasses:
            accessories_items.append("black sunglasses")

        if 'watch' in p_lower:
            accessories_items.append("luxury wristwatch")
        if 'chain' in p_lower or 'necklace' in p_lower:
            accessories_items.append("subtle metallic chain")
        if 'hat' in p_lower or 'cap' in p_lower:
            accessories_items.append("stylish hat")

        accessories_str = ", ".join(accessories_items) if accessories_items else "none"

        # Pose detection
        if re.search(r'\bhero\s+pose\b|\bheroic\s+pose\b|\bmovie\s+hero\b', p_lower):
            pose_str = "confident movie hero pose standing firmly"
        elif 'standing' in p_lower:
            pose_str = "standing confidently looking forward"
        elif 'walking' in p_lower:
            pose_str = "walking with cinematic purpose toward camera"
        elif 'sitting' in p_lower:
            pose_str = "seated with confident posture"
        else:
            pose_str = "confident natural posture"

        # Body detection
        if re.search(r'\bmuscular\b|\bathletic\b|\btoned\b|\bfit\b', p_lower):
            body_str = "natural muscular athletic physique"
        else:
            body_str = "natural proportional build"

        # Background detection
        has_dark_bg = bool(re.search(r'\b(black\s+background|pure\s+black|dark\s+background|dark\s+studio|black\s+backdrop|darkness)\b', p_lower))
        if has_dark_bg:
            background_str = "pure solid black studio background, clean infinite darkness"
        elif 'studio' in p_lower:
            background_str = "professional studio backdrop"
        elif 'city' in p_lower or 'street' in p_lower:
            background_str = "cinematic city street background"
        else:
            background_str = "cinematic atmospheric background"

        # Lighting detection
        if re.search(r'\b(dramatic|movie|cinematic|rim|low\s+key)\s+lighting\b', p_lower):
            lighting_str = "dramatic cinematic movie lighting with sharp rim highlights and deep contrast"
        elif 'studio lighting' in p_lower:
            lighting_str = "professional three-point studio lighting"
        else:
            lighting_str = "cinematic atmospheric lighting"

        # Camera framing
        if 'full-body' in p_lower or 'full body' in p_lower:
            camera_str = "full-body portrait composition, eye-level, 35mm lens"
        elif 'portrait' in p_lower:
            camera_str = "medium portrait shot, 50mm f/1.8 lens, shallow depth of field"
        else:
            camera_str = "cinematic composition, sharp focus"

        has_dark_clothing = bool(re.search(r'\bblack\b', clothing_str.lower()))

        return {
            'identity': 'Reference person identity',
            'clothing': clothing_str,
            'accessories': accessories_str,
            'body': body_str,
            'pose': pose_str,
            'background': background_str,
            'lighting': lighting_str,
            'camera': camera_str,
            'style': "photorealistic movie photography, 35mm film still, 8k resolution, razor-sharp detail",
            'expression': "intense movie hero expression",
            'has_sunglasses': has_sunglasses,
            'has_dark_clothing': has_dark_clothing,
            'has_dark_background': has_dark_bg,
            'raw_prompt': prompt
        }

    @classmethod
    def build_synthesis_prompt(cls, parsed: dict, ref_profile: dict = None, prompt_strength: float = 0.85) -> str:
        """
        Assembles a conditioning prompt for the diffusion scene generator.
        Forces generation of the new clothes, accessories, and background while
        matching the subject's demographic features (skin tone, structure).
        """
        # Demographic anchor derived from reference face profile
        skin_anchor = ""
        if ref_profile and 'skin_tone_lab' in ref_profile:
            l, a, b = ref_profile['skin_tone_lab']
            if l < 110:
                skin_anchor = "deep warm complexion"
            elif l < 155:
                skin_anchor = "medium warm olive-tan complexion"
            else:
                skin_anchor = "fair natural complexion"

        body_val = parsed.get('body') or "natural proportional athletic build"
        clothing_val = parsed.get('clothing') or "stylish black outfit"
        acc_val = parsed.get('accessories')
        pose_val = parsed.get('pose') or "confident cinematic posture"
        bg_val = parsed.get('background') or "dramatic black studio background"
        light_val = parsed.get('lighting') or "cinematic movie lighting with rim highlights"
        cam_val = parsed.get('camera') or "full-body cinematic framing, 35mm lens"

        parts = [
            f"Masterpiece photorealistic movie photograph",
            f"Subject: a person with {skin_anchor if skin_anchor else 'natural realistic skin tone'}, {body_val}",
            f"Wearing: {clothing_val}",
            f"Accessories: {acc_val}" if (acc_val and acc_val.lower() != 'none') else "",
            f"Pose & Stance: {pose_val}",
            f"Background: {bg_val}",
            f"Lighting: {light_val}",
            f"Camera & Framing: {cam_val}",
            f"Details: natural skin pores, realistic cloth fabric texture, razor-sharp focus, high micro-contrast, volumetric depth, 8k"
        ]

        # Filter empty parts and join
        return ", ".join([p for p in parts if p])

    @classmethod
    def build_negative_prompt(cls, parsed: dict, custom_negative: str = None) -> str:
        """
        Constructs a targeted negative prompt preventing the model from generating
        the original background, old clothes, or visual artifacts.
        """
        negatives = [
            "distorted face", "bad anatomy", "deformed eyes", "extra limbs",
            "plastic skin", "wax skin", "over-smoothed skin", "airbrushed",
            "blurry", "cartoon", "illustration", "3d render", "low resolution",
            "bad hands", "missing fingers", "distorted clothing"
        ]

        # Background exclusion
        if parsed.get('has_dark_background'):
            negatives.extend(["white background", "bright background", "outdoor", "daylight", "trees", "room", "windows"])

        # Clothing exclusion
        if parsed.get('has_dark_clothing'):
            negatives.extend(["white shirt", "bright shirt", "colorful clothes", "light clothing"])

        # Sunglasses exclusion
        if parsed.get('has_sunglasses'):
            negatives.extend(["bare eyes without sunglasses", "clear spectacles", "no sunglasses"])

        if custom_negative:
            negatives.insert(0, custom_negative.strip())

        return ", ".join(negatives)
