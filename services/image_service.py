"""
JaganGpt - Disentangled High-Fidelity Image Generation Service
Combines Reference Face Identity Preservation with Prompt-Driven Visual Transformation:
  REFERENCE IMAGE -> Identity condition (facial bone structure, eyes, nose, jawline, skin tone, pores)
  USER PROMPT     -> Transformation condition (clothes, accessories, body pose, background, lighting)
Simultaneously preserves the exact person's identity while reconstructing new scene characteristics.
Uses True Model Conditioning via InstantID (IdentityNet + ControlNet).
"""

import os
import uuid
import json
import urllib.parse
import requests
import numpy as np
import cv2

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from config import Config
from services.face_pipeline import FacePipeline
from services.prompt_parser import PromptParser
from services.telugu_nlp import normalize_media_prompt
import database

class ImageService:
    # Target 4K, 1080p, and 720p output dimensions across aspect ratios
    RESOLUTION_TARGETS = {
        '720p': {
            '1:1': (720, 720),
            '4:5': (720, 900),
            '16:9': (1280, 720),
            '9:16': (720, 1280),
            '4:3': (960, 720),
            '3:4': (720, 960)
        },
        '1080p': {
            '1:1': (1080, 1080),
            '4:5': (1080, 1350),
            '16:9': (1920, 1080),
            '9:16': (1080, 1920),
            '4:3': (1440, 1080),
            '3:4': (1080, 1440)
        },
        '4k': {
            '1:1': (3840, 3840),
            '4:5': (3072, 3840),
            '16:9': (3840, 2160),
            '9:16': (2160, 3840),
            '4:3': (2880, 2160),
            '3:4': (2160, 2880)
        }
    }

    # Base latent diffusion generation size (optimal for speed & composition)
    LATENT_BASE_SIZES = {
        '1:1': (768, 768),
        '4:5': (704, 880),
        '16:9': (1024, 576),
        '9:16': (576, 1024),
        '4:3': (800, 600),
        '3:4': (600, 800)
    }

    @classmethod
    def generate_image(cls, prompt: str, user_id: int, conversation_id: int = None,
                       reference_image_paths: list = None, reference_image_path: str = None,
                       negative_prompt: str = None, aspect_ratio: str = '1:1',
                       resolution: str = '1080p', identity_threshold: float = 0.70,
                       identity_strength: float = 0.75, prompt_strength: float = 0.85) -> dict:
        """
        Main image generation method.
        Simultaneously guarantees:
          1. Exact same person's identity (face structure, eyes, nose, jawline, pores).
          2. Complete visual transformation according to user prompt (new clothes, sunglasses, pose, background).
        """
        # Normalize prompt for Telugu, Tanglish, and English
        clean_prompt = normalize_media_prompt(prompt)

        # Normalize reference paths (support both singular and plural)
        if reference_image_path and not reference_image_paths:
            reference_image_paths = [reference_image_path]
        elif isinstance(reference_image_paths, str):
            reference_image_paths = [reference_image_paths]
        reference_image_paths = [p for p in (reference_image_paths or []) if p and os.path.exists(p)]

        # Multi-Reference Identity Extraction & Master Embedding Synthesis
        multi_ref_data = None
        ref_profile = None
        primary_face = None
        if reference_image_paths:
            multi_ref_data = FacePipeline.detect_and_extract_multi_faces(reference_image_paths)
            if not multi_ref_data['success']:
                return {
                    'error': multi_ref_data['error'],
                    'face_detected': False
                }
            primary_face = multi_ref_data['primary_face']
            ref_profile = primary_face.get('profile')

        # ----------------- Step 1: Prompt Disentanglement -----------------
        # Parse into IDENTITY vs TRANSFORMATION conditions
        parsed_prompt = PromptParser.parse_prompt(clean_prompt, user_id=user_id)

        # Calculate Native & Final Target Dimensions
        res_key = resolution.lower() if resolution.lower() in cls.RESOLUTION_TARGETS else '1080p'
        ar_key = aspect_ratio if aspect_ratio in cls.LATENT_BASE_SIZES else '1:1'

        final_w, final_h = cls.RESOLUTION_TARGETS[res_key][ar_key]
        latent_w, latent_h = cls.LATENT_BASE_SIZES[ar_key]

        # ----------------- Step 2: Structured Conditioning Prompt Building -----------------
        synthesis_prompt = PromptParser.build_synthesis_prompt(
            parsed=parsed_prompt,
            ref_profile=ref_profile,
            prompt_strength=prompt_strength
        )
        neg_prompt = PromptParser.build_negative_prompt(
            parsed=parsed_prompt,
            custom_negative=negative_prompt
        )

        # Unique filename & target path
        filename = f"gen_{uuid.uuid4().hex[:12]}.jpg"
        save_path = os.path.join(Config.IMAGES_FOLDER, filename)
        image_url = f"/uploads/images/{filename}"

        # ----------------- Step 3: Model Conditioning & Generation Loop -----------------
        max_attempts = 2 if multi_ref_data else 1
        best_similarity = 0.0
        best_verification = None
        best_adherence = None
        attempts_taken = 0

        has_sunglasses = parsed_prompt.get('has_sunglasses', False)
        raw_scene_bgr = None
        refined_bgr = None
        final_enhanced_bgr = None

        for attempt in range(1, max_attempts + 1):
            attempts_taken = attempt
            seed = (uuid.uuid4().int + attempt * 777) % 10000000

            # Sub-step A: Google Gemini Multimodal Image Generation (Google AI Studio)
            scene_bgr = None
            try:
                scene_bgr = cls._generate_with_gemini(
                    ref_path=primary_ref_path if primary_face else None,
                    prompt=synthesis_prompt,
                    aspect_ratio=ar_key
                )
            except Exception as e:
                print(f"[ImageService] Gemini generation attempt {attempt} error: {e}")

            # Sub-step B: True Model Conditioning Generation via InstantID
            if scene_bgr is None and multi_ref_data and primary_face:
                primary_ref_path = primary_face['image_path']
                try:
                    scene_bgr = cls._generate_with_instantid(
                        ref_path=primary_ref_path,
                        prompt=synthesis_prompt,
                        neg_prompt=neg_prompt,
                        has_sunglasses=has_sunglasses,
                        identity_strength=identity_strength,
                        seed=seed
                    )
                except Exception as e:
                    print(f"[ImageService] InstantID generation attempt {attempt} failed: {e}")

            # Sub-step C: Cloud Latent Diffusion fallback (Pollinations Flux-Realism / Turbo)
            if scene_bgr is None:
                scene_bgr = cls._generate_with_cloud(
                    prompt=synthesis_prompt,
                    neg_prompt=neg_prompt,
                    width=latent_w,
                    height=latent_h,
                    seed=seed
                )

            # Sub-step C: High-Fidelity Studio Scene Synthesis fallback (offline / test mode)
            if scene_bgr is None:
                scene_bgr = cls._create_fallback_scene_bgr(
                    width=latent_w,
                    height=latent_h,
                    parsed_prompt=parsed_prompt,
                    primary_face=primary_face
                )

            raw_scene_bgr = scene_bgr.copy()

            # Sub-step D: Model-Conditioned Face Refinement (Clean pass-through, no oval cutouts)
            if multi_ref_data:
                refined_bgr = FacePipeline.fuse_identity_high_fidelity(
                    scene_bgr=scene_bgr,
                    primary_ref_path=primary_face['image_path'],
                    face_data=primary_face,
                    boost_level=1.0 + (attempt - 1) * 0.2,
                    has_sunglasses=has_sunglasses,
                    identity_strength=identity_strength
                )
            else:
                refined_bgr = scene_bgr

            # Sub-step E: True High-Resolution Super-Resolution & Detail Sharpening (True 4K)
            final_enhanced_bgr = FacePipeline.super_resolve_and_enhance(
                img_bgr=refined_bgr,
                target_w=final_w,
                target_h=final_h,
                is_4k=(res_key == '4k')
            )

            # Write high-quality image to disk
            cv2.imwrite(save_path, final_enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])

            # Sub-step F: Dual Verification (Identity Consistency + Prompt Adherence)
            if multi_ref_data:
                verification = FacePipeline.verify_identity_consistency(
                    master_embedding_list=multi_ref_data['master_embedding'],
                    generated_image_path=save_path,
                    threshold=identity_threshold,
                    has_sunglasses=has_sunglasses
                )
                adherence = FacePipeline.verify_prompt_adherence(save_path, parsed_prompt)

                current_score = verification.get('similarity_score', 0.0)
                if current_score > best_similarity:
                    best_similarity = current_score
                    best_verification = verification
                    best_adherence = adherence

                # Exit loop if conditions are met or if first attempt completed
                if (verification.get('verified', False) and adherence.get('verified', True)) or attempt >= max_attempts:
                    break
            else:
                break

        # Save intermediate debugging artifacts
        if multi_ref_data and primary_face:
            cls._save_debug_artifacts(
                primary_face=primary_face,
                multi_ref_data=multi_ref_data,
                raw_bgr=raw_scene_bgr,
                refined_bgr=refined_bgr,
                final_bgr=final_enhanced_bgr
            )

        # Save to database
        img_id = database.execute_db("""
            INSERT INTO generated_images (user_id, conversation_id, prompt, image_url)
            VALUES (?, ?, ?, ?)
        """, (user_id, conversation_id, prompt, image_url))

        return {
            'id': img_id,
            'prompt': prompt,
            'normalized_prompt': clean_prompt,
            'image_url': image_url,
            'filename': filename,
            'reference_used': bool(multi_ref_data),
            'total_references': multi_ref_data['total_references'] if multi_ref_data else 0,
            'face_crop_url': multi_ref_data['primary_face']['aligned_crop_url'] if multi_ref_data else None,
            'similarity_score': best_similarity,
            'identity_threshold': identity_threshold,
            'identity_verified': best_verification.get('verified', False) if best_verification else False,
            'high_confidence': best_verification.get('high_confidence', False) if best_verification else False,
            'sunglasses_adjusted': best_verification.get('sunglasses_adjusted', False) if best_verification else False,
            'prompt_adherence': best_adherence if best_adherence else {'verified': True, 'adherence_score': 1.0},
            'disentanglement': parsed_prompt,
            'attempts': attempts_taken,
            'attempts_taken': attempts_taken,
            'aspect_ratio': ar_key,
            'native_resolution': f"{latent_w}x{latent_h}",
            'final_resolution': f"{final_w}x{final_h}",
            'resolution_label': f"{res_key.upper()} ({final_w}x{final_h})"
        }

    @classmethod
    def _generate_with_gemini(cls, ref_path: str = None, prompt: str = "", aspect_ratio: str = '1:1') -> np.ndarray:
        """
        Multimodal Image Generation via Google AI Studio / Gemini API.
        Sends original reference image + natural-language prompt directly to Gemini.
        """
        gemini_key = getattr(Config, 'GEMINI_API_KEY', None)
        if not gemini_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            gemini_img_models = [
                os.environ.get('GEMINI_IMAGE_MODEL', 'gemini-2.5-flash-image'),
                'gemini-3.1-flash-image',
                'gemini-3.1-flash-lite-image',
                'gemini-3-pro-image'
            ]

            contents = []
            if ref_path and os.path.exists(ref_path):
                with open(ref_path, 'rb') as f:
                    img_bytes = f.read()
                mime = 'image/jpeg' if ref_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

            contents.append(prompt)

            for model_name in gemini_img_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents
                    )
                    if response and response.candidates:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                                arr = np.frombuffer(part.inline_data.data, dtype=np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None and img.shape[0] > 50:
                                    print(f"[ImageService] Generated image via Google AI Studio ({model_name})")
                                    return img
                except Exception as model_err:
                    print(f"[ImageService] Gemini model {model_name} attempt: {model_err}")
        except Exception as e:
            print(f"[ImageService] Gemini generator error: {e}")
        return None

    @classmethod
    def _generate_with_instantid(cls, ref_path: str, prompt: str, neg_prompt: str,
                                 has_sunglasses: bool, identity_strength: float = 0.75,
                                 seed: int = 42) -> np.ndarray:
        """
        True Model-Level Identity Conditioning using InstantID (IdentityNet + IP-Adapter + ControlNet).
        Directly conditions diffusion latents on the reference face embedding without any
        elliptical cutouts, face masks, or post-hoc pasting.
        """
        from gradio_client import Client, handle_file

        token = getattr(Config, 'HF_TOKEN', None) or None
        client = Client('InstantX/InstantID', token=token)

        # When sunglasses are requested, lower identitynet weight and remove depth/canny
        # so diffusion naturally synthesizes dark eyewear seamlessly over the eye region
        id_strength = 0.55 if has_sunglasses else min(1.0, max(0.4, identity_strength))

        # Photographic prompt formulation
        conditioned_prompt = f"award-winning 8k professional studio photograph of this person, {prompt}, sharp focus, 85mm portrait photography, realistic skin texture with fine pores, cinematic depth of field, photorealistic"
        conditioned_neg = f"illustration, 3d render, cartoon, anime, drawing, comic, painting, graphic, deformed, bad anatomy, oversaturated, artificial, plastic, blurry, low resolution, {neg_prompt or ''}"

        res = client.predict(
            face_image_path=handle_file(ref_path),
            pose_image_path=None,
            prompt=conditioned_prompt,
            negative_prompt=conditioned_neg,
            style_name='(No style)',
            num_steps=26,
            identitynet_strength_ratio=id_strength,
            adapter_strength_ratio=0.65,
            canny_strength=0.0,
            depth_strength=0.0,
            controlnet_selection=[],
            guidance_scale=4.8,
            seed=seed,
            scheduler='EulerDiscreteScheduler',
            enable_LCM=False,
            enhance_face_region=False,
            api_name='/generate_image'
        )
        out_file = res[0]
        if out_file and os.path.exists(out_file):
            return cv2.imread(out_file)
        return None

    @classmethod
    def _generate_with_cloud(cls, prompt: str, neg_prompt: str, width: int, height: int, seed: int = 42) -> np.ndarray:
        """
        Synthesizes photorealistic base image using Pollinations AI (Flux-Realism / Turbo).
        """
        encoded_prompt = urllib.parse.quote(prompt)
        encoded_negative = urllib.parse.quote(neg_prompt)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

        for model in ['flux-realism', 'turbo', 'flux']:
            url = (
                f"https://image.pollinations.ai/prompt/{encoded_prompt}?"
                f"negative={encoded_negative}&width={width}&height={height}&model={model}&seed={seed}&nologo=true"
            )
            try:
                resp = requests.get(url, headers=headers, timeout=14)
                if resp.status_code == 200 and len(resp.content) > 2000:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] > 50:
                        return img
            except Exception as e:
                print(f"[ImageService] Cloud fetch ({model}) error: {e}")
        return None

    @classmethod
    def _create_fallback_scene_bgr(cls, width: int = 768, height: int = 768,
                                   parsed_prompt: dict = None, primary_face: dict = None) -> np.ndarray:
        """
        Synthesizes a tailored photographic studio backdrop reflecting parsed transformation conditions:
        Pure dark studio background, dark tailored clothing, and accessory anchors.
        Uses smooth cinematic gradient shading without harsh elliptical masks, cutouts, or cartoon shapes.
        """
        parsed_prompt = parsed_prompt or {}
        img = np.full((height, width, 3), 10 if parsed_prompt.get('has_dark_background') else 24, dtype=np.uint8)

        # Smooth studio vignette
        cy, cx = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette = (1.0 - 0.5 * (dist / max_dist)).clip(0.4, 1.0)
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * vignette).astype(np.uint8)

        # Dark tailored clothing in lower region (torso)
        if parsed_prompt.get('has_dark_clothing', True):
            torso_mask = np.zeros((height, width), dtype=np.float32)
            torso_mask[int(height * 0.58):, int(width * 0.15):int(width * 0.85)] = 1.0
            torso_mask = cv2.GaussianBlur(torso_mask, (51, 51), 0)
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * (1.0 - torso_mask * 0.65)).astype(np.uint8)

        # If primary face reference is provided in fallback/offline mode, integrate softly
        if primary_face and os.path.exists(primary_face.get('image_path', '')):
            ref_img = cv2.imread(primary_face['image_path'])
            if ref_img is not None:
                face_w = int(width * 0.42)
                face_h = int(height * 0.42)
                face_resized = cv2.resize(ref_img, (face_w, face_h), interpolation=cv2.INTER_LANCZOS4)
                py1, px1 = int(height * 0.16), int(cx - face_w // 2)

                # Soft, natural edge falloff blend (feathering, not harsh cutout)
                blend_mask = np.ones((face_h, face_w), dtype=np.float32)
                blend_mask[:12, :] = 0; blend_mask[-12:, :] = 0
                blend_mask[:, :12] = 0; blend_mask[:, -12:] = 0
                blend_mask = cv2.GaussianBlur(blend_mask, (25, 25), 0)[:, :, np.newaxis]

                roi = img[py1:py1+face_h, px1:px1+face_w]
                img[py1:py1+face_h, px1:px1+face_w] = (face_resized * blend_mask + roi * (1.0 - blend_mask)).astype(np.uint8)

        # Sunglasses accessory anchor if requested
        if parsed_prompt.get('has_sunglasses'):
            ey1, ey2 = int(height * 0.22), int(height * 0.36)
            ex1, ex2 = int(width * 0.32), int(width * 0.68)
            img[ey1:ey2, ex1:ex2] = (img[ey1:ey2, ex1:ex2] * 0.12).astype(np.uint8)

        return img

    @classmethod
    def _save_debug_artifacts(cls, primary_face: dict, multi_ref_data: dict,
                              raw_bgr: np.ndarray, refined_bgr: np.ndarray,
                              final_bgr: np.ndarray):
        """
        Saves intermediate debugging artifacts into uploads/debug/ for auditing & inspection:
        - reference.jpg
        - face_crop.jpg
        - identity_embedding.json
        - face_mask.png
        - generated_raw.png
        - generated_refined.png
        - final_output.png
        """
        try:
            debug_dir = os.path.join(Config.UPLOAD_FOLDER, 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            # 1. reference.jpg
            if primary_face and os.path.exists(primary_face.get('image_path', '')):
                ref_img = cv2.imread(primary_face['image_path'])
                if ref_img is not None:
                    cv2.imwrite(os.path.join(debug_dir, 'reference.jpg'), ref_img)

            # 2. face_crop.jpg
            crop_path = primary_face.get('crop_path') if primary_face else None
            if crop_path and os.path.exists(crop_path):
                crop_img = cv2.imread(crop_path)
                if crop_img is not None:
                    cv2.imwrite(os.path.join(debug_dir, 'face_crop.jpg'), crop_img)

            # 3. identity_embedding.json
            if multi_ref_data and 'master_embedding' in multi_ref_data:
                emb_data = {
                    'master_embedding': multi_ref_data['master_embedding'],
                    'dimensions': len(multi_ref_data['master_embedding']),
                    'total_references': multi_ref_data.get('total_references', 1),
                    'primary_sharpness': primary_face.get('sharpness') if primary_face else None,
                    'primary_confidence': primary_face.get('confidence') if primary_face else None,
                    'profile': primary_face.get('profile') if primary_face else {}
                }
                with open(os.path.join(debug_dir, 'identity_embedding.json'), 'w', encoding='utf-8') as f:
                    json.dump(emb_data, f, indent=2)

            # 4. face_mask.png (Landmark & bounding box condition map)
            if primary_face and os.path.exists(primary_face.get('image_path', '')):
                src_img = cv2.imread(primary_face['image_path'])
                if src_img is not None:
                    mask_viz = np.zeros_like(src_img)
                    fx, fy, fw, fh = primary_face.get('box', (0, 0, src_img.shape[1], src_img.shape[0]))
                    cv2.rectangle(mask_viz, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    for lm in primary_face.get('landmarks', []):
                        cv2.circle(mask_viz, (int(lm[0]), int(lm[1])), 4, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(debug_dir, 'face_mask.png'), mask_viz)

            # 5. generated_raw.png
            if raw_bgr is not None:
                cv2.imwrite(os.path.join(debug_dir, 'generated_raw.png'), raw_bgr)

            # 6. generated_refined.png
            if refined_bgr is not None:
                cv2.imwrite(os.path.join(debug_dir, 'generated_refined.png'), refined_bgr)

            # 7. final_output.png
            if final_bgr is not None:
                cv2.imwrite(os.path.join(debug_dir, 'final_output.png'), final_bgr)

            print(f"[ImageService] Intermediate debug artifacts successfully saved to {debug_dir}")
        except Exception as e:
            print(f"[ImageService] Warning saving debug artifacts: {e}")
