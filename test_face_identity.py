import unittest
import os
import cv2
import numpy as np
import json
from app import app
from config import Config
from services.face_pipeline import FacePipeline
from services.image_service import ImageService
from services.video_service import VideoService

class TestFaceIdentityPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
        os.makedirs(cls.test_dir, exist_ok=True)

        # 1. Create a synthetic realistic portrait image with clear facial features
        cls.portrait_path = os.path.join(cls.test_dir, 'test_portrait.jpg')
        portrait = np.full((400, 400, 3), (220, 220, 220), dtype=np.uint8)
        # Head ellipse
        cv2.ellipse(portrait, (200, 200), (90, 120), 0, 0, 360, (180, 195, 220), -1)
        # Left eye and right eye
        cv2.circle(portrait, (165, 175), 14, (255, 255, 255), -1)
        cv2.circle(portrait, (165, 175), 6, (50, 40, 30), -1)
        cv2.circle(portrait, (235, 175), 14, (255, 255, 255), -1)
        cv2.circle(portrait, (235, 175), 6, (50, 40, 30), -1)
        # Nose
        cv2.line(portrait, (200, 185), (200, 220), (140, 150, 180), 3)
        # Mouth
        cv2.ellipse(portrait, (200, 255), (30, 12), 0, 0, 180, (90, 100, 180), -1)
        cv2.imwrite(cls.portrait_path, portrait)

        # 2. Create a non-face image with texture/edges (passes quality check, but has no human face)
        cls.noface_path = os.path.join(cls.test_dir, 'test_noface.jpg')
        np.random.seed(42)
        noface = np.random.randint(40, 220, (300, 300, 3), dtype=np.uint8)
        # Add high-frequency checkerboard pattern to ensure high Laplacian sharpness
        for x in range(0, 300, 20):
            for y in range(0, 300, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    noface[y:y+20, x:x+20] = (240, 240, 240)
        cv2.imwrite(cls.noface_path, noface)

        # 3. Create a tiny image (<100x100)
        cls.tiny_path = os.path.join(cls.test_dir, 'test_tiny.jpg')
        cv2.imwrite(cls.tiny_path, np.zeros((60, 60, 3), dtype=np.uint8))

        # 4. Ensure non-guest user for member tests
        import database
        member = database.query_db("SELECT id FROM users WHERE username != 'guest'", one=True)
        if member:
            cls.test_user_id = member['id']
        else:
            cls.test_user_id = database.execute_db("""
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES ('testmember', 'member@jagangpt.local', 'hash', 'Test Member')
            """)

    @classmethod
    def tearDownClass(cls):
        import time
        time.sleep(0.5)
        for p in [cls.portrait_path, cls.noface_path, cls.tiny_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if os.path.exists(cls.test_dir):
            try:
                os.rmdir(cls.test_dir)
            except Exception:
                pass

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.test_user_id
            sess['is_guest'] = False

    def test_prompt_parser_disentanglement(self):
        """Test PromptParser extracting clothing, accessories, pose, and background."""
        from services.prompt_parser import PromptParser
        p = "Create a realistic cinematic full-body portrait of this person wearing a premium black shirt, black pants and black sunglasses, with a confident hero pose, standing against a pure black background, dramatic movie lighting."
        res = PromptParser.parse_prompt(p)
        self.assertTrue(res.get('has_sunglasses'))
        self.assertTrue(res.get('has_dark_clothing'))
        self.assertTrue(res.get('has_dark_background'))
        self.assertIn("black", res.get('clothing', '').lower())
        self.assertIn("sunglasses", res.get('accessories', '').lower())

    def test_image_validation(self):
        """Test validation rules for formats, file existence, and dimensions."""
        # Non-existent file
        v_missing = FacePipeline.validate_reference_image("non_existent_file.jpg")
        self.assertFalse(v_missing['valid'])

        # Image too small (<100x100)
        v_tiny = FacePipeline.validate_reference_image(self.tiny_path)
        self.assertFalse(v_tiny['valid'])
        self.assertIn("too small", v_tiny['error'])

        # Valid image
        v_ok = FacePipeline.validate_reference_image(self.noface_path)
        self.assertTrue(v_ok['valid'])

    def test_no_face_detection_error(self):
        """Verify user-friendly error when no face is present."""
        res = FacePipeline.detect_and_extract_face(self.noface_path)
        self.assertFalse(res['success'])
        self.assertIn("No human face detected", res['error'])

    def test_face_detection_and_embedding(self):
        """Verify face detection, landmark extraction, and 128-d SFace embedding vector."""
        res = FacePipeline.detect_and_extract_face(self.portrait_path)
        self.assertTrue(res['success'], f"Face detection failed: {res.get('error')}")
        self.assertEqual(len(res['landmarks']), 5)
        self.assertEqual(len(res['embedding']), 128)
        self.assertIn('aligned_crop_url', res)
        self.assertIn('profile', res)

    def test_identity_fusion(self):
        """Verify landmark-aligned Poisson seamless cloning produces valid output."""
        face_res = FacePipeline.detect_and_extract_face(self.portrait_path)
        scene = np.full((600, 600, 3), (40, 40, 60), dtype=np.uint8)
        fused = FacePipeline.fuse_identity_seamless(scene, self.portrait_path, face_res)
        self.assertEqual(fused.shape, scene.shape)
        self.assertIsInstance(fused, np.ndarray)

    def test_reference_upload_and_delete_endpoints(self):
        """Test API endpoints /api/reference/upload and /api/reference/delete."""
        with open(self.portrait_path, 'rb') as f:
            resp = self.client.post(
                '/api/reference/upload',
                data={'file': (f, 'test_portrait.jpg')},
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data['success'])
        self.assertIn('face_crop_url', data)
        self.assertIn('confidence', data)

        stored_file = data['file_path']

        # Delete for user privacy
        del_resp = self.client.post(
            '/api/reference/delete',
            data=json.dumps({'file_path': stored_file}),
            content_type='application/json'
        )
        self.assertEqual(del_resp.status_code, 200)
        self.assertFalse(os.path.exists(stored_file))

    def test_image_service_with_identity_reference(self):
        """Test ImageService with reference image preservation."""
        result = ImageService.generate_image(
            prompt="A movie hero portrait in a black jacket",
            user_id=self.test_user_id,
            reference_image_path=self.portrait_path,
            aspect_ratio='1:1',
            resolution='720p'
        )
        self.assertIn('image_url', result)
        self.assertTrue(result['reference_used'])
        self.assertTrue(result['image_url'].startswith('/uploads/images/'))

    def test_quality_assessment_blurry_rejection(self):
        """Verify Laplacian variance quality assessment rejects excessively blurry reference photos."""
        # Create heavily blurred image
        portrait = cv2.imread(self.portrait_path)
        blurry = cv2.GaussianBlur(portrait, (55, 55), 0)
        blurry_path = os.path.join(self.test_dir, 'test_blurry.jpg')
        cv2.imwrite(blurry_path, blurry)

        qual = FacePipeline.assess_reference_quality(blurry_path)
        if os.path.exists(blurry_path):
            os.remove(blurry_path)

        self.assertFalse(qual['valid'])
        self.assertIn("blurry", qual['error'].lower())

    def test_multi_reference_aggregation(self):
        """Verify multi-reference photo aggregation into normalized master vector E_master."""
        multi_res = FacePipeline.detect_and_extract_multi_faces([self.portrait_path, self.portrait_path])
        self.assertTrue(multi_res['success'])
        self.assertEqual(len(multi_res['master_embedding']), 128)
        self.assertEqual(len(multi_res['embeddings']), 2)

        # Vector norm should be approximately 1.0 (normalized)
        norm = np.linalg.norm(multi_res['master_embedding'])
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_super_resolution_4k_dimensions(self):
        """Verify super_resolve_and_enhance generates True 4K pixel dimensions."""
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)

        # 16:9 Landscape 4K -> 3840x2160
        res_4k_16_9 = FacePipeline.super_resolve_and_enhance(test_img, target_res='4k', aspect_ratio='16:9')
        self.assertEqual(res_4k_16_9.shape[0], 2160)
        self.assertEqual(res_4k_16_9.shape[1], 3840)

        # 9:16 Portrait Reel 4K -> 2160x3840
        res_4k_9_16 = FacePipeline.super_resolve_and_enhance(test_img, target_res='4k', aspect_ratio='9:16')
        self.assertEqual(res_4k_9_16.shape[0], 3840)
        self.assertEqual(res_4k_9_16.shape[1], 2160)

        # 1:1 Square 4K -> 3840x3840
        res_4k_1_1 = FacePipeline.super_resolve_and_enhance(test_img, target_res='4k', aspect_ratio='1:1')
        self.assertEqual(res_4k_1_1.shape[0], 3840)
        self.assertEqual(res_4k_1_1.shape[1], 3840)

    def test_image_service_with_multi_references(self):
        """Test ImageService generating image conditioned on multiple references and 4K output."""
        result = ImageService.generate_image(
            prompt="Cinematic hero standing in rain wearing black leather jacket",
            user_id=self.test_user_id,
            reference_image_paths=[self.portrait_path, self.portrait_path],
            aspect_ratio='16:9',
            resolution='4k',
            identity_threshold=0.70
        )
        self.assertIn('image_url', result)
        self.assertTrue(result['reference_used'])
        self.assertGreaterEqual(result.get('attempts', 1), 1)

    def test_hero_portrait_identity_and_transformation(self):
        """
        Verify simultaneous identity preservation and visual transformation:
        Same person + black shirt + black pants + black sunglasses + hero pose + pure black background.
        """
        hero_prompt = "Create a realistic cinematic full-body portrait of this person wearing a premium black shirt, black pants and black sunglasses, with a confident hero pose, standing against a pure black background, dramatic movie lighting."
        result = ImageService.generate_image(
            prompt=hero_prompt,
            user_id=self.test_user_id,
            reference_image_paths=[self.portrait_path],
            aspect_ratio='1:1',
            resolution='720p',
            identity_threshold=0.65
        )

        self.assertIn('image_url', result)
        self.assertTrue(result['reference_used'])
        self.assertTrue(result['disentanglement']['has_sunglasses'])
        self.assertTrue(result['disentanglement']['has_dark_clothing'])
        self.assertTrue(result['disentanglement']['has_dark_background'])
        self.assertTrue(result['prompt_adherence']['verified'])

        # Verify output image on disk
        local_path = os.path.join(Config.IMAGES_FOLDER, result['filename'])
        self.assertTrue(os.path.exists(local_path))
        self.assertGreater(os.path.getsize(local_path), 5000)

if __name__ == '__main__':
    unittest.main()
