import unittest
import json
import os
from app import app
import database
from services.telugu_nlp import detect_language, normalize_media_prompt
from services.file_analyzer import FileAnalyzer
from services.image_service import ImageService
from services.video_service import VideoService
from services.voice_service import VoiceService
from services.ai_router import AIRouter

class JaganGptTestSuite(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['is_guest'] = False

    def test_database_tables(self):
        """Verify all tables exist in SQLite."""
        conn = database.get_db()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r['name'] for r in cur.fetchall()}
        conn.close()

        expected = {'users', 'conversations', 'messages', 'attachments', 'generated_images', 'generated_videos', 'user_settings'}
        for t in expected:
            self.assertIn(t, tables)

    def test_telugu_and_tanglish_detection(self):
        """Test language detection for English, Telugu Unicode, and Tanglish."""
        self.assertEqual(detect_language("Hello, how are you?"), 'en')
        self.assertEqual(detect_language("నమస్కారం, మీరు ఎలా ఉన్నారు?"), 'te')
        self.assertEqual(detect_language("Python lo functions ela work avuthayo explain cheyyi"), 'tanglish')
        self.assertEqual(detect_language("Machine learning ante enti?"), 'tanglish')

    def test_prompt_normalization(self):
        """Test Telugu/Tanglish prompt translation for image/video engines."""
        prompt = "ఒక futuristic Hyderabad city ని చూపించే photo generate చేయి"
        normalized = normalize_media_prompt(prompt)
        self.assertTrue(any(w in normalized.lower() for w in ['futuristic', 'hyderabad', 'photo']))

    def test_telugu_visual_text_sanitization(self):
        """Verify that sanitize_visual_text purges 100% of Telugu Unicode characters."""
        from services.telugu_nlp import sanitize_visual_text, TELUGU_UNICODE_REGEX
        sample = "నమస్కారం! నేను JaganGptని. మీ ప్రశ్న: AI చిత్రం మరియు వీడియో."
        sanitized = sanitize_visual_text(sample)
        self.assertFalse(TELUGU_UNICODE_REGEX.search(sanitized))
        self.assertIn("JaganGpt", sanitized)

    def test_file_analyzer(self):
        """Test analysis on text and CSV files."""
        test_file = os.path.join(os.path.dirname(__file__), 'test_sample.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("JaganGpt is a modern multimodal AI assistant supporting English and Telugu.")

        res = FileAnalyzer.analyze_file(test_file, 'test_sample.txt')
        self.assertEqual(res['type_category'], 'text')
        self.assertIn("JaganGpt", res['extracted_text'])
        os.remove(test_file)

    def test_image_generation_service(self):
        """Test image generation pipeline."""
        res = ImageService.generate_image("A futuristic cyber lion logo", user_id=1)
        self.assertIn('image_url', res)
        self.assertTrue(res['image_url'].startswith('/uploads/images/'))

    def test_video_generation_job(self):
        """Test video generation asynchronous job queue."""
        job = VideoService.create_video_job("A drone shot over illuminated towers", user_id=1)
        self.assertIn('job_id', job)
        self.assertEqual(job['status'], 'processing')

        # Check status
        status = VideoService.get_job_status(job['job_id'])
        self.assertIn(status['status'], ['processing', 'completed'])

    def test_voice_tts(self):
        """Test Telugu and English TTS synthesis."""
        url_en = VoiceService.synthesize_speech("Welcome to JaganGpt", language='en')
        self.assertTrue(url_en.startswith('/uploads/audio/'))

        url_te = VoiceService.synthesize_speech("జగన్ జీపీటీకి స్వాగతం", language='te')
        self.assertTrue(url_te.startswith('/uploads/audio/'))

    def test_chat_api_flow(self):
        """Test end-to-end chat endpoint."""
        # 1. New chat
        res = self.client.post('/chat', 
            data=json.dumps({'message': 'Explain Python list comprehensions with a quick code sample'}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('reply', data)
        self.assertIn('conversation_id', data)

        conv_id = data['conversation_id']

        # 2. Query conversations API
        res_convs = self.client.get('/api/conversations')
        self.assertEqual(res_convs.status_code, 200)
        convs_data = json.loads(res_convs.data)
        self.assertTrue(any(c['id'] == conv_id for c in convs_data['conversations']))

    def test_settings_api(self):
        """Test saving Gemini API key and language preference."""
        res = self.client.post('/api/settings',
            data=json.dumps({
                'gemini_api_key': 'test-gemini-key-12345',
                'preferred_language': 'te'
            }),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])

        # Reset user 1 settings back so it doesn't pollute production environment
        database.execute_db("UPDATE user_settings SET gemini_api_key = NULL WHERE user_id = 1")

    def test_guest_text_allowed_and_media_restricted(self):
        """Verify Guest mode allows text chat, but strictly blocks images and videos."""
        # Log in as guest
        self.client.get('/guest-login')

        # 1. Text chat should succeed
        res_text = self.client.post('/chat',
            data=json.dumps({'message': 'Hello, what is Python?'}),
            content_type='application/json'
        )
        self.assertEqual(res_text.status_code, 200)
        data_text = json.loads(res_text.data)
        self.assertIn('reply', data_text)
        self.assertEqual(data_text.get('media_type'), 'text')

        # 2. Image generation request in chat must return guest restriction message
        res_img_chat = self.client.post('/chat',
            data=json.dumps({'message': 'Generate an image of a red sports car'}),
            content_type='application/json'
        )
        self.assertEqual(res_img_chat.status_code, 200)
        data_img_chat = json.loads(res_img_chat.data)
        self.assertIn('Guest Mode Restriction', data_img_chat['reply'])
        self.assertIsNone(data_img_chat.get('media_url'))

        # 3. Video generation request in chat must return guest restriction message
        res_vid_chat = self.client.post('/chat',
            data=json.dumps({'message': 'Generate a video of waves crashing on the beach'}),
            content_type='application/json'
        )
        self.assertEqual(res_vid_chat.status_code, 200)
        data_vid_chat = json.loads(res_vid_chat.data)
        self.assertIn('Guest Mode Restriction', data_vid_chat['reply'])

        # 4. Direct /api/generate-image endpoint must return 403 for guests
        res_api_img = self.client.post('/api/generate-image',
            data=json.dumps({'prompt': 'A cyber city'}),
            content_type='application/json'
        )
        self.assertEqual(res_api_img.status_code, 403)
        self.assertTrue(json.loads(res_api_img.data).get('is_guest_restricted'))

        # 5. Direct /api/generate-video endpoint must return 403 for guests
        res_api_vid = self.client.post('/api/generate-video',
            data=json.dumps({'prompt': 'A flying drone'}),
            content_type='application/json'
        )
        self.assertEqual(res_api_vid.status_code, 403)
        self.assertTrue(json.loads(res_api_vid.data).get('is_guest_restricted'))

if __name__ == '__main__':
    unittest.main()
