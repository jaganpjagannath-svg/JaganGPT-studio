import os
import uuid
import json
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_from_directory, abort, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
import database
from services.ai_router import AIRouter
from services.file_analyzer import FileAnalyzer
from services.image_service import ImageService
from services.video_service import VideoService
from services.voice_service import VoiceService
from services.face_pipeline import FacePipeline
from services.telugu_nlp import detect_language

app = Flask(__name__)
app.config.from_object(Config)

# Ensure database is initialized
database.init_db()

# ----------------- Helper Functions -----------------

def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return database.query_db("SELECT * FROM users WHERE id = ?", (user_id,), one=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('welcome'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/welcome')
def welcome():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('welcome.html')

@app.route('/guest-login')
def guest_login():
    guest = database.query_db("SELECT id FROM users WHERE username = 'guest'", one=True)
    if not guest:
        guest_id = database.execute_db("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES ('guest', 'guest@jagangpt.local', ?, 'Guest Explorer')
        """, (generate_password_hash('guest123'),))
        database.execute_db("INSERT INTO user_settings (user_id) VALUES (?)", (guest_id,))
        session['user_id'] = guest_id
    else:
        session['user_id'] = guest['id']
    session['is_guest'] = True
    return redirect(url_for('home'))

# ----------------- Media Serving -----------------

@app.route('/uploads/<folder>/<filename>')
def serve_upload(folder, filename):
    if folder not in ['docs', 'images', 'videos', 'audio', 'temp_references']:
        abort(404)
    target_dir = os.path.join(Config.UPLOAD_FOLDER, folder)
    return send_from_directory(target_dir, filename)

# ----------------- Authentication Routes -----------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        email = request.form.get('email', '').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template('register.html')

        existing = database.query_db("SELECT id FROM users WHERE username = ? OR email = ?", (username, email), one=True)
        if existing:
            flash("Username or email already exists.", "error")
            return render_template('register.html')

        user_id = database.execute_db("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        """, (username, email, generate_password_hash(password), full_name))

        # Initialize user settings
        database.execute_db("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))

        session['user_id'] = user_id
        flash(f"Welcome to JaganGpt, {username}!", "success")
        return redirect(url_for('home'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_input = request.form.get('login_input', '').strip().lower()
        password = request.form.get('password', '')

        user = database.query_db("SELECT * FROM users WHERE username = ? OR email = ?", (login_input, login_input), one=True)
        if not user or not check_password_hash(user['password_hash'], password):
            flash("Invalid credentials.", "error")
            return render_template('login.html')

        session['user_id'] = user['id']
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))

# ----------------- Main Chat Interface -----------------

@app.route('/')
@login_required
def home():
    current_user = get_current_user()
    user_id = current_user['id']
    is_guest = bool(session.get('is_guest') or (current_user and current_user['username'] == 'guest'))

    # Get user's conversations
    conversations = database.query_db("""
        SELECT * FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (user_id,))

    # Get user settings
    settings = database.query_db("SELECT * FROM user_settings WHERE user_id = ?", (user_id,), one=True)
    if not settings:
        database.execute_db("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
        settings = database.query_db("SELECT * FROM user_settings WHERE user_id = ?", (user_id,), one=True)

    return render_template(
        'index.html',
        current_user=dict(current_user),
        is_guest=is_guest,
        conversations=[dict(c) for c in conversations],
        settings=dict(settings) if settings else {},
        models=Config.MODELS,
        has_gemini_key=bool(settings['gemini_api_key'] or Config.GEMINI_API_KEY)
    )

# ----------------- Chat & AI APIs -----------------

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    current_user = get_current_user()
    user_id = current_user['id']
    is_guest = bool(session.get('is_guest') or (current_user and current_user['username'] == 'guest'))
    data = request.get_json() or {}

    user_message = data.get('message', '').strip()
    conversation_id = data.get('conversation_id')
    model_choice = data.get('model', Config.DEFAULT_MODEL)
    attachments = data.get('attachments', []) # list of attachment dicts

    if not user_message and not attachments:
        return jsonify({'error': 'Message or attachment is required.'}), 400

    # Ensure conversation exists
    if not conversation_id:
        # Generate initial title from message
        title = (user_message[:35] + '...') if len(user_message) > 35 else (user_message or 'Multimodal Chat')
        conversation_id = database.execute_db("""
            INSERT INTO conversations (user_id, title, model)
            VALUES (?, ?, ?)
        """, (user_id, title, model_choice))
    else:
        # Update timestamp
        database.execute_db("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))

    # Save user message
    database.execute_db("""
        INSERT INTO messages (conversation_id, role, content, attachments_json)
        VALUES (?, 'user', ?, ?)
    """, (conversation_id, user_message, json.dumps(attachments)))

    # Process through Multimodal AI Router
    result = AIRouter.process_query(
        user_message=user_message,
        user_id=user_id,
        conversation_id=conversation_id,
        model_choice=model_choice,
        attachments=attachments,
        is_guest=is_guest
    )

    # Save assistant message
    msg_id = database.execute_db("""
        INSERT INTO messages (conversation_id, role, content, media_type, media_url)
        VALUES (?, 'assistant', ?, ?, ?)
    """, (conversation_id, result['reply'], result.get('media_type', 'text'), result.get('media_url')))

    # Fetch updated conversation title
    conv = database.query_db("SELECT title FROM conversations WHERE id = ?", (conversation_id,), one=True)

    return jsonify({
        'reply': result['reply'],
        'conversation_id': conversation_id,
        'title': conv['title'] if conv else 'Chat',
        'media_type': result.get('media_type', 'text'),
        'media_url': result.get('media_url'),
        'job_id': result.get('job_id'),
        'model_used': result.get('model_used', 'JaganGpt Smart'),
        'language_detected': result.get('language_detected', 'en'),
        'message_id': msg_id
    })

# ----------------- Conversations APIs -----------------

@app.route('/api/conversations')
@login_required
def get_conversations():
    current_user = get_current_user()
    convs = database.query_db("""
        SELECT * FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (current_user['id'],))
    return jsonify({'conversations': [dict(c) for c in convs]})

@app.route('/api/conversations/<int:conv_id>')
@login_required
def get_conversation_messages(conv_id):
    current_user = get_current_user()
    conv = database.query_db("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conv_id, current_user['id']), one=True)
    if not conv:
        return jsonify({'error': 'Conversation not found'}), 404

    messages = database.query_db("""
        SELECT * FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, (conv_id,))

    return jsonify({
        'conversation': dict(conv),
        'messages': [dict(m) for m in messages]
    })

@app.route('/api/conversations/<int:conv_id>', methods=['DELETE'])
@login_required
def delete_conversation(conv_id):
    current_user = get_current_user()
    database.execute_db("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, current_user['id']))
    return jsonify({'success': True})

@app.route('/api/conversations/<int:conv_id>/rename', methods=['POST'])
@login_required
def rename_conversation(conv_id):
    current_user = get_current_user()
    data = request.get_json() or {}
    new_title = data.get('title', '').strip()
    if not new_title:
        return jsonify({'error': 'Title required'}), 400

    database.execute_db("UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?", (new_title, conv_id, current_user['id']))
    return jsonify({'success': True, 'title': new_title})

# ----------------- File Upload & Analysis -----------------

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    current_user = get_current_user()
    file = request.files.get('file')
    conversation_id = request.form.get('conversation_id')

    if not file or file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    unique_name = f"doc_{uuid.uuid4().hex[:10]}.{ext}"
    
    # Place images in images folder, others in docs
    folder_type = 'images' if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif'] else 'docs'
    target_path = os.path.join(Config.UPLOAD_FOLDER, folder_type, unique_name)
    file.save(target_path)

    # Run deep analysis
    analysis = FileAnalyzer.analyze_file(target_path, file.filename)

    # Save to database if conversation exists
    attachment_id = 0
    if conversation_id:
        attachment_id = database.execute_db("""
            INSERT INTO attachments (conversation_id, user_id, original_name, stored_name, file_type, file_size, file_path, extracted_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, current_user['id'], file.filename, unique_name, ext, analysis['file_size'], target_path, analysis['extracted_text']))

    return jsonify({
        'success': True,
        'attachment_id': attachment_id,
        'original_name': file.filename,
        'stored_name': unique_name,
        'file_size': analysis['file_size'],
        'summary': analysis['summary'],
        'extracted_text': analysis['extracted_text'],
        'is_image': analysis['is_image'],
        'preview_url': f"/uploads/{folder_type}/{unique_name}"
    })

# ----------------- Reference Face & Identity Conditioning APIs -----------------

@app.route('/api/reference/upload', methods=['POST'])
@login_required
def api_reference_upload():
    current_user = get_current_user()
    is_guest = bool(session.get('is_guest') or (current_user and current_user['username'] == 'guest'))
    if is_guest:
        return jsonify({
            'error': 'Guest mode allows text only. Please log in or sign up to use reference face conditioning.',
            'is_guest_restricted': True
        }), 403

    file = request.files.get('file')
    slot = request.form.get('slot', '0')
    if not file or file.filename == '':
        return jsonify({'error': 'No reference image file provided.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        return jsonify({'error': f'Unsupported format {ext}. Please upload a PNG, JPG, JPEG, or WEBP photo.'}), 400

    temp_dir = os.path.join(Config.UPLOAD_FOLDER, 'temp_references')
    os.makedirs(temp_dir, exist_ok=True)
    unique_name = f"ref_{slot}_{uuid.uuid4().hex[:10]}{ext}"
    target_path = os.path.join(temp_dir, unique_name)
    file.save(target_path)

    # 1. Image Quality Assessment (sharpness, blurriness, resolution)
    qual = FacePipeline.assess_reference_quality(target_path)
    if not qual['valid']:
        if os.path.exists(target_path):
            os.remove(target_path)
        return jsonify({'success': False, 'error': qual['error']}), 400

    # 2. Face Detection & Landmark Extraction
    face_data = FacePipeline.detect_and_extract_face(target_path)
    if not face_data['success']:
        if os.path.exists(target_path):
            os.remove(target_path)
        return jsonify({
            'success': False,
            'error': face_data['error']
        }), 400

    return jsonify({
        'success': True,
        'slot': slot,
        'file_path': target_path,
        'filename': unique_name,
        'preview_url': f"/uploads/temp_references/{unique_name}",
        'face_crop_url': face_data.get('aligned_crop_url'),
        'confidence': face_data.get('confidence'),
        'sharpness': qual.get('sharpness', 0.0),
        'is_high_quality': qual.get('is_high_quality', False),
        'profile': face_data.get('profile')
    })

@app.route('/api/reference/delete', methods=['POST'])
@login_required
def api_reference_delete():
    data = request.get_json() or {}
    file_path = data.get('file_path', '').strip()
    file_paths = data.get('file_paths', [])

    if file_path:
        FacePipeline.delete_temporary_reference(file_path)

    for p in file_paths:
        if p:
            FacePipeline.delete_temporary_reference(p)

    return jsonify({'success': True})

# ----------------- Dedicated Image & Video APIs -----------------

@app.route('/api/generate-image', methods=['POST'])
@login_required
def api_generate_image():
    current_user = get_current_user()
    is_guest = bool(session.get('is_guest') or (current_user and current_user['username'] == 'guest'))
    if is_guest:
        return jsonify({
            'error': 'Guest mode allows text only. Please log in or sign up to generate images.',
            'is_guest_restricted': True
        }), 403

    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    conversation_id = data.get('conversation_id')

    # Support multiple reference photos (list of 1-4 paths)
    reference_paths = data.get('reference_paths') or []
    single_ref = data.get('reference_path')
    if single_ref and single_ref not in reference_paths:
        reference_paths.append(single_ref)

    identity_threshold = float(data.get('identity_threshold', 0.70))
    identity_strength = float(data.get('identity_strength', 0.75))
    prompt_strength = float(data.get('prompt_strength', 0.85))
    negative_prompt = data.get('negative_prompt')
    aspect_ratio = data.get('aspect_ratio', '1:1')
    resolution = data.get('resolution', '1080p')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    res = ImageService.generate_image(
        prompt=prompt,
        user_id=current_user['id'],
        conversation_id=conversation_id,
        reference_image_paths=reference_paths,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        identity_threshold=identity_threshold,
        identity_strength=identity_strength,
        prompt_strength=prompt_strength
    )

    if 'error' in res and not res.get('image_url'):
        return jsonify({'error': res['error']}), 400

    return jsonify(res)

@app.route('/api/generate-video', methods=['POST'])
@login_required
def api_generate_video():
    current_user = get_current_user()
    is_guest = bool(session.get('is_guest') or (current_user and current_user['username'] == 'guest'))
    if is_guest:
        return jsonify({
            'error': 'Guest mode allows text only. Please log in or sign up to generate videos.',
            'is_guest_restricted': True
        }), 403

    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    conversation_id = data.get('conversation_id')

    # Support multiple reference photos (1-4 paths)
    reference_paths = data.get('reference_paths') or []
    single_ref = data.get('reference_path')
    if single_ref and single_ref not in reference_paths:
        reference_paths.append(single_ref)

    identity_threshold = float(data.get('identity_threshold', 0.70))
    duration = int(data.get('duration', 5))
    fps = int(data.get('fps', 24))
    motion_strength = data.get('motion_strength', 'medium')
    camera_movement = data.get('camera_movement', 'zoom_in')
    resolution = data.get('resolution', '1080p')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    res = VideoService.create_video_job(
        prompt=prompt,
        user_id=current_user['id'],
        conversation_id=conversation_id,
        reference_image_paths=reference_paths,
        duration=duration,
        fps=fps,
        motion_strength=motion_strength,
        camera_movement=camera_movement,
        resolution=resolution,
        identity_threshold=identity_threshold
    )

    if 'error' in res and not res.get('job_id'):
        return jsonify({'error': res['error']}), 400

    return jsonify(res)

@app.route('/api/video-status/<job_id>')
def api_video_status(job_id):
    status = VideoService.get_job_status(job_id)
    return jsonify(status)

# ----------------- Voice & Speech Synthesis -----------------

@app.route('/api/voice/tts', methods=['POST'])
def api_tts():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    lang = data.get('lang', 'auto')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    audio_url = VoiceService.synthesize_speech(text, lang)
    return jsonify({'audio_url': audio_url})

# ----------------- Ratings & Settings -----------------

@app.route('/api/message/rate', methods=['POST'])
@login_required
def rate_message():
    data = request.get_json() or {}
    msg_id = data.get('message_id')
    rating = data.get('rating', 0)
    database.execute_db("UPDATE messages SET rating = ? WHERE id = ?", (rating, msg_id))
    return jsonify({'success': True})

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    current_user = get_current_user()
    user_id = current_user['id']

    if request.method == 'POST':
        data = request.get_json() or {}
        gemini_key = data.get('gemini_api_key', '').strip()
        openai_key = data.get('openai_api_key', '').strip()
        replicate_token = data.get('replicate_api_token', '').strip()
        fal_key = data.get('fal_api_key', '').strip()
        hf_token = data.get('huggingface_token', '').strip()
        pref_lang = data.get('preferred_language', 'auto')
        theme = data.get('theme', 'dark')
        voice_speed = float(data.get('voice_speed', 1.0))

        database.execute_db("""
            UPDATE user_settings
            SET gemini_api_key = ?, openai_api_key = ?, replicate_api_token = ?,
                fal_api_key = ?, huggingface_token = ?, preferred_language = ?,
                theme = ?, voice_speed = ?
            WHERE user_id = ?
        """, (gemini_key, openai_key, replicate_token, fal_key, hf_token, pref_lang, theme, voice_speed, user_id))

        return jsonify({'success': True, 'message': 'Settings saved successfully'})

    settings = database.query_db("SELECT * FROM user_settings WHERE user_id = ?", (user_id,), one=True)
    return jsonify({'settings': dict(settings) if settings else {}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)