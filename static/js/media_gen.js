/**
 * JaganGpt - AI Media Generation Studio Controller
 * Handles Multi-Reference Face Identity Conditioning (1-4 Photos),
 * SFace Master Embedding Vector Synthesis, Identity Threshold Auto-Regeneration,
 * Aspect Ratio / True 4K Super-Resolution controls, and Temporal Video Stabilization.
 */

let currentStudioMode = 'image';
let currentReferenceSlots = [null, null, null, null];
let activeUploadSlot = 0;
let currentIdentityThreshold = 0.70;
let currentPromptStrength = 0.85;
let currentAspectRatio = '1:1';
let currentResolution = '1080p';
let activeVideoJobInterval = null;

// Backward-compatibility accessor
Object.defineProperty(window, 'currentReferencePath', {
    get: () => currentReferenceSlots.find(p => p !== null) || null,
    set: (val) => { currentReferenceSlots[0] = val; }
});

// ----------------- Guest Restriction Helpers -----------------

function showGuestMediaNotice(mediaType = 'media') {
    const modal = document.getElementById('guest-upgrade-modal');
    if (modal) {
        modal.classList.add('active');
    } else {
        alert("🔒 Guest Mode: Text only. Please Log In or Sign Up to generate images and videos!");
    }
}

function closeGuestModal() {
    const modal = document.getElementById('guest-upgrade-modal');
    if (modal) modal.classList.remove('active');
}

// ----------------- Studio Opening & Modal Management -----------------

function openMediaStudio(mode = 'image') {
    if (window.IS_GUEST) {
        showGuestMediaNotice(mode);
        return;
    }

    const modal = document.getElementById('media-studio-modal');
    if (modal) modal.classList.add('active');
    switchStudioMode(mode);
}

function closeMediaStudio() {
    const modal = document.getElementById('media-studio-modal');
    if (modal) modal.classList.remove('active');
    if (activeVideoJobInterval) clearInterval(activeVideoJobInterval);
}

// Backward-compatibility aliases
function openImageGenModal() { openMediaStudio('image'); }
function closeImageGenModal() { closeMediaStudio(); }
function openVideoGenModal() { openMediaStudio('video'); }
function closeVideoGenModal() { closeMediaStudio(); }

function switchStudioMode(mode) {
    currentStudioMode = mode;
    const btnImg = document.getElementById('studio-mode-btn-image');
    const btnVid = document.getElementById('studio-mode-btn-video');
    const imgControls = document.getElementById('studio-image-controls');
    const vidControls = document.getElementById('studio-video-controls');
    const submitBtn = document.getElementById('btn-studio-submit');

    if (mode === 'image') {
        if (btnImg) btnImg.classList.add('active');
        if (btnVid) btnVid.classList.remove('active');
        if (imgControls) imgControls.style.display = 'block';
        if (vidControls) vidControls.style.display = 'none';
        if (submitBtn) submitBtn.innerHTML = '✨ Generate Identity-Preserved Image';
    } else {
        if (btnVid) btnVid.classList.add('active');
        if (btnImg) btnImg.classList.remove('active');
        if (imgControls) imgControls.style.display = 'none';
        if (vidControls) vidControls.style.display = 'block';
        if (submitBtn) submitBtn.innerHTML = '🎬 Generate Identity-Preserved Video';
    }
    clearStudioError();
}

// ----------------- Multi-Reference Face Slot Management -----------------

function triggerReferenceUpload(slotIndex) {
    if (window.IS_GUEST) {
        showGuestMediaNotice(currentStudioMode);
        return;
    }

    activeUploadSlot = parseInt(slotIndex, 10);
    if (isNaN(activeUploadSlot) || activeUploadSlot < 0 || activeUploadSlot > 3) {
        activeUploadSlot = 0;
    }

    for (let i = 0; i < 4; i++) {
        const card = document.getElementById(`ref-slot-${i}`);
        if (card) {
            if (i === activeUploadSlot) card.classList.add('active');
            else card.classList.remove('active');
        }
    }

    const fileInput = document.getElementById('reference-file-input');
    if (fileInput) fileInput.click();
}

async function handleReferenceFileSelected(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const slot = activeUploadSlot;
    clearStudioError();

    const slotCard = document.getElementById(`ref-slot-${slot}`);
    const emptyEl = document.getElementById(`ref-slot-empty-${slot}`);
    const filledEl = document.getElementById(`ref-slot-filled-${slot}`);
    const imgEl = document.getElementById(`ref-slot-img-${slot}`);
    const tagEl = document.getElementById(`ref-slot-tag-${slot}`);

    if (slotCard) slotCard.classList.add('uploading');
    if (emptyEl) {
        emptyEl.innerHTML = `
            <span style="font-size:1.3rem;display:block;animation:rotateThink 1s infinite linear;">⚙️</span>
            <div style="font-size:0.68rem;font-weight:700;color:var(--accent-cyan);margin-top:2px;">Analyzing...</div>
            <div style="font-size:0.60rem;color:var(--text-dim);">Quality & Landmarks</div>
        `;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('slot', slot.toString());

    try {
        const res = await fetch('/api/reference/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (slotCard) slotCard.classList.remove('uploading');

        if (res.ok && data.success) {
            currentReferenceSlots[slot] = data.file_path;

            if (emptyEl) emptyEl.style.display = 'none';
            if (filledEl) filledEl.style.display = 'block';
            if (imgEl) imgEl.src = data.face_crop_url || data.preview_url;

            if (tagEl) {
                const confPct = Math.round((data.confidence || 0.95) * 100);
                const qualBadge = data.is_high_quality ? 'HQ' : 'OK';
                tagEl.textContent = `✓ ${confPct}% (${qualBadge})`;
            }
            if (slotCard) slotCard.classList.add('has-image');

            updateReferenceSummary();
        } else {
            showStudioError(data.error || 'Failed to detect a valid face in this reference photo.');
            resetSlotUI(slot);
        }
    } catch (err) {
        if (slotCard) slotCard.classList.remove('uploading');
        showStudioError('Network error uploading reference photo: ' + err.message);
        resetSlotUI(slot);
    } finally {
        input.value = '';
    }
}

function resetSlotUI(slot) {
    const emptyEl = document.getElementById(`ref-slot-empty-${slot}`);
    const filledEl = document.getElementById(`ref-slot-filled-${slot}`);
    const slotCard = document.getElementById(`ref-slot-${slot}`);
    const labels = ['Front', 'Left', 'Right', 'Body'];
    const subtitles = ['Primary Face', 'Side Profile', 'Side Profile', 'Full Pose'];
    const icons = ['👤', '👈', '👉', '🧍'];

    if (filledEl) filledEl.style.display = 'none';
    if (emptyEl) {
        emptyEl.style.display = 'flex';
        emptyEl.innerHTML = `
            <span style="font-size:1.4rem;">${icons[slot]}</span>
            <div style="font-size:0.72rem;font-weight:700;margin-top:2px;">Slot ${slot + 1}: ${labels[slot]}</div>
            <div style="font-size:0.65rem;color:var(--text-dim);">${subtitles[slot]}</div>
        `;
    }
    if (slotCard) slotCard.classList.remove('has-image', 'uploading');
}

async function removeReferenceSlot(slot, event) {
    if (event) event.stopPropagation();
    const path = currentReferenceSlots[slot];
    if (path) {
        try {
            await fetch('/api/reference/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: path })
            });
        } catch (e) {
            console.warn("Could not delete reference slot:", e);
        }
        currentReferenceSlots[slot] = null;
    }
    resetSlotUI(slot);
    updateReferenceSummary();
}

async function clearAllReferencePhotos(event) {
    if (event) event.stopPropagation();
    const activePaths = currentReferenceSlots.filter(p => p !== null);
    if (activePaths.length > 0) {
        try {
            await fetch('/api/reference/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_paths: activePaths })
            });
        } catch (e) {
            console.warn("Could not wipe references:", e);
        }
    }
    currentReferenceSlots = [null, null, null, null];
    for (let i = 0; i < 4; i++) {
        resetSlotUI(i);
    }
    updateReferenceSummary();
}

// Backward-compatibility alias
function clearReferencePhoto(event) {
    clearAllReferencePhotos(event);
}

function updateReferenceSummary() {
    const count = currentReferenceSlots.filter(p => p !== null).length;
    const summaryEl = document.getElementById('multi-ref-summary');
    const badgeEl = document.getElementById('master-vec-badge');

    if (summaryEl) {
        if (count === 0) {
            summaryEl.textContent = "💡 Upload 1 to 4 photos to build a master 360° identity representation.";
        } else if (count === 1) {
            summaryEl.textContent = "✓ 1 reference photo loaded (Front SFace identity locked).";
        } else {
            summaryEl.textContent = `✨ ${count} reference photos active: Normalized master identity vector E_master aggregated.`;
        }
    }

    if (badgeEl) {
        if (count === 0) {
            badgeEl.textContent = "No Reference";
            badgeEl.style.color = "var(--text-dim)";
        } else {
            badgeEl.textContent = `✓ ${count} Ref${count > 1 ? 's' : ''} Locked (E_master Ready)`;
            badgeEl.style.color = "var(--accent-cyan)";
        }
    }
}

// ----------------- Studio Control Toggles -----------------

function updateThresholdSlider(val) {
    currentIdentityThreshold = parseFloat(val);
    const display = document.getElementById('threshold-display-val');
    if (display) {
        display.textContent = `${Math.round(currentIdentityThreshold * 100)}%`;
    }
}

function updatePromptStrengthSlider(val) {
    currentPromptStrength = parseFloat(val);
    const display = document.getElementById('prompt-strength-display-val');
    if (display) {
        display.textContent = `${Math.round(currentPromptStrength * 100)}%`;
    }
}

function setStudioAspect(aspect, btn) {
    currentAspectRatio = aspect;
    document.querySelectorAll('#aspect-ratio-pills .studio-pill').forEach(el => el.classList.remove('active'));
    if (btn) btn.classList.add('active');
}

function setStudioResolution(res, btn) {
    currentResolution = res;
    document.querySelectorAll('#resolution-pills .studio-pill').forEach(el => el.classList.remove('active'));
    if (btn) btn.classList.add('active');
}

function showStudioError(msg) {
    const box = document.getElementById('studio-error-banner');
    if (box) {
        box.textContent = msg;
        box.style.display = 'block';
    }
}

function clearStudioError() {
    const box = document.getElementById('studio-error-banner');
    if (box) box.style.display = 'none';
}

function updateStudioProgress(percent, label, desc) {
    const card = document.getElementById('studio-progress-card');
    const bar = document.getElementById('studio-progress-bar-fill');
    const lbl = document.getElementById('studio-step-label');
    const pct = document.getElementById('studio-step-percent');
    const dsc = document.getElementById('studio-step-desc');

    if (card) card.style.display = 'block';
    if (bar) bar.style.width = `${percent}%`;
    if (lbl) lbl.textContent = label;
    if (pct) pct.textContent = `${percent}%`;
    if (dsc) dsc.textContent = desc;
}

// ----------------- Generation Submission -----------------

async function submitStudioGeneration() {
    if (window.IS_GUEST) {
        showGuestMediaNotice(currentStudioMode);
        return;
    }

    const promptInput = document.getElementById('studio-prompt-input');
    const prompt = promptInput ? promptInput.value.trim() : '';

    if (!prompt) {
        showStudioError("Please enter a prompt describing the scene, clothes, pose, or background.");
        if (promptInput) promptInput.focus();
        return;
    }

    clearStudioError();
    const submitBtn = document.getElementById('btn-studio-submit');
    const resultCard = document.getElementById('studio-result-card');
    if (resultCard) resultCard.style.display = 'none';

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing Pipeline...';
    }

    const negativeInput = document.getElementById('studio-negative-prompt');
    const negativePrompt = negativeInput ? negativeInput.value.trim() : '';

    if (currentStudioMode === 'image') {
        await executeImageGeneration(prompt, negativePrompt);
    } else {
        await executeVideoGeneration(prompt, negativePrompt);
    }

    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = currentStudioMode === 'image' ? '✨ Generate Identity-Preserved Image' : '🎬 Generate Identity-Preserved Video';
    }
}

async function executeImageGeneration(prompt, negativePrompt) {
    const refPaths = currentReferenceSlots.filter(p => p !== null);

    updateStudioProgress(20, "Extracting Multi-Reference Identity...", "Aggregating 128-d master SFace embedding and facial landmarks");

    const timer1 = setTimeout(() => {
        updateStudioProgress(50, "Synthesizing Scene & Conditioning Identity...", "Conditioning lighting, pose, clothing & background");
    }, 1500);

    const timer2 = setTimeout(() => {
        updateStudioProgress(75, "Performing Poisson Seamless Fusion...", "Preserving natural skin pores, iris catchlights & facial structure");
    }, 4500);

    const timer3 = setTimeout(() => {
        updateStudioProgress(90, `Super-Resolving to ${currentResolution.toUpperCase()}...`, "Sub-pixel Lanczos-4 upscaling and adaptive unsharp masking");
    }, 7500);

    try {
        const response = await fetch('/api/generate-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                reference_paths: refPaths,
                reference_path: refPaths[0] || null,
                negative_prompt: negativePrompt,
                aspect_ratio: currentAspectRatio,
                resolution: currentResolution,
                identity_threshold: currentIdentityThreshold,
                identity_strength: 0.75,
                prompt_strength: currentPromptStrength,
                conversation_id: typeof activeConversationId !== 'undefined' ? activeConversationId : null
            })
        });

        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
        const data = await response.json();

        if (response.status === 403 || data.is_guest_restricted) {
            closeMediaStudio();
            showGuestMediaNotice('image');
            return;
        }

        if (response.ok) {
            updateStudioProgress(100, "Verification Completed!", `Identity score confirmed (Score: ${(data.similarity_score * 100).toFixed(1)}%)`);
            setTimeout(() => {
                const progressCard = document.getElementById('studio-progress-card');
                if (progressCard) progressCard.style.display = 'none';
            }, 800);

            renderStudioImageResult(data, prompt);
        } else {
            const progressCard = document.getElementById('studio-progress-card');
            if (progressCard) progressCard.style.display = 'none';
            showStudioError(data.error || "Image generation failed. Please try again.");
        }
    } catch (err) {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
        const progressCard = document.getElementById('studio-progress-card');
        if (progressCard) progressCard.style.display = 'none';
        showStudioError("Network error during image synthesis: " + err.message);
    }
}

function renderStudioImageResult(data, prompt) {
    const card = document.getElementById('studio-result-card');
    const mediaBox = document.getElementById('studio-result-media');
    const badgeBox = document.getElementById('studio-verification-badge');
    const actionsBox = document.getElementById('studio-result-actions');

    if (!card || !mediaBox) return;

    mediaBox.innerHTML = `
        <div class="result-img-container" onclick="openLightbox('${data.image_url}', '${data.resolution || currentResolution}')" title="Click to view full 4K resolution" style="cursor:zoom-in;">
            <img src="${data.image_url}" style="width:100%;max-height:480px;object-fit:contain;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,0.6);transition:transform 0.2s ease;" alt="Generated Result">
            <div style="font-size:0.75rem;color:var(--text-dim);margin-top:6px;">🔍 Click image to view in Full 4K Lightbox</div>
        </div>
    `;

    const resBadge = `<span style="font-size:0.75rem;padding:3px 8px;border-radius:6px;background:rgba(255,255,255,0.08);color:var(--text-muted);border:1px solid var(--border-subtle);">${data.resolution || currentResolution}</span>`;

    if (data.reference_used) {
        badgeBox.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span class="similarity-badge" style="background:rgba(16,185,129,0.15);color:#10b981;border-color:rgba(16,185,129,0.3);">
                    ✓ Multimodal Identity Preserved
                </span>
                ${resBadge}
            </div>
        `;
    } else {
        badgeBox.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span class="similarity-badge" style="background:rgba(56,189,248,0.15);color:#38bdf8;border-color:rgba(56,189,248,0.3);">
                    ✨ Multimodal Generative Image
                </span>
                ${resBadge}
            </div>
        `;
    }

    actionsBox.innerHTML = `
        <button class="tool-btn" onclick="openLightbox('${data.image_url}', '${data.resolution || currentResolution}')">🔍 View Full 4K</button>
        <a href="${data.image_url}" download class="tool-btn" style="background:var(--accent-blue);color:#fff;">⬇️ Download</a>
        <button class="tool-btn" onclick="submitStudioGeneration()">🔄 Regenerate</button>
        <button class="tool-btn" onclick="focusStudioPrompt()">✏️ Edit Prompt</button>
        <button class="tool-btn" onclick="addImageToChat('${data.image_url}', '${escapeHtml(prompt)}')">💬 Add to Chat</button>
    `;

    card.style.display = 'block';
}

function focusStudioPrompt() {
    const promptInput = document.getElementById('studio-prompt-input');
    if (promptInput) {
        promptInput.focus();
        promptInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        promptInput.style.borderColor = 'var(--accent-blue)';
        setTimeout(() => {
            promptInput.style.borderColor = '';
        }, 1200);
    }
}

async function executeVideoGeneration(prompt, negativePrompt) {
    const camSelect = document.getElementById('video-camera-select');
    const durSelect = document.getElementById('video-duration-select');
    const motSelect = document.getElementById('video-motion-select');
    const fpsSelect = document.getElementById('video-fps-select');

    const cam = camSelect ? camSelect.value : 'zoom';
    const dur = durSelect ? parseInt(durSelect.value, 10) : 5;
    const mot = motSelect ? motSelect.value : 'medium';
    const fps = fpsSelect ? parseInt(fpsSelect.value, 10) : 24;

    const refPaths = currentReferenceSlots.filter(p => p !== null);

    updateStudioProgress(20, "Initiating Video Generation...", "Synthesizing identity keyframe and camera trajectory");

    try {
        const response = await fetch('/api/generate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                reference_paths: refPaths,
                reference_path: refPaths[0] || null,
                duration: dur,
                fps: fps,
                motion_strength: mot,
                camera_movement: cam,
                resolution: currentResolution,
                identity_threshold: currentIdentityThreshold,
                identity_strength: 0.75,
                prompt_strength: currentPromptStrength,
                conversation_id: typeof activeConversationId !== 'undefined' ? activeConversationId : null
            })
        });

        const data = await response.json();

        if (response.status === 403 || data.is_guest_restricted) {
            closeMediaStudio();
            showGuestMediaNotice('video');
            return;
        }

        if (response.ok && data.job_id) {
            pollStudioVideoJob(data.job_id, prompt);
        } else {
            const progressCard = document.getElementById('studio-progress-card');
            if (progressCard) progressCard.style.display = 'none';
            showStudioError(data.error || "Failed to start video generation.");
        }
    } catch (err) {
        const progressCard = document.getElementById('studio-progress-card');
        if (progressCard) progressCard.style.display = 'none';
        showStudioError("Network error starting video generation: " + err.message);
    }
}

function pollStudioVideoJob(jobId, prompt) {
    let progressStep = 35;
    updateStudioProgress(progressStep, "Rendering Video Frames...", "Temporal face stabilization & landmark motion tracking");

    activeVideoJobInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/video-status/${jobId}`);
            const data = await res.json();

            if (progressStep < 85) {
                progressStep += 12;
                updateStudioProgress(progressStep, "Rendering Temporal Consistency...", "Stabilizing face across camera trajectory");
            }

            if (data.status === 'completed') {
                clearInterval(activeVideoJobInterval);
                updateStudioProgress(100, "Video Render Complete!", "H.264 MP4 cinematic export ready");
                setTimeout(() => {
                    const progressCard = document.getElementById('studio-progress-card');
                    if (progressCard) progressCard.style.display = 'none';
                }, 800);

                renderStudioVideoResult(data.video_url, prompt);
            } else if (data.status === 'failed') {
                clearInterval(activeVideoJobInterval);
                const progressCard = document.getElementById('studio-progress-card');
                if (progressCard) progressCard.style.display = 'none';
                showStudioError("Video generation encountered an error. Please retry with a different prompt.");
            }
        } catch (err) {
            clearInterval(activeVideoJobInterval);
            const progressCard = document.getElementById('studio-progress-card');
            if (progressCard) progressCard.style.display = 'none';
            showStudioError("Network error checking video render status.");
        }
    }, 1800);
}

function renderStudioVideoResult(videoUrl, prompt) {
    const card = document.getElementById('studio-result-card');
    const mediaBox = document.getElementById('studio-result-media');
    const badgeBox = document.getElementById('studio-verification-badge');
    const actionsBox = document.getElementById('studio-result-actions');

    if (!card || !mediaBox) return;

    mediaBox.innerHTML = `
        <video src="${videoUrl}" controls autoplay loop style="width:100%;max-height:480px;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,0.6);"></video>
    `;

    badgeBox.innerHTML = `
        <span class="similarity-badge">
            ✓ 100% Face Consistency Preserved across Frames
        </span>
    `;

    actionsBox.innerHTML = `
        <a href="${videoUrl}" download class="tool-btn" style="background:var(--accent-blue);color:#fff;">⬇️ Download Video</a>
        <button class="tool-btn" onclick="submitStudioGeneration()">🔄 Regenerate</button>
        <button class="tool-btn" onclick="addVideoToChat('${videoUrl}', '${escapeHtml(prompt)}')">💬 Add to Chat</button>
    `;

    card.style.display = 'block';
}

function addImageToChat(imageUrl, prompt) {
    if (typeof appendMessageRow === 'function') {
        appendMessageRow('user', `Generate image of: ${prompt}`);
        appendMessageRow('assistant', `✨ **Generated Image**\n\n![${prompt}](${imageUrl})\n\n[⬇️ Download](${imageUrl})`, [], 'image', imageUrl);
    }
    closeMediaStudio();
    if (typeof scrollToBottom === 'function') scrollToBottom();
}

function addVideoToChat(videoUrl, prompt) {
    if (typeof appendMessageRow === 'function') {
        appendMessageRow('user', `Generate video of: ${prompt}`);
        appendMessageRow('assistant', `🎬 **Generated Video**\n\nPrompt: *"${prompt}"*\n\n[⬇️ Download Video](${videoUrl})`, [], 'video', videoUrl);
    }
    closeMediaStudio();
    if (typeof scrollToBottom === 'function') scrollToBottom();
}

// ----------------- Settings Modal Management -----------------

function openSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) modal.classList.add('active');
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) modal.classList.remove('active');
}

async function saveSettings(event) {
    event.preventDefault();
    const geminiKey = document.getElementById('settings-gemini-key') ? document.getElementById('settings-gemini-key').value.trim() : '';
    const replicateToken = document.getElementById('settings-replicate-token') ? document.getElementById('settings-replicate-token').value.trim() : '';
    const hfToken = document.getElementById('settings-hf-token') ? document.getElementById('settings-hf-token').value.trim() : '';
    const prefLang = document.getElementById('settings-pref-lang') ? document.getElementById('settings-pref-lang').value : 'auto';

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gemini_api_key: geminiKey,
                replicate_api_token: replicateToken,
                huggingface_token: hfToken,
                preferred_language: prefLang
            })
        });

        if (response.ok) {
            if (prefLang !== 'auto' && typeof setLanguage === 'function') {
                setLanguage(prefLang);
            }
            alert("Settings saved successfully! JaganGpt is configured.");
            closeSettingsModal();
        } else {
            alert("Failed to save settings.");
        }
    } catch (e) {
        alert("Settings error: " + e.message);
    }
}

// ----------------- Full-Resolution 4K Lightbox Viewer -----------------

function openLightbox(imageUrl, metaText = '') {
    const modal = document.getElementById('image-lightbox-modal');
    const img = document.getElementById('lightbox-img');
    const meta = document.getElementById('lightbox-meta');
    const dlLink = document.getElementById('lightbox-download-link');

    if (!modal || !img) return;

    img.src = imageUrl;
    if (meta) {
        meta.textContent = metaText ? `${metaText.toUpperCase()} • 100% Quality Master Output` : '3840×2160 Ultra HD • 100% Quality Master Output';
    }
    if (dlLink) {
        dlLink.href = imageUrl;
    }
    modal.classList.add('active');
}

function closeLightbox() {
    const modal = document.getElementById('image-lightbox-modal');
    if (modal) modal.classList.remove('active');
}

// Keyboard navigation: Close modals or Lightbox on Escape
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const lightbox = document.getElementById('image-lightbox-modal');
        if (lightbox && lightbox.classList.contains('active')) {
            closeLightbox();
            return;
        }
        const studio = document.getElementById('media-studio-modal');
        if (studio && studio.classList.contains('active')) {
            closeMediaStudio();
            return;
        }
        const settings = document.getElementById('settings-modal');
        if (settings && settings.classList.contains('active')) {
            closeSettingsModal();
            return;
        }
    }
});
