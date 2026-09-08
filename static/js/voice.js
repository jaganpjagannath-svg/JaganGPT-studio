/**
 * JaganGpt - Real-Time Voice Mode Engine
 * Supports Telugu and English speech recognition, visualizer orb, and voice response synthesis.
 */

let voiceRecognition = null;
let isVoiceSessionActive = false;
let isVoiceMuted = false;
let voiceCurrentLang = 'en-US'; // default or 'te-IN'

// Open Dedicated Voice Mode Overlay
function openVoiceMode() {
    const overlay = document.getElementById('voice-mode-overlay');
    if (!overlay) return;

    overlay.classList.add('active');
    isVoiceSessionActive = true;

    // Detect language preference
    const preferredLang = localStorage.getItem('jagangpt_lang') || 'en';
    voiceCurrentLang = (preferredLang === 'te') ? 'te-IN' : 'en-US';

    initSpeechRecognition();
    startListening();
}

function closeVoiceMode() {
    const overlay = document.getElementById('voice-mode-overlay');
    if (overlay) overlay.classList.remove('active');

    isVoiceSessionActive = false;
    stopListening();
    stopSpeaking();
}

// Initialize Speech Recognition
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        setVoiceStatus("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
        return;
    }

    voiceRecognition = new SpeechRecognition();
    voiceRecognition.continuous = false;
    voiceRecognition.interimResults = true;
    voiceRecognition.lang = voiceCurrentLang;

    voiceRecognition.onstart = () => {
        setVoiceOrbState('listening');
        setVoiceStatus(voiceCurrentLang === 'te-IN' ? "Listening (Telugu)..." : "Listening to you...");
    };

    voiceRecognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript;
        }

        const snippet = document.getElementById('voice-transcript-snippet');
        if (snippet) snippet.textContent = `"${transcript}"`;

        if (event.results[0].isFinal) {
            handleVoiceQuery(transcript);
        }
    };

    voiceRecognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            console.warn("Voice error:", event.error);
        }
    };

    voiceRecognition.onend = () => {
        if (isVoiceSessionActive && !isVoiceMuted && document.getElementById('voice-orb').classList.contains('listening')) {
            // Restart listening if still active
            try { voiceRecognition.start(); } catch (e) {}
        }
    };
}

function startListening() {
    if (voiceRecognition && !isVoiceMuted) {
        try {
            voiceRecognition.lang = voiceCurrentLang;
            voiceRecognition.start();
        } catch (e) {}
    }
}

function stopListening() {
    if (voiceRecognition) {
        try { voiceRecognition.stop(); } catch (e) {}
    }
}

// Process Query Sent via Voice
async function handleVoiceQuery(userQuery) {
    if (!userQuery.trim()) return;

    stopListening();
    stopSpeaking(); // User interruption support
    setVoiceOrbState('thinking');
    setVoiceStatus("JaganGpt is thinking...");

    // Also append to main conversation stream in background
    appendMessageRow('user', userQuery);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userQuery,
                conversation_id: activeConversationId,
                model: 'jagangpt-voice'
            })
        });

        const data = await response.json();
        if (response.ok) {
            activeConversationId = data.conversation_id;
            appendMessageRow('assistant', data.reply, [], data.media_type, data.media_url, data.message_id);
            refreshConversationsList();

            // Speak AI response
            speakResponse(data.reply);
        } else {
            speakResponse("Sorry, I could not process your request.");
        }
    } catch (e) {
        speakResponse("There was a connection error.");
    }
}

// Voice Output Synthesis
function speakResponse(text, audioUrl = null) {
    setVoiceOrbState('speaking');
    setVoiceStatus("JaganGpt is speaking...");

    // Clean text of markdown
    const clean = text.replace(/[*#`_]/g, '').substring(0, 350);

    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(clean);

        // Check if text is Telugu
        if (/[\u0C00-\u0C7F]/.test(clean)) {
            utterance.lang = 'te-IN';
        } else {
            utterance.lang = 'en-US';
        }

        utterance.onend = () => {
            if (isVoiceSessionActive) {
                setTimeout(startListening, 600);
            }
        };
        utterance.onerror = () => {
            if (isVoiceSessionActive) startListening();
        };

        window.speechSynthesis.speak(utterance);
    } else {
        // Resume listening if TTS unsupported
        if (isVoiceSessionActive) setTimeout(startListening, 1000);
    }
}

function stopSpeaking() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}

function toggleVoiceMute() {
    isVoiceMuted = !isVoiceMuted;
    const btn = document.getElementById('voice-mute-btn');
    if (isVoiceMuted) {
        stopListening();
        stopSpeaking();
        if (btn) btn.style.color = '#ef4444';
        setVoiceStatus("Muted");
    } else {
        if (btn) btn.style.color = 'inherit';
        startListening();
    }
}

function setVoiceOrbState(state) {
    const orb = document.getElementById('voice-orb');
    if (!orb) return;
    orb.className = `voice-orb ${state}`;
}

function setVoiceStatus(statusText) {
    const statusEl = document.getElementById('voice-status-text');
    if (statusEl) statusEl.textContent = statusText;
}

function toggleVoiceLanguage() {
    voiceCurrentLang = (voiceCurrentLang === 'en-US') ? 'te-IN' : 'en-US';
    const langLabel = document.getElementById('voice-lang-indicator');
    if (langLabel) {
        langLabel.textContent = (voiceCurrentLang === 'te-IN') ? 'Telugu Voice' : 'English Voice';
    }
    stopListening();
    startListening();
}
