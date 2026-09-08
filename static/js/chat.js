/**
 * JaganGpt - Core Chat Interactions Client
 * Handles messaging, markdown rendering, code block highlighting, and attachments.
 */

let activeConversationId = null;
let currentAttachments = [];

// Send Message Handler
async function handleSendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message && currentAttachments.length === 0) return;

    input.value = '';
    adjustTextareaHeight(input);

    const attachmentsToSend = [...currentAttachments];
    clearAttachments();

    // Hide welcome screen if present
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'none';

    // Append User Message to UI
    appendMessageRow('user', message, attachmentsToSend);

    // Append Assistant Loading Indicator
    const loadingRow = appendAssistantLoading();
    scrollToBottom();

    try {
        const modelSelect = document.getElementById('model-select');
        const selectedModel = modelSelect ? modelSelect.value : 'jagangpt-smart';

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_id: activeConversationId,
                model: selectedModel,
                attachments: attachmentsToSend
            })
        });

        const data = await response.json();
        loadingRow.remove();

        if (response.ok) {
            activeConversationId = data.conversation_id;
            appendMessageRow('assistant', data.reply, [], data.media_type, data.media_url, data.message_id);
            refreshConversationsList();
        } else {
            appendMessageRow('assistant', `⚠️ **Error**: ${data.error || 'Failed to generate response'}`);
        }
    } catch (err) {
        loadingRow.remove();
        appendMessageRow('assistant', `⚠️ **Connection error**. Please check your network connection and try again.`);
    }

    scrollToBottom();
}

// Append Message Row to UI
function appendMessageRow(role, content, attachments = [], mediaType = 'text', mediaUrl = null, messageId = null) {
    const stream = document.getElementById('messages-stream-inner');
    if (!stream) return;

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    let attachmentsHtml = '';
    if (attachments && attachments.length > 0) {
        attachmentsHtml = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;">';
        attachments.forEach(att => {
            attachmentsHtml += `
                <div style="font-size:0.75rem;padding:3px 8px;border-radius:6px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);">
                    📎 ${escapeHtml(att.original_name)}
                </div>
            `;
        });
        attachmentsHtml += '</div>';
    }

    let parsedContent = renderMarkdown(content);

    if (role === 'user') {
        row.innerHTML = `
            <div class="message-bubble">
                ${attachmentsHtml}
                <div>${parsedContent}</div>
            </div>
        `;
    } else {
        const msgIdAttr = messageId ? `data-message-id="${messageId}"` : '';
        row.innerHTML = `
            <img src="/static/images/logo.svg" class="avatar-assistant" alt="JaganGpt">
            <div class="message-bubble" ${msgIdAttr}>
                <div class="msg-text-content">${parsedContent}</div>
                <div class="message-actions-toolbar">
                    <button class="msg-act-btn" onclick="rateMessage(${messageId}, 1, this)" title="Good response">👍</button>
                    <button class="msg-act-btn" onclick="rateMessage(${messageId}, -1, this)" title="Bad response">👎</button>
                    <button class="msg-act-btn" onclick="copyMessageText(this)" title="Copy text">📋 Copy</button>
                    <button class="msg-act-btn" onclick="readAloud(this)" title="Read Aloud">🔊 Speak</button>
                </div>
            </div>
        `;
    }

    stream.appendChild(row);
}

// Assistant Loading Row
function appendAssistantLoading() {
    const stream = document.getElementById('messages-stream-inner');
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
        <img src="/static/images/logo.svg" class="avatar-assistant" alt="JaganGpt">
        <div class="message-bubble" style="color:var(--text-dim);display:flex;align-items:center;gap:8px;">
            <span>Thinking</span>
            <span class="loading-dots">...</span>
        </div>
    `;
    stream.appendChild(row);
    return row;
}

// Simple Markdown Renderer
function renderMarkdown(text) {
    if (!text) return '';

    // Code blocks with language and copy button
    text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const cleanLang = lang || 'code';
        return `
            <div class="code-block-wrapper">
                <div class="code-header">
                    <span>${cleanLang}</span>
                    <button class="btn-copy-code" onclick="copyCodeSnippet(this)">Copy Code</button>
                </div>
                <pre><code>${escapeHtml(code.trim())}</code></pre>
            </div>
        `;
    });

    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 5px;border-radius:4px;font-family:monospace;color:#38bdf8;">$1</code>');

    // Images: ![alt](url)
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:12px;margin:8px 0;display:block;">');

    // Links: [label](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:#38bdf8;text-decoration:underline;">$1</a>');

    // Headings
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold and Italics
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Blockquotes
    text = text.replace(/^\> (.*$)/gim, '<blockquote style="border-left:3px solid var(--accent-blue);padding-left:12px;color:var(--text-dim);margin:8px 0;">$1</blockquote>');

    // Line breaks
    text = text.replace(/\n/g, '<br>');

    return text;
}

// Copy Code Helper
function copyCodeSnippet(btn) {
    const wrapper = btn.closest('.code-block-wrapper');
    const code = wrapper.querySelector('pre code').innerText;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy Code', 2000);
    });
}

// Copy Message Text
function copyMessageText(btn) {
    const bubble = btn.closest('.message-bubble');
    const textEl = bubble.querySelector('.msg-text-content');
    navigator.clipboard.writeText(textEl.innerText).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = '📋 Copy', 2000);
    });
}

// Read Aloud Text-to-Speech
function readAloud(btn) {
    const bubble = btn.closest('.message-bubble');
    const text = bubble.querySelector('.msg-text-content').innerText;

    if (!window.speechSynthesis) {
        alert("Speech synthesis is not supported on this browser.");
        return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.substring(0, 400));
    
    // Check if Telugu text
    if (/[\u0C00-\u0C7F]/.test(text)) {
        utterance.lang = 'te-IN';
    } else {
        utterance.lang = 'en-US';
    }

    btn.textContent = '🔊 Playing...';
    utterance.onend = () => { btn.textContent = '🔊 Speak'; };
    utterance.onerror = () => { btn.textContent = '🔊 Speak'; };
    window.speechSynthesis.speak(utterance);
}

// Thumbs Up / Down Rating
async function rateMessage(messageId, rating, btn) {
    if (!messageId) return;
    try {
        await fetch('/api/message/rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message_id: messageId, rating: rating })
        });
        btn.style.transform = 'scale(1.2)';
        btn.style.filter = 'brightness(1.5)';
    } catch (e) {
        console.error(e);
    }
}

// File Attachment Handling
async function handleFileUpload(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    if (activeConversationId) {
        formData.append('conversation_id', activeConversationId);
    }

    // Show temporary uploading chip
    const previewRow = document.getElementById('attachments-preview-row');
    const tempChip = document.createElement('div');
    tempChip.className = 'attachment-chip';
    tempChip.textContent = `Uploading ${file.name}...`;
    previewRow.appendChild(tempChip);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        tempChip.remove();

        if (response.ok) {
            currentAttachments.push(data);
            renderAttachmentChip(data);
        } else {
            alert(data.error || "File upload failed");
        }
    } catch (err) {
        tempChip.remove();
        alert("File upload error");
    }
}

function renderAttachmentChip(att) {
    const previewRow = document.getElementById('attachments-preview-row');
    const chip = document.createElement('div');
    chip.className = 'attachment-chip';
    chip.innerHTML = `
        <span>📄 ${escapeHtml(att.original_name)}</span>
        <span class="attachment-chip-remove" onclick="removeAttachment('${att.stored_name}', this)">✕</span>
    `;
    previewRow.appendChild(chip);
}

function removeAttachment(storedName, el) {
    currentAttachments = currentAttachments.filter(a => a.stored_name !== storedName);
    el.closest('.attachment-chip').remove();
}

function clearAttachments() {
    currentAttachments = [];
    const previewRow = document.getElementById('attachments-preview-row');
    if (previewRow) previewRow.innerHTML = '';
}

// Conversation Management
async function loadConversation(convId) {
    activeConversationId = convId;
    clearAttachments();

    try {
        const response = await fetch(`/api/conversations/${convId}`);
        const data = await response.json();

        // Clear welcome screen
        const welcome = document.getElementById('welcome-screen');
        if (welcome) welcome.style.display = 'none';

        const stream = document.getElementById('messages-stream-inner');
        stream.innerHTML = '';

        data.messages.forEach(msg => {
            const atts = JSON.parse(msg.attachments_json || '[]');
            appendMessageRow(msg.role, msg.content, atts, msg.media_type, msg.media_url, msg.id);
        });

        highlightActiveConversationInSidebar(convId);
        scrollToBottom();
    } catch (err) {
        console.error("Failed to load conversation:", err);
    }
}

function startNewChat() {
    activeConversationId = null;
    clearAttachments();

    const stream = document.getElementById('messages-stream-inner');
    if (stream) stream.innerHTML = '';

    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'flex';

    document.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
}

async function refreshConversationsList() {
    try {
        const res = await fetch('/api/conversations');
        const data = await res.json();
        const list = document.getElementById('conversations-list');
        if (!list) return;

        list.innerHTML = '';
        data.conversations.forEach(c => {
            const item = document.createElement('div');
            item.className = `conv-item ${c.id === activeConversationId ? 'active' : ''}`;
            item.setAttribute('data-conv-id', c.id);
            item.onclick = (e) => {
                if (!e.target.closest('.conv-actions')) loadConversation(c.id);
            };

            item.innerHTML = `
                <span class="conv-title">${escapeHtml(c.title)}</span>
                <div class="conv-actions">
                    <button class="conv-act-btn" onclick="deleteConversation(${c.id}, event)" title="Delete">🗑️</button>
                </div>
            `;
            list.appendChild(item);
        });
    } catch (e) {
        console.error("Refresh conversations error", e);
    }
}

async function deleteConversation(convId, event) {
    event.stopPropagation();
    if (!confirm("Delete this conversation?")) return;

    try {
        await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
        if (activeConversationId === convId) startNewChat();
        refreshConversationsList();
    } catch (e) {
        console.error(e);
    }
}

function highlightActiveConversationInSidebar(convId) {
    document.querySelectorAll('.conv-item').forEach(el => {
        if (parseInt(el.getAttribute('data-conv-id')) === convId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

function scrollToBottom() {
    const stream = document.getElementById('messages-stream');
    if (stream) stream.scrollTop = stream.scrollHeight;
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
}

function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, s => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[s]));
}

// Quick Prompt click
function insertQuickPrompt(promptText) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = promptText;
        handleSendMessage();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
        chatInput.addEventListener('input', () => adjustTextareaHeight(chatInput));
    }

    const fileInput = document.getElementById('file-upload-input');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }
});
