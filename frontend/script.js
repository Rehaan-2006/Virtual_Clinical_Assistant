// ==========================================
// CONFIG — change this to your deployed URL
// ==========================================
const API_BASE_URL = "http://localhost:8000";

// ==========================================
// DOM REFS
// ==========================================
const navRag            = document.getElementById('nav-rag');
const navCompare        = document.getElementById('nav-compare');
const windowGeneric     = document.getElementById('window-generic');
const userInput         = document.getElementById('user-input');
const sendBtn           = document.getElementById('send-btn');
const chatHistoryRag    = document.getElementById('chat-history-rag');
const chatHistoryGeneric = document.getElementById('chat-history-generic');
const modelSelect       = document.getElementById('model-select');
const clearBtn          = document.getElementById('clear-btn');
const topbarLabel       = document.getElementById('topbar-mode-label');
const welcomeRag        = document.getElementById('welcome-rag');
const welcomeGeneric    = document.getElementById('welcome-generic');

// ==========================================
// STATE
// ==========================================
let isCompareMode = false;

// ==========================================
// MODE SWITCHING
// ==========================================
navRag.addEventListener('click', () => {
    if (!isCompareMode) return;
    isCompareMode = false;
    windowGeneric.classList.add('hidden');
    navRag.classList.add('active');
    navCompare.classList.remove('active');
    topbarLabel.textContent = 'Evidence-Based Clinical Assistant';
});

navCompare.addEventListener('click', () => {
    if (isCompareMode) return;
    isCompareMode = true;
    windowGeneric.classList.remove('hidden');
    navCompare.classList.add('active');
    navRag.classList.remove('active');
    topbarLabel.textContent = 'Compare Mode — RAG vs Standard LLM';
});

// ==========================================
// CLEAR CONVERSATION
// ==========================================
clearBtn.addEventListener('click', () => {
    // Remove all messages but restore welcome screens
    chatHistoryRag.innerHTML = '';
    chatHistoryGeneric.innerHTML = '';
    if (welcomeRag)     chatHistoryRag.appendChild(welcomeRag);
    if (welcomeGeneric) chatHistoryGeneric.appendChild(welcomeGeneric);
});

// ==========================================
// SUGGESTION CHIPS
// ==========================================
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('chip')) {
        userInput.value = e.target.dataset.query;
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
        userInput.focus();
    }
});

// ==========================================
// AUTO-RESIZE TEXTAREA
// ==========================================
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// ==========================================
// XSS ESCAPE
// ==========================================
function escapeHTML(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// ==========================================
// HIDE WELCOME SCREEN ON FIRST MESSAGE
// ==========================================
function hideWelcome(container) {
    const welcome = container.querySelector('.welcome-screen');
    if (welcome) {
        welcome.style.transition = 'opacity 0.2s';
        welcome.style.opacity = '0';
        setTimeout(() => welcome.remove(), 200);
    }
}

// ==========================================
// CHAT BUBBLE BUILDERS
// ==========================================
function appendUserMessage(text, container) {
    hideWelcome(container);

    const row = document.createElement('div');
    row.className = 'message-row user-row';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user-bubble';
    bubble.textContent = text;

    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

function appendAILoading(container) {
    const row = document.createElement('div');
    row.className = 'message-row ai-row';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble ai-bubble';
    bubble.innerHTML = `
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
    `;

    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;

    return bubble;
}

// ==========================================
// SOURCES UI BUILDER
// ==========================================
function buildSourcesHTML(sources) {
    if (!sources || sources.length === 0) return '';

    let html = `
        <div class="sources-container">
            <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>
                ${sources.length} Source${sources.length > 1 ? 's' : ''} Retrieved
            </button>
            <div class="sources-list hidden">
    `;

    sources.forEach((src, index) => {
        const title   = escapeHTML(src.metadata?.title || `Clinical Document ${index + 1}`);
        const snippet = escapeHTML((src.content || '').substring(0, 160)) + '…';
        html += `
            <div class="source-card">
                <div class="source-meta">${title}</div>
                <div class="source-content">${snippet}</div>
            </div>
        `;
    });

    html += `</div></div>`;
    return html;
}

// ==========================================
// API CALL
// ==========================================
async function fetchAIResponse(endpoint, queryText, modelName, targetBubble) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryText, model: modelName })
        });

        if (!response.ok) {
            let errorDetail = `Server error (${response.status})`;
            try {
                const errData = await response.json();
                if (errData.detail) errorDetail = errData.detail;
            } catch (_) { /* ignore */ }
            throw new Error(errorDetail);
        }

        const data = await response.json();

        let finalHTML = marked.parse(data.response);
        finalHTML += buildSourcesHTML(data.sources);

        targetBubble.innerHTML = finalHTML;

        const history = targetBubble.closest('.chat-history');
        if (history) history.scrollTop = history.scrollHeight;

    } catch (error) {
        console.error("API Error:", error);
        targetBubble.innerHTML = `
            <span class="error-message">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                ${escapeHTML(error.message || 'Could not connect. Is the backend running?')}
            </span>
        `;
    }
}

// ==========================================
// SEND HANDLER
// ==========================================
async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    const selectedModel = modelSelect.value;
    setSendingState(true);

    appendUserMessage(text, chatHistoryRag);
    const ragBubble = appendAILoading(chatHistoryRag);

    let genericBubble = null;
    if (isCompareMode) {
        appendUserMessage(text, chatHistoryGeneric);
        genericBubble = appendAILoading(chatHistoryGeneric);
    }

    userInput.value = '';
    userInput.style.height = 'auto';

    const ragPromise = fetchAIResponse('/api/chat', text, selectedModel, ragBubble);

    if (isCompareMode) {
        await Promise.all([
            ragPromise,
            fetchAIResponse('/api/chat/generic', text, selectedModel, genericBubble)
        ]);
    } else {
        await ragPromise;
    }

    setSendingState(false);
    userInput.focus();
}

function setSendingState(isSending) {
    sendBtn.disabled   = isSending;
    userInput.disabled = isSending;
}

// ==========================================
// EVENT LISTENERS
// ==========================================
sendBtn.addEventListener('click', handleSend);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});
