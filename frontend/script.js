// --- 1. Grab all the HTML Elements ---
const toggleBtn = document.getElementById('toggle-compare-btn');
const windowGeneric = document.getElementById('window-generic');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatHistoryRag = document.getElementById('chat-history-rag');
const chatHistoryGeneric = document.getElementById('chat-history-generic');
const modelSelect = document.getElementById('model-select');

// --- 2. The Compare Mode Toggle Logic ---
let isCompareMode = false;

toggleBtn.addEventListener('click', () => {
    isCompareMode = !isCompareMode; // Flip the state
    
    if (isCompareMode) {
        windowGeneric.classList.remove('hidden');
        toggleBtn.classList.add('active');
        toggleBtn.textContent = 'Compare Mode: ON';
    } else {
        windowGeneric.classList.add('hidden');
        toggleBtn.classList.remove('active');
        toggleBtn.textContent = 'Compare Mode: OFF';
    }
});

// --- 3. Auto-Resize Textarea ---
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// --- 4. Chat Bubble Generators ---
function appendUserMessage(text, container) {
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
    bubble.innerHTML = '<em>Thinking...</em>'; // Temporary loading text
    
    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
    
    return bubble; 
}

// --- 5. The API Call Logic ---
// This function talks to your Python FastAPI backend
async function fetchAIResponse(endpoint, queryText, modelName, targetBubble) {
    try {
        const response = await fetch(`http://localhost:8000${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Send the query AND the selected model to the backend
            body: JSON.stringify({ query: queryText, model: modelName }) 
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const data = await response.json();

        let finalHTML = marked.parse(data.response);
        // 2. If this is the RAG model and it returned sources, build the UI!
        if (data.sources && data.sources.length > 0) {
            let sourcesHtml = `
                <div class="sources-container">
                    <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
                        View Sources (${data.sources.length}) ▼
                    </button>
                    
                    <div class="sources-list hidden">
            `;
            
            // Create a card for each source chunk
            data.sources.forEach((src, index) => {
                // Fallback to "Document X" if your metadata doesn't have a title yet
                const title = src.metadata.title || src.metadata.source || `Clinical Document ${index + 1}`;
                // Grab the first 150 characters of the chunk as a preview snippet
                const snippet = src.content.substring(0, 150) + "..."; 
                
                sourcesHtml += `
                    <div class="source-card">
                        <div class="source-meta">${title}</div>
                        <div class="source-content">"${snippet}"</div>
                    </div>
                `;
            });
            
            sourcesHtml += `</div></div>`;
            
            // Append the sources directly below the AI's text answer
            finalHTML += sourcesHtml;
        }

        // 3. Render it to the screen
        targetBubble.innerHTML = finalHTML;
        // Use marked.js to convert the AI's markdown response into beautiful HTML
        // targetBubble.innerHTML = marked.parse(data.response);

    } catch (error) {
        console.error("API Error:", error);
        targetBubble.innerHTML = '<span style="color: #d93025;">Error: Could not connect to the AI. Make sure your backend is running!</span>';
    }
}

// --- 6. Handle the "Send" Action ---
async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return; // Do nothing if box is empty
    
    const selectedModel = modelSelect.value; // Get the dropdown value

    // 1. Put user's message in the RAG window and show "Thinking..."
    appendUserMessage(text, chatHistoryRag);
    const ragAiBubble = appendAILoading(chatHistoryRag);

    // 2. If Compare Mode is ON, do the exact same for the Generic window
    let genericAiBubble = null;
    if (isCompareMode) {
        appendUserMessage(text, chatHistoryGeneric);
        genericAiBubble = appendAILoading(chatHistoryGeneric);
    }

    // 3. Clear the input box and reset its height
    userInput.value = '';
    userInput.style.height = 'auto';

    // 4. FIRE THE REAL API REQUESTS!
    const ragPromise = fetchAIResponse('/api/chat', text, selectedModel, ragAiBubble);
    
    // If we are in compare mode, fire the second request simultaneously
    if (isCompareMode) {
        const genericPromise = fetchAIResponse('/api/chat/generic', text, selectedModel, genericAiBubble);
        // Wait for BOTH to finish
        await Promise.all([ragPromise, genericPromise]);
    } else {
        // Just wait for the RAG response
        await ragPromise;
    }
}

// --- 7. Event Listeners for Sending ---
sendBtn.addEventListener('click', handleSend);

userInput.addEventListener('keydown', (e) => {
    // If they hit Enter (but NOT Shift+Enter), send the message
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); 
        handleSend();
    }
});