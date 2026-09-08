const input = document.getElementById("messageInput");
const messages = document.getElementById("messages");
const sendButton = document.getElementById("sendButton");


async function sendMessage() {

    const text = input.value.trim();

    if (!text) {
        return;
    }

    const welcome = document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }

    // Show user message
    const userMessage = document.createElement("div");

    userMessage.className = "message user-message";

    userMessage.innerHTML = `
        <strong>You</strong><br>
        ${escapeHtml(text)}
    `;

    messages.appendChild(userMessage);

    // Show loading
    const botMessage = document.createElement("div");

    botMessage.className = "message bot-message";

    botMessage.innerHTML = `
        <strong>🤖 JaganGPT</strong><br>
        Thinking...
    `;

    messages.appendChild(botMessage);

    input.value = "";

    messages.scrollTop = messages.scrollHeight;


    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text
            })
        });

        const data = await response.json();

        if (response.ok && data.reply) {

            botMessage.innerHTML = `
                <strong>🤖 JaganGPT</strong><br>
                ${escapeHtml(data.reply)}
            `;

        } else {

            botMessage.innerHTML = `
                <strong>🤖 JaganGPT</strong><br>
                ❌ ${escapeHtml(data.error || "Something went wrong")}
            `;

        }

    } catch (error) {

        console.error(error);

        botMessage.innerHTML = `
            <strong>🤖 JaganGPT</strong><br>
            ❌ Server connection error.
        `;
    }

    messages.scrollTop = messages.scrollHeight;
}


// Enter key
input.addEventListener("keydown", function(event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();
    }

});


// New Chat
function newChat() {

    messages.innerHTML = `
        <div class="welcome">
            <div class="robot">🤖</div>
            <h1>Welcome to JaganGPT</h1>
            <p>Your personal AI assistant. Ask me anything!</p>
        </div>
    `;
}


// Basic HTML protection
function escapeHtml(text) {

    const div = document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}