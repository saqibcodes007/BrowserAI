document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    const initialAgentMessage = "Hello! I am Potentia AI. What task should I perform today?";
    let conversationHistory = [`Assistant: ${initialAgentMessage}`];

    const addMessage = (sender, text) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = sender === 'agent' ? '<i class="fas fa-robot"></i>' : 'U';

        const textBubble = document.createElement('div');
        textBubble.classList.add('text-bubble');
        textBubble.textContent = text;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(textBubble);
        
        if (sender === 'user') {
            messageDiv.removeChild(avatar);
            messageDiv.appendChild(avatar);
        }

        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    // Add the initial agent message to the UI when the page loads
    addMessage('agent', initialAgentMessage);

    const handleSend = async () => {
        const userText = userInput.value.trim();
        if (!userText) return;

        addMessage('user', userText);
        conversationHistory.push(`User: ${userText}`);
        userInput.value = '';
        userInput.style.height = 'auto';

        sendBtn.disabled = true;
        userInput.disabled = true;

        const thinkingMessage = document.createElement('div');
        thinkingMessage.classList.add('message', 'agent-message', 'thinking');
        thinkingMessage.innerHTML = `
            <div class="avatar"><i class="fas fa-robot"></i></div>
            <div class="text-bubble">Thinking...</div>`;
        chatWindow.appendChild(thinkingMessage);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            const response = await fetch('/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history: conversationHistory }),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            
            chatWindow.removeChild(thinkingMessage);

            const agentText = data.response || data.error || "Sorry, I encountered an issue.";
            addMessage('agent', agentText);
            conversationHistory.push(`Assistant: ${agentText}`);

            // Heuristic: If the agent is NOT asking a question, we can consider the sub-task complete
            if ('?' in agentText) {
                console.log('Agent is asking a question, continuing conversation...');
            } else {
                console.log('✅ Task sequence complete. Ready for new task or continuation.');
            }

        } catch (error) {
            console.error('Error fetching from agent:', error);
            chatWindow.removeChild(thinkingMessage);
            addMessage('agent', 'Error: Could not connect to the agent server. Please check if the server is running and try again.');
        } finally {
            sendBtn.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    };

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = `${userInput.scrollHeight}px`;
    });

    // Focus on input field when page loads
    userInput.focus();
});
