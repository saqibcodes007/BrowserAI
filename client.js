const messageLog = document.getElementById('message-log');
const inputForm = document.getElementById('input-form');
const userInput = document.getElementById('user-input');
const sendButton = inputForm.querySelector('button');

const socket = new WebSocket('ws://localhost:3000');

function addMessage(data, sender) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add(data.type || 'message');
    messageContainer.classList.add(sender);

    // Sanitize and display the message
    const messageText = document.createElement('div');
    messageText.textContent = data.message;
    messageContainer.appendChild(messageText);

    // If there's a screenshot, display it
    if (data.screenshot) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${data.screenshot}`;
        img.classList.add('screenshot');
        messageContainer.appendChild(img);
    }
    
    messageLog.appendChild(messageContainer);
    messageLog.scrollTop = messageLog.scrollHeight;
}

function enableInput(placeholder = "Type your response...") {
    userInput.disabled = false;
    sendButton.disabled = false;
    userInput.placeholder = placeholder;
    userInput.focus();
}

socket.onopen = () => console.log('Connected to agent backend.');

socket.onmessage = event => {
    const data = JSON.parse(event.data);
    
    switch (data.type) {
        case 'greeting':
            addMessage(data, 'agent');
            break;
        case 'status':
            addMessage(data, 'agent');
            break;
        case 'request_input':
            addMessage(data, 'agent');
            if (data.options) {
                const optionsText = "Please respond with one of the following: " + data.options.join(', ');
                addMessage({ message: optionsText, type: 'status'}, 'agent');
            }
            enableInput();
            break;
        case 'final_answer':
            addMessage({ message: 'Objective complete!', type: 'status' }, 'agent');
            addMessage(data, 'agent');
            enableInput("Type your next objective...");
            break;
        default:
            addMessage(data, 'agent');
    }
};

inputForm.addEventListener('submit', event => {
    event.preventDefault();
    const message = userInput.value;
    if (!message) return;

    addMessage({ message: message }, 'user');
    socket.send(JSON.stringify({ type: 'user_input', message: message }));
    
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;
    userInput.placeholder = "Agent is working...";
});