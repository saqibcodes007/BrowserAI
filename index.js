require('dotenv').config();
const fs = require('fs');
const http = require('http');
const express = require('express');
const { WebSocketServer } = require('ws');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// --- SERVER SETUP ---
const app = express();
app.use(express.static('.')); // Serve files from the current directory
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
// let clientSocket = null; // To hold the single client connection

// --- PUPPETEER & AI SETUP (from your working file) ---
puppeteer.use(StealthPlugin());
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const orchestratorModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
const specialistModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
const orchestratorPrompt = fs.readFileSync('prompt-orchestrator.txt', 'utf-8');
const specialistPrompt = fs.readFileSync('prompt.txt', 'utf-8');

// --- HELPER FUNCTIONS ---
function simplifyHtml(html) {
    let simpleHtml = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    simpleHtml = simpleHtml.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');
    simpleHtml = simpleHtml.replace(/<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>/gi, '');
    simpleHtml = simpleHtml.replace(/<([a-zA-Z0-9]+)\s[^>]*>/g, (match, tag) => {
        const attrs = match.match(/(href|src|alt|value|id|class|name|type|placeholder|aria-label|role|onclick|for)=["']([^"']*)["']/g) || [];
        return `<${tag} ${attrs.join(' ')}>`;
    });
    return simpleHtml.replace(/\s{2,}/g, ' ').trim();
}

async function captureScreenAndDom(page, fullPage = false) {
    const screenshot = await page.screenshot({ encoding: 'base64', fullPage });
    const html = await page.content();
    const simpleHtml = simplifyHtml(html);
    return { screenshot, simpleHtml };
}

// --- COMMUNICATION FUNCTIONS ---
const sendMessageToFrontend = (ws, data) => {
    if (ws && ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify(data));
    }
};

const requestInputFromFrontend = (ws, message, options) => {
    sendMessageToFrontend(ws, { type: 'request_input', message, options });
    return new Promise(resolve => {
        const messageHandler = response => {
            try {
                const data = JSON.parse(response);
                if (data.type === 'user_input') {
                    ws.removeListener('message', messageHandler); // Clean up the listener
                    resolve(data.message);
                }
            } catch (e) {
                console.error("Error parsing user input:", e);
            }
        };
        ws.on('message', messageHandler);
    });
};

// --- AGENT LOGIC FUNCTIONS ---
async function askOrchestrator(objective, history, summary) {
    console.log('🧠 Orchestrator is planning...'); // MOVED TO TERMINAL
    const historyString = history.length > 0 ? history.map((h, i) => `Step ${i + 1}: ${h.reasoning}`).join('\n') : "No actions taken yet.";
    let filledPrompt = orchestratorPrompt.replace('{objective}', objective);
    filledPrompt = filledPrompt.replace('{summary}', summary);
    filledPrompt = filledPrompt.replace('{history}', historyString);
    try {
        const result = await orchestratorModel.generateContent(filledPrompt);
        const cleanedJson = result.response.text().replace(/```json|```/g, '').trim();
        return JSON.parse(cleanedJson);
    } catch (e) {
        return { tool: 'error', reasoning: `Failed to get plan from Orchestrator. Error: ${e.message}` };
    }
}

async function askSpecialist(screenshot, simpleHtml, task, isCorrection) {
    const htmlProvided = !!simpleHtml;
    console.log(`👁️ Specialist is analyzing... (HTML: ${htmlProvided}, Correction: ${isCorrection})`); // MOVED TO TERMINAL
    const correctionInstruction = isCorrection ? "Your previous action failed. You MUST analyze the provided HTML carefully to find a more stable and correct selector. Do not repeat your last action.\n\n" : "";
    let filledPrompt = specialistPrompt.replace('{task}', task);
    const prompt = `${correctionInstruction}${filledPrompt}`;
    const promptParts = [prompt, { inlineData: { data: screenshot, mimeType: 'image/png' } }];
    if (htmlProvided) {
        promptParts.push(`\n\nHere is the simplified HTML of the page:\n\`\`\`html\n${simpleHtml}\n\`\`\``);
    }
    try {
        const result = await specialistModel.generateContent(promptParts);
        const cleanedJson = result.response.text().replace(/```json|```/g, '').trim();
        return JSON.parse(cleanedJson);
    } catch (e) {
        return { action: 'error', reasoning: `Failed to get action from Specialist. Error: ${e.message}` };
    }
}

async function executeAction(ws, page, actionPlan) {
    const { action, selector, text, url, query } = actionPlan;
    console.log(`⚡ Executing: ${action.toUpperCase()}`); // MOVED TO TERMINAL
    try {
        switch (action) {
            case 'type':
                await page.waitForSelector(selector, { visible: true, timeout: 5000 });
                await page.type(selector, text, { delay: 50 });
                break;
            case 'click':
                await page.waitForSelector(selector, { visible: true, timeout: 5000 });
                await page.click(selector);
                break;
            case 'Google Search':
                await page.goto(`https://www.google.com/search?q=${encodeURIComponent(query)}`, { waitUntil: 'networkidle2' });
                break;
            case 'scroll': // CORRECTLY IMPLEMENTED SCROLL
                await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
                break;
            default:
                console.log(`🤷 Action '${action}' is a meta-instruction.`);
        }
        return true;
    } catch (e) {
        const errorMsg = `❌ Action Failed: ${e.message.split('\n')[0]}`;
        console.error(errorMsg); // Log detailed error to terminal
        // Check for specific errors that indicate a page closure
        if (e.message.includes('Target closed') || e.message.includes('detached Frame')) {
             sendMessageToFrontend(ws, { type: 'status', message: "The page seems to have closed. I will try to recover." });
        } else {
             sendMessageToFrontend(ws, { type: 'status', message: "An action failed, trying to self-correct..." }); // User-friendly message
        }
        return false;
    }
}

// THIS IS YOUR EXACT WORKING `runAgentSession` LOGIC, MODIFIED FOR CLEANER UI
async function runAgentSession(ws, objective, browser, page) { // Add browser here
    const context = { credentials: {} };
    const actionHistory = [];
    let currentStateSummary = "Just started. The browser is on a blank page.";
    let loopCount = 0;
    const maxLoops = 20;
    let lastActionFailed = false;

    sendMessageToFrontend(ws, { type: 'status', message: "Okay, starting objective..." });

    while (loopCount < maxLoops) {
        loopCount++;
        
        if (page.isClosed()) {
            console.log('Page was closed, creating a new one.');
            page = await browser.newPage();
            await page.goto('about:blank'); // Start with a clean slate
            currentStateSummary = "Recovered from a page closure. The browser is on a blank page.";
        }

        console.log(`\n--- Loop ${loopCount} ---`);
        const highLevelPlan = await askOrchestrator(objective, actionHistory, currentStateSummary);
        console.log('🗺️ Orchestrator Plan:', highLevelPlan);

        if (highLevelPlan.tool === 'finished') {
            const { screenshot } = await captureScreenAndDom(page, true);
            sendMessageToFrontend(ws, { type: 'final_answer', message: highLevelPlan.answer || "Task completed." }); // Add this line
            break;
        }

        if (highLevelPlan.tool === 'request_credentials') {
            for (const cred of highLevelPlan.credentials_needed) {
                const value = await requestInputFromFrontend(ws, `Please provide the ${cred}:`);
                context.credentials[cred] = value;
            }
            objective += ` (use the provided credentials: ${Object.keys(context.credentials).join(', ')})`;
            currentStateSummary = "Credentials received from user.";
            sendMessageToFrontend(ws, { type: 'status', message: currentStateSummary });
            actionHistory.push({ reasoning: currentStateSummary, success: true });
            continue;
        }

        let actionPlan;
        if (highLevelPlan.tool === 'analyze_screen') {
            const { screenshot, simpleHtml } = await captureScreenAndDom(page);
            actionPlan = await askSpecialist(screenshot, lastActionFailed ? simpleHtml : null, highLevelPlan.task, lastActionFailed);
            if (actionPlan.reasoning === 'HTML_NEEDED') {
                console.log('...Specialist needs more context, re-analyzing with HTML.');
                actionPlan = await askSpecialist(screenshot, simpleHtml, highLevelPlan.task, false);
            }
        } else {
            actionPlan = { ...highLevelPlan, action: highLevelPlan.tool };
        }
        
        if (actionPlan.action === 'type') {
            if (actionPlan.text === '{username}' && context.credentials.username) {
                actionPlan.text = context.credentials.username;
            } else if (actionPlan.text === '{password}' && context.credentials.password) {
                actionPlan.text = context.credentials.password;
            }
        }

        console.log('💡 Specialist Action:', actionPlan);

        if (actionPlan.action === 'captcha' || actionPlan.action === 'request_decision') {
            const userInput = await requestInputFromFrontend(ws, actionPlan.question || "A CAPTCHA has appeared. Please solve it in the browser, then type 'ok'.", actionPlan.options);
            currentStateSummary = `Human provided input: ${userInput}`;
            sendMessageToFrontend(ws, { type: 'status', message: "Resuming..." });
            continue;
        }

        if (actionPlan.action === 'finished') {
             const { screenshot } = await captureScreenAndDom(page, true);
             sendMessageToFrontend(ws, { type: 'final_answer', message: actionPlan.answer || "Task completed.", screenshot });
             break;
        }

        const success = await executeAction(ws, page, actionPlan);
        lastActionFailed = !success;
        
        currentStateSummary = actionPlan.summary || "Action completed.";
        // This is the ONLY summary sent to the frontend during a normal loop
        sendMessageToFrontend(ws, { type: 'status', message: currentStateSummary });
        
        actionHistory.push({ reasoning: highLevelPlan.reasoning, success: success });
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
    if (loopCount >= maxLoops) sendMessageToFrontend(ws, { type: 'status', message: '⚠️ Reached max loop limit.' });
}

// --- MAIN SERVER LOGIC (from your working file) ---
async function main() {
    const app = express();
    app.use(express.static('.'));
    const server = http.createServer(app);
    const wss = new WebSocketServer({ server });

    // Initialize the browser ONCE, outside the connection handler
    const browser = await puppeteer.launch({ headless: false, args: ['--start-maximized'], defaultViewport: null, userDataDir: './my_browser_profile' });

    wss.on('connection', async ws => {
        clientSocket = ws;
        console.log('✅ Frontend connected.');

        // Now 'browser' is accessible here.
        const page = await browser.newPage();
        page.on('dialog', async dialog => dialog.dismiss());

        sendMessageToFrontend(ws, { type: 'greeting', message: 'Hello! I am your AI Browser Agent.' });
        sendMessageToFrontend(ws, { type: 'request_input', message: 'Please provide your first objective.'});

        // This handler is ONLY for the very first objective from the user.
        const initialObjectiveHandler = async (message) => {
            try {
                const data = JSON.parse(message);
                if (data.type === 'user_input') {
                    // Once the first objective is received, remove this listener.
                    // All subsequent 'user_input' messages will be handled by the promise
                    // inside `requestInputFromFrontend`.
                    ws.removeListener('message', initialObjectiveHandler);
                    await runAgentSession(ws, data.message, browser, page);
                }
            } catch (e) {
                console.error("Error processing initial objective:", e);
            }
        };

        ws.on('message', initialObjectiveHandler);

        ws.on('close', async () => {
            console.log('❌ Frontend disconnected.');
            if (page && !page.isClosed()) {
                await page.close(); // Close the page associated with this session
            }
        });
    });

    server.listen(3000, () => {
        console.log('🚀 Server is ready and listening on http://localhost:3000');
        console.log('Please open your Chrome browser and navigate to that address to use the agent.');
    });
}

main();
