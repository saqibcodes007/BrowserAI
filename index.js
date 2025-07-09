require('dotenv').config();
const fs = require('fs');
const http = require('http');
const express = require('express');
const { WebSocketServer } = require('ws');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// --- NEW: LANGGRAPH SETUP ---
const { START, StateGraph } = require("@langchain/langgraph");

/**
 * Represents the state of our agent. This is the central "memory" that
 * will be passed between all the nodes in our graph.
 *
 * @property {string} objective - The initial user request.
 * @property {object} credentials - Any credentials collected from the user.
 * @property {any[]} history - The history of actions taken by the agent.
 * @property {string} summary - The summary of the current page state.
 * @property {boolean} lastActionFailed - Flag to indicate if the last tool call failed.
 * @property {object} nextPage - The next node to execute in the graph.
 */
// NOTE: In a real TypeScript project, this would be an interface.
// We define it as a comment here for clarity in our JavaScript file.
/*
interface AgentState {
    objective: string;
    credentials: { [key: string]: string };
    history: any[];
    summary: string;
    lastActionFailed: boolean;
    nextPage: any;
    needsKeyRotation: boolean; // ADD THIS FOR CLARITY
}
*/

// --- NEW: DEFINE GRAPH NODES ---

/**
 * Calls the Orchestrator model to decide the next high-level tool to use.
 * This node takes the current agent state and returns an updated state with the
 * orchestrator's plan.
 */
async function callOrchestrator(state) {
    const { genAI } = state; // Receive the AI instance from the state
    const orchestratorModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    console.log('🧠 Orchestrator is planning...');
    const { objective, history, summary, credentials } = state;
    const historyString = history.length > 0 ? history.map((h, i) => `Step ${i + 1}: ${JSON.stringify(h)}`).join('\n') : "No actions taken yet.";
    const credentialsString = Object.keys(credentials).length > 0 ? `The following credentials have been provided: [${Object.keys(credentials).join(', ')}]` : "No credentials have been provided.";

    let filledPrompt = orchestratorPrompt
        .replace('{objective}', objective)
        .replace('{summary}', summary)
        .replace('{history}', historyString)
        .replace('{credentials}', credentialsString);

    let rawResult = '';
    try {
        const result = await orchestratorModel.generateContent(filledPrompt);
        rawResult = result.response.text();
        const cleanedJson = rawResult.replace(/```json|```/g, '').trim();
        const plan = JSON.parse(cleanedJson);
        console.log('🗺️ Orchestrator Plan:', plan);
        // This is a crucial change: we now put the orchestrator's plan into the 'actionPlan' field
        // if it's a direct action, bypassing the specialist.
        return { ...state, highLevelPlan: plan, actionPlan: plan };
    } 
    
    catch (e) {
        console.error("Orchestrator failed:", e.message);
        if (e.message && e.message.includes('429')) {
            console.log("🛑 Rate limit hit on Orchestrator. Triggering key rotation.");
            sendMessageToFrontend(state.ws, { type: 'status', message: 'Rate limit reached. Rotating API key...' });
            // --- MODIFIED: Explicitly state which node failed ---
            return { ...state, needsKeyRotation: true, lastFailingNode: "orchestrator" };
        }
        console.error("--- RAW ORCHESTRATOR OUTPUT ---");
        console.error(rawResult);
        console.error("-------------------------------");
        return { ...state, lastActionFailed: true, summary: `Orchestrator failed: ${e.message}` };
    }
}
/**
 * Calls the Specialist model to get a specific browser action based on the
 * orchestrator's task and a screenshot.
 */
async function callSpecialist(state) {
    const { genAI } = state;
    console.log('👁️ Specialist is analyzing...');
    const { highLevelPlan, lastActionFailed, page } = state;
    const { task } = highLevelPlan;

    try {
        const { screenshot, simpleHtml } = await captureScreenAndDom(page);
        const htmlNeeded = lastActionFailed || (highLevelPlan.tool === 'analyze_screen' && task.includes('HTML'));
        
        const specialistResult = await askSpecialist(genAI, screenshot, htmlNeeded ? simpleHtml : null, task, lastActionFailed);
        console.log('💡 Specialist Action:', specialistResult);

        // If the specialist needs HTML but didn't get it, retry with HTML
        if (specialistResult.reasoning === 'HTML_NEEDED' && !htmlNeeded) {
            console.log('...Specialist needs HTML, re-analyzing with it.');
            const retryResult = await askSpecialist(genAI, screenshot, simpleHtml, task, false);
            return { ...state, actionPlan: retryResult, lastActionFailed: false };
        }

        return { ...state, actionPlan: specialistResult, lastActionFailed: false };
    } catch (e) {
        console.error("Specialist failed:", e.message);
        if (e.message && e.message.includes('429')) {
            console.log("🛑 Rate limit hit on Specialist. Triggering key rotation.");
            sendMessageToFrontend(state.ws, { type: 'status', message: 'Rate limit reached. Rotating API key...' });
            // --- MODIFIED: Explicitly state which node failed ---
            return { ...state, needsKeyRotation: true, lastFailingNode: "specialist" };
        }
        return { ...state, lastActionFailed: true, summary: `Specialist failed: ${e.message}` };
    }
}

/**
 * This node is used to parse and store context (like URLs and credentials)
 * found in the initial user objective.
 */
async function updateContextNode(state) {
    const { actionPlan } = state;
    console.log('📝 Updating context with initial information...', actionPlan);

    const newCredentials = actionPlan.credentials || {};
    const summary = actionPlan.url 
        ? `Context updated. Navigating to initial URL: ${actionPlan.url}`
        : `Context updated with provided credentials: ${Object.keys(newCredentials).join(', ')}`;

    // If a URL was found, we will execute a 'goto' action immediately after this.
    // We set the next action directly in the state.
    const nextAction = actionPlan.url ? { action: 'goto', url: actionPlan.url } : null;

    return {
        ...state,
        credentials: { ...state.credentials, ...newCredentials },
        summary: summary,
        // If a URL was parsed, tee up the 'goto' action for the executor
        actionPlan: nextAction || state.actionPlan, 
        history: [...state.history, { context_update: actionPlan }]
    };
}

/**
 * Executes the specific browser action planned by the Specialist.
 * This is where the agent interacts with the Puppeteer page.
 */
async function executeActionNode(state) {
    const { actionPlan, page, ws } = state;

    if (!actionPlan.action) {
        actionPlan.action = actionPlan.tool;
    }

    // The placeholder substitution logic has been removed from here.

    const success = await executeAction(ws, page, actionPlan);
    const summary = actionPlan.summary || (success ? "Action completed successfully." : "Action failed.");
    sendMessageToFrontend(ws, { type: 'status', message: summary });

    return { ...state, summary, lastActionFailed: !success, history: [...state.history, { specialist: actionPlan, success }] };
}

/**
 * A helper node to pause the graph and request input from the human user.
 * This is used for credentials, CAPTCHAs, and decisions.
 */
/**
 * A generalized node to pause the graph and request any necessary input
 * from the human user, based on a question formulated by the Orchestrator.
 */
async function humanInputNode(state) {
    const { highLevelPlan, ws } = state;
    const question = highLevelPlan.question; // The AI formulates the question

    // Request the input from the frontend
    const userInput = await requestInputFromFrontend(ws, question);
    sendMessageToFrontend(ws, { type: 'status', message: `Received user input.` });

    // We update the summary to reflect the human's contribution,
    // which gives the Orchestrator context for its next plan.
    return {
        ...state,
        summary: `Human provided the following input: ${userInput}`,
        history: [...state.history, { human_input: userInput }]
    };
}

/**
 * This node rotates to the next available API key and initializes a new
 * GoogleGenerativeAI client. It's triggered when a 429 error is detected.
 */
async function rotateKeyNode(state) {
    const keyIndex = getNextKeyIndex();
    const currentKey = API_KEYS[keyIndex];
    console.log(`🔑 Rotating to new API Key #${keyIndex + 1}`);
    const newGenAI = new GoogleGenerativeAI(currentKey);

    // Reset the flag and update the AI client in the state
    return {
        ...state,
        genAI: newGenAI,
        needsKeyRotation: false,
        // Crucially, we tell the graph which node failed so we can return to it
        // lastFailingNode: state.nextPage.name
    };
}

// --- SERVER SETUP ---
const app = express();
app.use(express.static('.')); // Serve files from the current directory
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
// let clientSocket = null; // To hold the single client connection

// --- PUPPETEER & AI SETUP (from your working file) ---
puppeteer.use(StealthPlugin());

// --- NEW: API KEY ROTATION SETUP ---
const API_KEYS = [
    process.env.GEMINI_API_KEY_1,
    process.env.GEMINI_API_KEY_2,
    process.env.GEMINI_API_KEY_3,
    process.env.GEMINI_API_KEY_4 // Added the new key
].filter(Boolean); // Filter out any undefined keys

const KEY_STATE_PATH = './key_state.json';

function getNextKeyIndex() {
    try {
        const state = JSON.parse(fs.readFileSync(KEY_STATE_PATH, 'utf-8'));
        const nextIndex = (state.lastKeyIndex + 1) % API_KEYS.length;
        fs.writeFileSync(KEY_STATE_PATH, JSON.stringify({ lastKeyIndex: nextIndex }));
        return nextIndex;
    } catch (e) {
        console.error("Could not read key state, starting from index 0.", e);
        fs.writeFileSync(KEY_STATE_PATH, JSON.stringify({ lastKeyIndex: 0 }));
        return 0;
    }
}

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

async function askSpecialist(genAI, screenshot, simpleHtml, task, isCorrection) {
    const specialistModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
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
            case 'goto':
                await page.goto(url, { waitUntil: 'networkidle2' });
                break;
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



// --- MAIN SERVER LOGIC (from your working file) ---
// --- NEW: BUILD AND COMPILE THE GRAPH ---

const workflow = new StateGraph({
    channels: {
        objective: { value: (x, y) => y, default: () => "" },
        credentials: { value: (x, y) => ({...x, ...y}), default: () => ({}) },
        history: { value: (x, y) => x.concat(y), default: () => [] },
        summary: { value: (x, y) => y, default: () => "No summary yet." },
        lastActionFailed: { value: (x, y) => y, default: () => false },
        highLevelPlan: { value: (x, y) => y, default: () => null },
        actionPlan: { value: (x, y) => y, default: () => null },
        page: { value: (x, y) => y, default: () => null },
        ws: { value: (x, y) => y, default: () => null },
        genAI: { value: (x, y) => y, default: () => null },
        needsKeyRotation: { value: (x, y) => y, default: () => false },
        lastFailingNode: { value: (x, y) => y, default: () => null } // This line is the critical addition
    }
});

// Add the nodes to the graph
workflow.addNode("orchestrator", callOrchestrator);
workflow.addNode("specialist", callSpecialist);
workflow.addNode("executor", executeActionNode);
workflow.addNode("human", humanInputNode);
workflow.addNode("rotate_key", rotateKeyNode);
workflow.addNode("update_context", updateContextNode); // Add this line


// Define the entry point for the graph
workflow.addEdge(START, "orchestrator");

// Define the conditional logic for routing between nodes
workflow.addConditionalEdges("orchestrator", (state) => {
    if (state.needsKeyRotation) return "rotate_key";
    const { tool } = state.highLevelPlan;
    console.log(`Routing based on orchestrator tool: ${tool}`);

    if (tool === 'update_context') return "update_context";
    if (tool === 'request_human_input') return "human"; // Universal route to human
    if (tool === 'finished') return "__end__";
    if (tool === 'analyze_screen') return "specialist";
    if (tool === 'Google Search' || tool === 'goto') return "executor";
    return "__end__";
}, {
    "specialist": "specialist",
    "executor": "executor",
    "human": "human",
    "rotate_key": "rotate_key",
    "update_context": "update_context",
    "__end__": "__end__"
});

workflow.addConditionalEdges("specialist", (state) => {
    // Check if the specialist itself failed with a rate-limit error
    if (state.needsKeyRotation) {
        return "rotate_key";
    }
    if (state.actionPlan.action === 'error') {
        return "orchestrator"; // On regular failure, re-plan
    }
    return "executor";
}, {
    "executor": "executor",
    "orchestrator": "orchestrator",
    "rotate_key": "rotate_key" // Add the new rotation route
});

// This is the new logic for handling recovery after a key rotation
workflow.addConditionalEdges("rotate_key", (state) => {
    // After rotating a key, we check which node failed and route back to it
    // so it can retry the action with the new key.
    const lastFailingNode = state.lastFailingNode;
    console.log(`↩️ Returning to failed node: ${lastFailingNode}`);
    if (lastFailingNode === "orchestrator" || lastFailingNode === "specialist") {
        return lastFailingNode;
    }
    return "end"; // Fallback if the failing node is unknown
}, {
    "orchestrator": "orchestrator",
    "specialist": "specialist",
    "end": "__end__"
});

// This new edge handles the flow after parsing the initial prompt
 workflow.addConditionalEdges("update_context", (state) => {
     // If the update included a URL, the next action is to execute the 'goto' command
     if (state.actionPlan && state.actionPlan.action === 'goto') {
         return "executor";
     }
     // Otherwise, go back to the orchestrator for the next plan
     return "orchestrator";
 }, {
     "executor": "executor",
     "orchestrator": "orchestrator"
 });

// After a successful action or human input, always loop back to the orchestrator
workflow.addEdge("executor", "orchestrator");
workflow.addEdge("human", "orchestrator");

// Compile the graph into a runnable agent
const appGraph = workflow.compile();


// --- MAIN SERVER LOGIC (REVISED FOR LANGGRAPH) ---
async function main() {
    const app = express();
    app.use(express.static('.'));
    const server = http.createServer(app);
    const wss = new WebSocketServer({ server });

    const browser = await puppeteer.launch({ headless: false, args: ['--start-maximized'], defaultViewport: null, userDataDir: './my_browser_profile' });

    wss.on('connection', async ws => {
        console.log('✅ Frontend connected.');

        sendMessageToFrontend(ws, { type: 'greeting', message: 'Hello! I am your AI Browser Agent.' });
        sendMessageToFrontend(ws, { type: 'request_input', message: 'Please provide your objective.'});

        const initialObjectiveHandler = async (message) => {
            try {
                const data = JSON.parse(message);
                if (data.type === 'user_input') {
                    ws.removeListener('message', initialObjectiveHandler);

                                // --- NEW: Perform Key Rotation ---
                    const keyIndex = getNextKeyIndex();
                    const currentKey = API_KEYS[keyIndex];
                    console.log(`🔑 Using API Key #${keyIndex + 1}`);
                    const genAI = new GoogleGenerativeAI(currentKey);
            // --- End of new block ---

                    const page = await browser.newPage();
                    page.on('dialog', async dialog => dialog.dismiss());

                    const initialState = {
                        objective: data.message,
                        page: page,
                        ws: ws,
                        summary: "The browser is on a blank page.",
                        genAI: genAI // Pass the AI instance to the state
                    };

                    // This is where the magic happens: we invoke the graph
                    await appGraph.invoke(initialState);

                    // After the graph finishes, clean up the page
                    if (page && !page.isClosed()) {
                        await page.close();
                    }
                     sendMessageToFrontend(ws, { type: 'request_input', message: 'Please provide your next objective.'});
                     ws.on('message', initialObjectiveHandler); // Re-attach listener for the next task
                }
            } catch (e) {
                console.error("Error processing objective:", e);
                 sendMessageToFrontend(ws, { type: 'status', message: `A critical error occurred: ${e.message}` });
            }
        };

        ws.on('message', initialObjectiveHandler);

        ws.on('close', () => {
            console.log('❌ Frontend disconnected.');
        });
    });

    process.on('SIGINT', async () => {
        console.log("Shutting down browser...");
        await browser.close();
        process.exit(0);
    });

    server.listen(3000, () => {
        console.log('🚀 Server is ready and listening on http://localhost:3000');
    });
}

main();
