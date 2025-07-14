require('dotenv').config();
const fs = require('fs');
const http = require('http');
const express = require('express');
const { WebSocketServer } = require('ws');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
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
 * Creates a summarized version of the history to keep token usage low.
 */
function summarizeHistory(history) {
    if (history.length === 0) {
        return "No actions taken yet.";
    }
    const summary = history.slice(-5).map((event, i) => {
        const step = `Step ${history.length - 4 + i}:`;
        if (event.specialist) {
            const { action, reasoning, selector } = event.specialist;
            return `${step} I performed the action '${action}' on selector '${selector}'. My reasoning was: ${reasoning}`;
        } else if (event.context_update) {
            return `${step} I updated the context with new information.`;
        } else if (event.human_input) {
            return `${step} I received the following input from the human: ${event.human_input}`;
        }
        return `${step} An unknown action was taken.`;
    }).join('\n');
    return summary;
}

/**
 * Calls the Orchestrator model to decide the next high-level tool to use.
 * This node takes the current agent state and returns an updated state with the
 * orchestrator's plan.
 */
async function callOrchestrator(state) {
    const { genAI } = state;
    const orchestratorModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
    console.log('🧠 Orchestrator is planning...');
    const { objective, history, summary } = state;
    const historySummary = summarizeHistory(history);

    const filledPrompt = orchestratorPrompt
        .replace('{objective}', objective)
        .replace('{summary}', summary)
        .replace('{history}', historySummary);
    
    let attempt = 0;
    const maxAttempts = 3;

    while (attempt < maxAttempts) {
        let rawResult = '';
        try {
            const result = await orchestratorModel.generateContent(filledPrompt);
            rawResult = await result.response.text();

            const jsonMatch = rawResult.match(/```json\s*([\s\S]*?)\s*```|({[\s\S]*})/);
            if (!jsonMatch) {
                throw new Error("No valid JSON object found in the response.");
            }
            const cleanedJson = jsonMatch[1] || jsonMatch[2];

            const plan = JSON.parse(cleanedJson);
            console.log('🗺️ Orchestrator Plan:', plan);
            sendMessageToFrontend(state.ws, {
                type: 'status',
                message: `🧠 **Plan:** ${plan.reasoning}`
            });
            return { ...state, highLevelPlan: plan, actionPlan: plan };
        } catch (e) {
            console.error(`Orchestrator failed on attempt ${attempt + 1}:`, e.message);
            if (e.message && e.message.includes('503')) {
                attempt++;
                if (attempt < maxAttempts) {
                    const delay = 5000 * attempt; // Wait longer each time
                    console.log(`...model is overloaded. Retrying in ${delay / 1000} seconds...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue; // Go to the next loop iteration
                }
            }
            if (e.message && e.message.includes('429')) {
                console.log("🛑 Rate limit hit on Orchestrator. Triggering key rotation.");
                return { ...state, needsKeyRotation: true, lastFailingNode: "orchestrator" };
            }
            console.error("--- RAW ORCHESTRATOR OUTPUT ---");
            console.error(rawResult);
            console.error("-------------------------------");
            return { ...state, lastActionFailed: true, summary: `Orchestrator failed. Raw output: ${rawResult.substring(0, 100)}...` };
        }
    }
    // If all retries fail
    return { ...state, lastActionFailed: true, summary: "Orchestrator failed after multiple retries due to model overload." };
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

    let attempt = 0;
    const maxAttempts = 3;

    while (attempt < maxAttempts) {
        try {
            const { screenshot, simpleHtml } = await captureScreenAndDom(page);

            // --- NEW: HTML Summarization Step ---
            let contextHtml = null;
            // We only get the HTML summary if it's needed (for a correction)
            if (lastActionFailed) {
                contextHtml = await getHtmlSummary(genAI, simpleHtml);
            }
            // --- End of new block ---

            const specialistResult = await askSpecialist(genAI, screenshot, contextHtml, task, lastActionFailed);
            console.log('💡 Specialist Action:', specialistResult);
            sendMessageToFrontend(state.ws, {
                type: 'status',
                message: `💡 **Action:** ${specialistResult.reasoning}`
            });

            // We now expect the specialist to return a multi-step plan
            return { ...state, highLevelPlan: { ...state.highLevelPlan, plan: specialistResult.plan }, lastActionFailed: false, needsKeyRotation: false };
        } catch (e) {
            console.error(`Specialist failed on attempt ${attempt + 1}:`, e.message);
            if (e.message && e.message.includes('503')) {
                attempt++;
                if (attempt < maxAttempts) {
                    const delay = 5000 * attempt;
                    console.log(`...model is overloaded. Retrying in ${delay / 1000} seconds...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
            }
            if (e.message && e.message.includes('429')) {
                console.log("🛑 Rate limit hit on Specialist. Triggering key rotation.");
                return { ...state, needsKeyRotation: true, lastFailingNode: "specialist" };
            }
            return { ...state, lastActionFailed: true, summary: `Specialist failed: ${e.message}` };
        }
    }
    return { ...state, lastActionFailed: true, summary: "Specialist failed after multiple retries due to model overload." };
}

/**
 * Processes the output of the specialist to update state and history correctly
 * before routing to the next step.
 */
/**
 * Processes non-executable results from the specialist (e.g., data extraction or errors)
 * to update the state history correctly before returning to the orchestrator.
 */
async function processSpecialistResultNode(state) {
    const { actionPlan } = state;
    const { action, data, reasoning } = actionPlan;

    let summary;
    if (action === "extract_data") {
        console.log("✅ Data extracted. Preparing to finish.");
        summary = `I have successfully extracted the following data: ${data}`;
    } else { // This handles the 'error' action
        summary = `The specialist reported an error: ${reasoning}`;
    }

    // We add the event to history and send the updated summary to the orchestrator.
    const historyEvent = { specialist: actionPlan };
    return { ...state, summary: summary, history: [...state.history, historyEvent] };
}



/**
 * Executes the specific browser action planned by the Specialist.
 * This is where the agent interacts with the Puppeteer page.
 */
async function executeActionNode(state) {
    /**
     * Executes a sequence of browser actions planned by the Specialist,
     * or a single action from the Orchestrator, and provides a rich summary.
     */
    const { highLevelPlan, page, ws } = state;
    const plan = highLevelPlan.plan || [state.actionPlan];

    if (!plan || plan.length === 0 || !plan[0]) {
        const errorSummary = "Execution failed: No valid action or plan provided.";
        console.error(`❌ ${errorSummary}`);
        return { ...state, lastActionFailed: true, summary: errorSummary };
    }

    console.log(`▶️ Executing action sequence of ${plan.length} step(s)...`);

    for (const step of plan) {
        if (!step || !step.action) continue;

        try {
            sendMessageToFrontend(ws, { type: 'status', message: `⚡ Executing: ${step.action.toUpperCase()}...` });
            const success = await executeAction(ws, page, step);

            if (!success) {
                const errorSummary = `Action failed: ${step.action} on selector ${step.selector}`;
                console.error(`❌ ${errorSummary}`);
                sendMessageToFrontend(ws, { type: 'status', message: `❌ ${errorSummary}. Aborting plan.` });
                return { ...state, lastActionFailed: true, summary: errorSummary };
            }

            await new Promise(resolve => setTimeout(resolve, 1500));

        } catch (e) {
            const errorSummary = `Critical error during execution of ${step.action}: ${e.message}`;
            console.error(`❌ ${errorSummary}`);
            return { ...state, lastActionFailed: true, summary: errorSummary };
        }
    }

    // --- CRITICAL FIX: Create a descriptive summary of the result ---
    const newUrl = page.url();
    const finalSummary = `✅ Action sequence completed successfully. The browser is now on the page: ${newUrl}`;
    console.log(finalSummary);
    sendMessageToFrontend(ws, { type: 'status', message: finalSummary });
    return { ...state, summary: finalSummary, lastActionFailed: false };
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

    // Reset the rotation flag and update the AI client in the state
    return {
        ...state,
        genAI: newGenAI,
        needsKeyRotation: false,
        // We pass the lastFailingNode to the next state, so the graph knows where to go.
        lastFailingNode: state.lastFailingNode 
    };
}

/**
 * This node retrieves a relevant, successful plan from the vector store
 * if a similar objective has been completed in the past.
 */
async function retrieveMemoryNode(state) {
    const { objective } = state;
    console.log(`🧠 Searching for memories related to: "${objective}"`);
    sendMessageToFrontend(state.ws, { type: 'status', message: 'Searching for similar tasks in my memory...' });

    try {
        const relevantMemories = await vectorStore.similaritySearch(objective, 1, {
            scoreThreshold: 0.75 // This is the corrected, more tolerant threshold
        });

        if (relevantMemories && relevantMemories.length > 0) {
            const recalledPlan = JSON.parse(relevantMemories[0].metadata.plan); // We need to parse the recalled plan
            console.log('✅ Memory recalled. Executing stored plan.');
            sendMessageToFrontend(state.ws, { type: 'status', message: 'Found a similar completed task! Executing the stored plan.' });
            // We will now execute the recalled plan step-by-step
            return { ...state, highLevelPlan: { tool: 'execute_memory', plan: recalledPlan } };
        } else {
            console.log('❌ No relevant memories found.');
            sendMessageToFrontend(state.ws, { type: 'status', message: 'This is a new task. I will create a new plan.' });
            return { ...state, highLevelPlan: { tool: 'orchestrator' } };
        }
    } catch (e) {
        console.error("Error retrieving memory:", e);
        return { ...state, highLevelPlan: { tool: 'orchestrator' } };
    }
}

/**
 * Executes a recalled plan from memory, step by step.
 */
async function executeMemoryNode(state) {
    /**
     * Executes a recalled plan from memory, step by step.
     */
    const { plan } = state.highLevelPlan;
    const { page, ws } = state;
    console.log("▶️ Executing recalled plan from memory...");
    sendMessageToFrontend(ws, { type: 'status', message: '▶️ Memory recalled. Executing stored plan...' });

    // Loop through each step of the recalled plan
    for (const step of plan) {
        // --- THIS IS THE REAL EXECUTION LOGIC ---
        try {
            if (!step || !step.action) continue;

            // We use the same robust executeAction function
            console.log(`Executing from memory: ${step.action}`);
            sendMessageToFrontend(ws, { type: 'status', message: `⚡ Memory: ${step.reasoning || step.action}` });
            const success = await executeAction(ws, page, step);

            if (!success) {
                const errorSummary = "A step in the recalled plan failed. Aborting memory execution.";
                console.error(`❌ ${errorSummary}`);
                sendMessageToFrontend(ws, { type: 'status', message: `❌ ${errorSummary}` });
                return { ...state, lastActionFailed: true, summary: errorSummary };
            }

            await new Promise(resolve => setTimeout(resolve, 2000));

        } catch (e) {
            const errorSummary = `Critical error during memory execution: ${e.message}`;
            console.error(`❌ ${errorSummary}`);
            return { ...state, lastActionFailed: true, summary: errorSummary };
        }
    }

    // --- Provide a final, descriptive summary ---
    const newUrl = page.url();
    const finalSummary = `✅ Recalled plan executed successfully. The browser is now on the page: ${newUrl}`;
    console.log(finalSummary);
    sendMessageToFrontend(ws, { type: 'status', message: finalSummary });
    return { ...state, summary: finalSummary, lastActionFailed: false };
}

/**
 * This node saves the successful plan and the objective to the vector store
 * for future recall.
 */
async function saveMemoryNode(state) {
    const { objective, history, highLevelPlan } = state;
    console.log('💾 Saving successful plan to memory...');

    // --- NEW: Create a clean, simplified plan for storage ---
    const simplifiedPlan = history
        .filter(event => event.specialist || event.human_input) // Filter for meaningful events
        .map(event => {
            if (event.specialist) {
                const { action, selector, text, reasoning } = event.specialist;
                return { action, selector, text, reasoning }; // Capture the core action details
            }
            return { action: 'human_input', text: event.human_input };
        });

    const memoryDocument = {
        pageContent: objective,
        metadata: {
            objective: objective,
            // We now save the clean, stringified plan
            plan: JSON.stringify(simplifiedPlan) 
        }
    };

    try {
        await vectorStore.addDocuments([memoryDocument]);
        console.log('✅ Memory saved successfully.');
    } catch (e) {
        console.error("Error saving memory:", e);
    }

    // --- NEW: Send the final answer to the UI ---
    sendMessageToFrontend(state.ws, {
        type: 'final_answer',
        message: highLevelPlan.answer
    });

    return { ...state, highLevelPlan: { tool: 'finished' } };
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
const summarizerPrompt = fs.readFileSync('prompt-summarizer.txt', 'utf-8');
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "embedding-001", // A specialized model for creating numerical representations
    apiKey: API_KEYS[0] // We can use any of your keys for this
});

const vectorStore = new Chroma(embeddings, {
    collectionName: "agent_memories",
    url: "http://localhost:8000" // This is the default address for a local ChromaDB instance
});
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

/**
 * Uses a fast model to summarize the HTML, extracting only interactive elements.
 */
async function getHtmlSummary(genAI, html) {
    console.log('📄 Summarizing HTML...');
    try {
        // Use a fast and cheap model for this summarization task
        const summarizerModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const prompt = summarizerPrompt.replace('{html_content}', html);
        const result = await summarizerModel.generateContent(prompt);
        const summary = await result.response.text();
        console.log('✅ HTML summarized successfully.');
        return summary;
    } catch (e) {
        console.error("Error summarizing HTML:", e.message);
        // If summarization fails, return the simplified HTML as a fallback
        return simplifyHtml(html); 
    }
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
        lastFailingNode: { value: (x, y) => y, default: () => null }, // This line is the critical addition
    }
});

// Add the nodes to the graph
workflow.addNode("orchestrator", callOrchestrator);
workflow.addNode("specialist", callSpecialist);
workflow.addNode("process_result", processSpecialistResultNode);
workflow.addEdge("process_result", "orchestrator");
workflow.addNode("executor", executeActionNode);
workflow.addNode("human", humanInputNode);
workflow.addNode("rotate_key", rotateKeyNode);
workflow.addNode("retrieve_memory", retrieveMemoryNode);
workflow.addNode("save_memory", saveMemoryNode);
workflow.addEdge(START, "retrieve_memory");
workflow.addNode("execute_memory", executeMemoryNode);
workflow.addEdge("execute_memory", "orchestrator");
workflow.addConditionalEdges("specialist", (state) => {
    if (state.needsKeyRotation) {
        return "rotate_key";
    }

    // The specialist now returns a 'plan'. If the plan is empty, it's an error/obstacle.
    if (!state.highLevelPlan.plan || state.highLevelPlan.plan.length === 0) {
        return "process_result"; // Go to processing node to handle the obstacle
    }

    // If a plan exists, execute it.
    return "executor";
}, {
    "rotate_key": "rotate_key",
    "process_result": "process_result",
    "executor": "executor"
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

// After a successful action or human input, always loop back to the orchestrator
workflow.addEdge("executor", "orchestrator");
workflow.addEdge("human", "orchestrator");

// This is the new logic for memory-based routing
workflow.addConditionalEdges("retrieve_memory", (state) => {
    const { tool } = state.highLevelPlan;
    if (tool === 'execute_memory') {
        return "execute_memory"; // Route to our new executor
    }
    return "orchestrator";
}, {
    "execute_memory": "execute_memory",
    "orchestrator": "orchestrator"
});

// When the orchestrator decides the task is 'finished', we now route to save the memory
workflow.addConditionalEdges(
  "orchestrator",
  (state) => {
    if (state.needsKeyRotation) return "rotate_key";
    const { tool } = state.highLevelPlan;
    console.log(`Routing based on orchestrator tool: ${tool}`);

    if (tool === "finished") return "save_memory";
    // if (tool === "update_context") return "update_context";
    if (tool === "request_human_input") return "human";
    if (tool === "analyze_screen") return "specialist";
    if (tool === "Google Search" || tool === "goto") {
      // Pass the specific action to the executor node
      state.actionPlan = state.highLevelPlan;
      return "executor";
    }
    return "__end__";
  },
  {
    specialist: "specialist",
    executor: "executor",
    human: "human",
    rotate_key: "rotate_key",
    save_memory: "save_memory",
    // update_context: "update_context",
    __end__: "__end__",
  }
);

// After saving, the process is complete
workflow.addEdge("save_memory", "__end__");

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

        // --- NEW: Create ONE persistent page for the entire session ---
        const page = await browser.newPage();
        page.on('dialog', async dialog => dialog.dismiss());
        console.log('📄 New persistent page created for session.');

        sendMessageToFrontend(ws, { type: 'greeting', message: 'Hello! I am your AI Browser Agent. A persistent session has started.' });
        sendMessageToFrontend(ws, { type: 'request_input', message: 'Please provide your first objective.'});

        const objectiveHandler = async (message) => {
            try {
                const data = JSON.parse(message);
                if (data.type === 'user_input') {
                    ws.removeListener('message', objectiveHandler); // Remove listener to avoid duplicates

                    const keyIndex = getNextKeyIndex();
                    const currentKey = API_KEYS[keyIndex];
                    console.log(`🔑 Using API Key #${keyIndex + 1}`);
                    const genAI = new GoogleGenerativeAI(currentKey);

                    // --- NEW: Get current page state before starting a new task ---
                    const currentUrl = page.url();
                    const { screenshot } = await captureScreenAndDom(page); // Get a screenshot for context

                    // The 'summary' now reflects the REAL current state of the persistent page
                    const initialState = {
                        objective: data.message,
                        page: page, // Use the single, persistent page
                        ws: ws,
                        summary: `The browser is currently on the page: ${currentUrl}. The user has provided a new objective.`,
                        genAI: genAI,
                        // We start with a fresh history for each main objective for clarity
                        history: [], 
                        lastActionFailed: false,
                        needsKeyRotation: false
                    };

                    sendMessageToFrontend(ws, {type: 'status', message: `Objective received. Current page is ${currentUrl}. Starting task...`, screenshot: screenshot});

                    await appGraph.invoke(initialState, { recursionLimit: 100 });

                    // --- The page.close() call has been REMOVED ---

                    // After the task is done, get the final state of the page for the next prompt
                    const finalUrl = page.url();
                    sendMessageToFrontend(ws, { type: 'request_input', message: `Task complete. I am currently on ${finalUrl}. Please provide your next objective.`});
                    ws.on('message', objectiveHandler); // Re-attach listener for the next task
                }
            } catch (e) {
                console.error("Error processing objective:", e);
                sendMessageToFrontend(ws, { type: 'status', message: `A critical error occurred: ${e.message}` });
                // In case of error, re-attach listener so user can try again
                ws.on('message', objectiveHandler);
            }
        };

        ws.on('message', objectiveHandler);

        // --- NEW: Clean up the persistent page when the user disconnects ---
        ws.on('close', async () => {
            console.log('❌ Frontend disconnected. Closing persistent page.');
            if (page && !page.isClosed()) {
                await page.close();
            }
        });
    });

    process.on('SIGINT', async () => {
        console.log("\n🛑 Script terminated. The browser will remain open.");
        // The browser.close() command has been removed to ensure persistence.
        process.exit(0);
    });

    server.listen(3000, () => {
        console.log('🚀 Server is ready and listening on http://localhost:3000');
    });
}

main();
