# BrowserAI

BrowserAI is an advanced, modular, AI-powered web automation agent that leverages Google Gemini models, Puppeteer, and a LangGraph-based workflow to perform complex browser tasks based on natural language objectives. It features robust error recovery, API key rotation, and a human-in-the-loop design for safe, adaptive automation.

---

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [File-by-File Deep Dive](#file-by-file-deep-dive)
  - [index.js](#indexjs---core-server--agent-logic)
  - [prompt-orchestrator.txt](#prompt-orchestratortxt---orchestrator-prompt)
  - [prompt.txt](#prompttxt---specialist-prompt)
  - [client.js](#clientjs---frontend-logic)
  - [index.html](#indexhtml---frontend-ui)
  - [style.css](#stylecss---styling)
  - [package.json](#packagejson---dependencies)
  - [NOTES.md](#notesmd---project-notes)
- [Setup & Usage](#setup--usage)
- [API Key Management](#api-key-management)
- [Security & Best Practices](#security--best-practices)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features
- **Natural Language Automation**: Users describe objectives in plain English; the agent decomposes and executes them step-by-step.
- **LangGraph Workflow**: Modular, stateful agent logic using a directed graph of nodes (Orchestrator, Specialist, Executor, Human Input, Key Rotation, Context Update).
- **Google Gemini AI**: Uses Gemini 2.5 models for both high-level planning and low-level browser actions.
- **Puppeteer + Stealth**: Automates Chrome with anti-bot detection evasion.
- **API Key Rotation**: Handles rate limits by rotating through multiple Gemini API keys.
- **Human-in-the-Loop**: Requests user input for credentials, CAPTCHAs, or ambiguous decisions.
- **Robust Error Recovery**: Detects and recovers from failed actions, rate limits, and looping failures.
- **WebSocket Frontend**: Real-time chat interface for objectives, status, and agent interaction.

---

## Architecture Overview

```
User <-> Web UI (client.js, index.html) <-> WebSocket <-> Node.js Server (index.js)
                                                        |
                                                        v
                                              LangGraph Agent (Orchestrator, Specialist, etc.)
                                                        |
                                                        v
                                              Puppeteer (Browser Automation)
                                                        |
                                                        v
                                              Google Gemini API (AI Planning/Action)
```

- **Orchestrator**: Plans high-level steps using only the objective, state summary, and action history.
- **Specialist**: Receives a specific task and analyzes screenshots/HTML to generate precise browser actions.
- **Executor**: Runs the action in Puppeteer and updates state.
- **Human Input**: Pauses for user input when needed.
- **Key Rotation**: Handles API rate limits by switching keys.
- **Context Update**: Parses credentials/URLs from the objective.

---

## How It Works
1. **User connects via browser** and submits an objective (e.g., "Log into Gmail and check unread emails").
2. **Server launches a Chrome instance** (with a persistent profile for session continuity).
3. **LangGraph agent** orchestrates the workflow:
   - Orchestrator plans the next tool (analyze screen, Google Search, goto, request input, finish).
   - Specialist receives a screenshot/HTML and outputs a JSON action (click, type, scroll, error).
   - Executor runs the action in Puppeteer.
   - Human Input node requests user input if blocked.
   - Key Rotation node handles API rate limits.
   - Context Update node parses credentials/URLs from the objective.
4. **Frontend updates in real time** with status, screenshots, and input requests.
5. **Loop continues** until the objective is achieved or the user stops.

---

## File-by-File Deep Dive

### index.js — Core Server & Agent Logic
- **Dependencies**: dotenv, fs, http, express, ws, puppeteer-extra (+stealth), @google/generative-ai, @langchain/langgraph.
- **LangGraph State**: Tracks objective, credentials, history, summary, lastActionFailed, nextPage, needsKeyRotation, etc.
- **Nodes**:
  - `callOrchestrator`: Fills the orchestrator prompt, gets a plan from Gemini, handles JSON parsing, and error recovery.
  - `callSpecialist`: Sends screenshot/HTML to Gemini, gets a precise action, retries with HTML if needed.
  - `updateContextNode`: Parses credentials/URLs from the objective and updates state.
  - `executeActionNode`: Runs the planned action in Puppeteer, updates summary/history, and handles errors.
  - `humanInputNode`: Requests input from the frontend/user.
  - `rotateKeyNode`: Rotates Gemini API keys on rate limit errors.
- **Graph Edges**: Conditional routing between nodes based on tool, errors, or completion.
- **API Key Management**: Reads/writes `key_state.json` to persist key rotation state.
- **Prompts**: Loads `prompt-orchestrator.txt` and `prompt.txt` for AI planning/action.
- **Server**: Express serves static files; WebSocket handles real-time communication.
- **Browser Profile**: Uses `my_browser_profile/` for persistent sessions.

### prompt-orchestrator.txt — Orchestrator Prompt
- **Mission**: High-level planner; decomposes objectives into logical steps.
- **Principles**: Sequential decomposition, constant re-evaluation, intelligent navigation, human-in-the-loop, anti-looping, strict completion.
- **Tool Schema**: JSON output specifying next tool (`analyze_screen`, `Google Search`, `goto`, `request_human_input`, `finished`), with reasoning and required fields.
- **Directives**: Always request human input if blocked, never guess, always output valid JSON, recover from failures, avoid infinite loops.

### prompt.txt — Specialist Prompt
- **Mission**: Receives a single task and outputs the most robust browser action as JSON.
- **Rules**: JSON-only output, scope limitation, vision-first, mandatory HTML request if needed, forensic self-correction, no hallucination, strict CSS selector hierarchy, standard CSS only.
- **Action Schema**: JSON with reasoning, action (`type`, `click`, `scroll`, `error`), selector, text, and summary.

### client.js — Frontend Logic
- **WebSocket**: Connects to backend, handles messages (`greeting`, `status`, `request_input`, `final_answer`).
- **UI Updates**: Adds messages to chat log, displays status, enables/disables input as needed.
- **Input Handling**: Sends user input to backend, disables input while agent is working.
- **Screenshot Support**: (Commented out) Can display screenshots from backend.

### index.html — Frontend UI
- **Structure**: Minimal chat interface with message log and input form.
- **Script**: Loads `client.js` for frontend logic.
- **Styling**: Linked to `style.css`.

### style.css — Styling
- **Modern Chat UI**: Responsive, clean, and visually appealing.
- **Message Types**: Distinct styles for agent, user, and status messages.
- **Input Form**: Styled for usability and accessibility.
- **Screenshot Support**: Styles for displaying screenshots.

### package.json — Dependencies
- **Key Packages**:
  - `@google/generative-ai`: Gemini API
  - `@langchain/langgraph`: LangGraph workflow
  - `puppeteer`, `puppeteer-extra`, `puppeteer-extra-plugin-stealth`: Browser automation
  - `express`, `ws`: Server and WebSocket
  - `dotenv`: Environment variable management
- **No test script** (placeholder only).

### NOTES.md — Project Notes
- **Known Issue**: After completing a task, the script closes the Chrome tab. (May affect session continuity for multi-step objectives.)

---

## Setup & Usage

1. **Install dependencies**:
   ```sh
   npm install
   ```
2. **Set up environment variables**:
   - Create a `.env` file with your Gemini API keys:
     ```env
     GEMINI_API_KEY_1=your_key_1
     GEMINI_API_KEY_2=your_key_2
     GEMINI_API_KEY_3=your_key_3
     GEMINI_API_KEY_4=your_key_4
     ```
3. **Run the server**:
   ```sh
   node index.js
   ```
4. **Open your browser** to [http://localhost:3000](http://localhost:3000)
5. **Interact via the chat UI**: Type your objective and follow prompts.

---

## API Key Management
- **Multiple Keys**: Supports up to 4 Gemini API keys for rate limit resilience.
- **Rotation**: On 429 errors, rotates to the next key and persists state in `key_state.json`.
- **Add/Remove Keys**: Edit your `.env` file and restart the server.

---

## Security & Best Practices
- **Never share your API keys** or commit `.env` to version control.
- **Human-in-the-loop**: Agent will always ask for sensitive info (credentials, OTPs) rather than guessing.
- **Error Recovery**: Agent will not loop endlessly; it will try alternative strategies or request human help.
- **Session Isolation**: Each user session uses a separate browser profile for privacy.

---

## Troubleshooting
- **Browser closes after task**: This is a known issue (see `NOTES.md`). For multi-step workflows, consider modifying the cleanup logic.
- **API Rate Limits**: Ensure you have multiple valid Gemini API keys in your `.env`.
- **WebSocket Connection Issues**: Make sure the server is running and accessible at `ws://localhost:3000`.
- **Puppeteer Errors**: Ensure Chrome is installed and accessible, or use Puppeteer's bundled Chromium.

---

## License
This project is licensed under the ISC License. See `package.json` for details.
