# BrowserAI: The Autonomous Web Agent - Developer's Manual

**Version**: 2.0 (LangGraph, RAG Memory Integration)
**Author**: Saqib Sherwani
**Last Updated**: July 11, 2025

---

## 1. Project Overview

BrowserAI is a sophisticated autonomous agent designed to understand high-level human objectives and translate them into browser actions to achieve goals. It moves beyond simple scripting by leveraging a powerful dual-model AI architecture, a state machine for logical reasoning (LangGraph), and a persistent vector store for long-term memory, enabling it to learn from experience.

This document serves as the complete technical guide for developers working on the BrowserAI project. It covers the system architecture, setup, developer guidelines for debugging and adding features, and a strategic roadmap for future enhancements.

---

## 2. Core Features

* **Autonomous Web Navigation**: Understands natural language commands and executes them using a robust, memory-driven, and context-aware architecture.
* **Dual-Model AI Architecture**:
    * **Orchestrator (`gemini-2.5-pro`)**: High-level planner that creates strategies based on rich, descriptive summaries and history.
    * **Specialist (`gemini-2.5-pro`)**: Vision model that now produces multi-step action plans for complex tasks, or single actions for simple ones.
* **Stateful Reasoning Engine (LangGraph)**: The agent's logic is a robust state graph, supporting conditional branching, error handling, and auditable reasoning.
* **Long-Term Memory (RAG, ChromaDB)**: Stores successful plans and can recall and execute them for similar objectives, enabling true learning and repeatability.
* **Intelligent Self-Correction & Resilience**:
    * Retries API calls on server errors (`503`).
    * Rotates API keys on rate-limiting (`429`).
    * Re-analyzes pages with HTML context for improved selector stability.
    * Classifies and recovers from different error types (planned upgrade).
* **Advanced Human-in-the-Loop (HITL) Collaboration**: Proactively requests help for credentials, CAPTCHAs, and ambiguous UI choices.
* **Persistent Browser Session**: Maintains session data for reduced CAPTCHAs and improved continuity.
* **Rich Action Summaries**: After every action, the agent generates a descriptive summary (including the current URL) for improved planning and context-awareness.
* **Fully Functional Memory Execution**: The agent can recall and robustly execute multi-step plans from memory, aborting and reporting if any step fails.

---

## 3. System Architecture

The agent operates as a cyclical graph, managed by LangGraph. Each node in the graph represents a distinct state or capability.

### 3.1. Architectural Diagram (Conceptual Flow)

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

### 3.2. Core Components & Nodes

* **`index.js` (The Main Application)**:
    * Initializes the Express server and WebSocket for frontend communication.
    * Launches the Puppeteer browser instance.
    * Defines and compiles the LangGraph state machine.
    * Manages the main application loop.

* **`prompt-orchestrator.txt` (The Planner's Brain)**:
    * This is the master prompt for the high-level planning AI. It receives the user's objective and a *summary* of recent history and decides which tool to use next (`analyze_screen`, `Google Search`, `request_human_input`, etc.). It does **not** see the screen.

* **`prompt.txt` (The Specialist's Brain)**:
    * This prompt now instructs the Specialist to return a complete, step-by-step plan (array of actions) for complex tasks, or a single action for simple ones. It receives a high-level task, screenshot, and HTML context.

* **LangGraph Nodes (The Agent's Skills)**:
    * **`retrieveMemoryNode`**: Searches ChromaDB for similar, successful plans and enables robust memory recall and execution.
    * **`callOrchestrator`**: High-level planner. Now receives rich, descriptive summaries after every action for improved reasoning.
    * **`callSpecialist`**: Produces multi-step plans or single actions, depending on the task complexity.
    * **`executeActionNode`**: Executes a sequence of actions (plan) or a single action, and generates a descriptive summary including the current URL.
    * **`executeMemoryNode`**: Executes recalled plans from memory, step-by-step, aborting and reporting if any step fails, and provides a final summary.
    * **`humanInputNode`**: Requests input from the user via the web UI.
    * **`rotateKeyNode`**: Manages API key rotation on rate limits.
    * **`saveMemoryNode`**: Saves successful plans to ChromaDB for future recall.

* **ChromaDB (The Long-Term Memory)**:
    * A separate Docker container running a vector database.
    * Stores embeddings of successful `(objective, plan)` pairs.
    * Enables the `retrieveMemoryNode` to perform similarity searches.

---

## 4. Setup and Installation Guide

Follow these steps precisely to ensure a clean and correct setup.

### 4.1. Prerequisites

1.  **Node.js**: Ensure you have Node.js version 22.x or higher installed.
2.  **Docker Desktop**: You **must** have Docker Desktop installed and running on your machine. This is required to run the ChromaDB database. [Download Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 4.2. Initial Project Setup

1.  **Clone/Download**: Place all project files into a single folder.
2.  **Clean Slate (Crucial for Upgrades)**: If you are upgrading or have had previous installation issues, **delete** the following from your project folder:
    * The entire `node_modules` folder.
    * The `package-lock.json` file.
3.  **Configure Environment Variables**:
    * Create a file named `.env` in the root of the project folder.
    * Add your Google Gemini API keys to this file. You must have at least one, but 4-5 are recommended for resilience.
    ```
    GEMINI_API_KEY_1="YOUR_API_KEY_HERE"
    GEMINI_API_KEY_2="YOUR_API_KEY_HERE"
    GEMINI_API_KEY_3="YOUR_API_KEY_HERE"
    GEMINI_API_KEY_4="YOUR_API_KEY_HERE"
    ```

### 4.3. Install Dependencies

* With a clean folder and configured `.env` file, open your terminal in the project directory and run the standard installation command.

    ```bash
    npm install
    ```
    *Note: If you encounter `ERESOLVE` errors, it indicates a deep dependency conflict. The `package.json` in this repository should contain a stable set of versions. If issues persist, running `npm install --force` can override these conflicts.*

### 4.4. Running the Agent

The agent requires **two** separate terminal windows to run.

1.  **Terminal 1: Start the Database**
    * Open a new terminal window.
    * Run the following command to download and start the ChromaDB server.
        ```bash
        docker run -p 8000:8000 chromadb/chroma
        ```
    * You must **leave this terminal window open**. This is the agent's memory.

2.  **Terminal 2: Start the Agent**
    * Open a second terminal window in your project folder.
    * Run the following command to start the main application.
        ```bash
        node index.js
        ```
    * You should see the message: `🚀 Server is ready and listening on http://localhost:3000`.

3.  **Access the UI**:
    * Open your web browser and navigate to `http://localhost:3000`.
    * You can now interact with the agent.

---

## 5. Developer's Guide

This section provides essential information for debugging, maintaining, and extending the agent.

### 5.1. Understanding the Console Logs

The console output is verbose by design. Understanding the emoji prefixes is key to debugging:
* `🚀, ✅`: Server status and successful connections.
* `🔑`: API key rotation events.
* `🧠`: The agent is "thinking." This prefix appears for memory searches and Orchestrator planning.
* `💡`: The Specialist has decided on a specific action.
* `⚡`: The agent is executing a browser action.
* `❌, 🛑`: An error has occurred or a rate limit has been hit.

### 5.2. Debugging a Failed Task: A Step-by-Step Process

1.  **Identify the Point of Failure**: Look at the last `⚡ Executing` or `💡 Specialist Action` log before the error. What was the agent trying to do?
2.  **Check the Orchestrator's Reasoning**: Read the `🗺️ Orchestrator Plan` that led to the failure. Did its reasoning make sense based on the previous step's summary?
3.  **Analyze the Specialist's Decision**: If the failure was at the Specialist level, read its `reasoning`. Why did it choose that specific action or selector? Why did it return an `error`?
4.  **Examine the Raw AI Output**: If the application crashes due to a JSON parsing error, the logs will show `--- RAW ORCHESTRATOR OUTPUT ---`. This is the most critical piece of debugging information. It shows you exactly what the AI sent back, often revealing malformed JSON or unexpected text.
5.  **Look at the Browser**: The Puppeteer window is visible (`headless: false`). Observe the agent's actions in real-time. Does the visual state of the page match what the agent *thinks* it's seeing? This is the best way to spot discrepancies (e.g., the agent thinks it's on a search results page, but it's actually on a CAPTCHA page).

### 5.3. How to Add a New Feature (Example: A `read_pdf` Tool)

1.  **Update the Orchestrator's Prompt**:
    * Open `prompt-orchestrator.txt`.
    * Add `'read_pdf'` to the list of available tools in the `TOOL SCHEMA` section.
    * Add a guideline explaining when to use this tool (e.g., "If the page is a PDF document and the objective is to find information within it, use the `read_pdf` tool.").

2.  **Implement the Tool's Logic in `index.js`**:
    * Create a new async function: `async function readPdfNode(state) { ... }`.
    * Inside this function, add the logic to download the PDF from the current page URL, parse its text content (using a library like `pdf-parse`), and store the extracted text in the `state.summary`.

3.  **Integrate the New Node into the Graph**:
    * In `index.js`, register the new node: `workflow.addNode("read_pdf", readPdfNode);`.
    * Update the Orchestrator's conditional routing (`workflow.addConditionalEdges("orchestrator", ...)`). Add a new condition: `if (tool === 'read_pdf') return "read_pdf";`.
    * Add a new edge to tell the graph where to go *after* the PDF is read: `workflow.addEdge("read_pdf", "orchestrator");`. This sends the extracted text back to the Orchestrator for the next planning step.

---

## 6. Strategic Enhancement Pipeline

This is the official roadmap for future development, prioritized by impact.

* **Phase 1: Foundational Efficiency**:
    1.  **HTML Content Summarization & Targeted Extraction**: Pre-process HTML using a fast model to extract only interactive elements and their key attributes, dramatically reducing token usage and improving selector stability.
    2.  **Action Sequence Specialist**: Specialist now produces multi-step plans for complex tasks, or single actions for simple ones, in a single API call.
    3.  **AI-Powered History Summarization**: Planned upgrade to use an AI model for more narrative, compact summaries of action history.

* **Phase 2: Advanced Reasoning & Resilience**:
    4.  **Dynamic Model Selection**: Planned triage node to route tasks to the most cost-effective AI model.
    5.  **Advanced Error Classification & Correction**: Planned upgrade for error type classification and targeted recovery strategies.
    6.  **Goal-Oriented Memory (RAG V2)**: Planned upgrade to store and recall chains of sub-goals for complex, multi-stage objectives.

* **Phase 3: State-of-the-Art Capabilities**:
    7.  **Multi-Tab/Window Management**: Planned support for multiple browser tabs.
    8.  **Visual Grounding & Ambiguity Resolution**: Planned upgrade for human-in-the-loop ambiguity resolution with screenshots and element highlighting.
    9.  **Cross-Application Capabilities**: Planned extension to control desktop applications beyond the browser.

---

## 7. Contributing

Contributions are welcome. Please open an issue to discuss proposed changes or submit a pull request with a clear description of the enhancement.

## 8. License

This project is licensed under the MIT License.

## 9. Contact

Developed by Saqib Sherwani.
* [GitHub](https://github.com/saqibcodes007)
* [Email](mailto:sherwanisaqib@gmail.com)
