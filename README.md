# BrowserAI - The AI Browser Agent

## 🚀 Introduction

Welcome to the BrowserAI! This is a sophisticated autonomous agent designed to understand high-level human objectives and translate them into browser actions to achieve goals. It leverages a powerful dual-model AI architecture to navigate websites, interact with elements, handle errors, and retrieve information.

This agent moves beyond simple scripts by incorporating a **persistent browser session** (to reduce CAPTCHAs), a **self-correction mechanism** for failed actions, and an **interactive chat UI** for seamless user collaboration.

---

## ✨ Core Features

* **Autonomous Web Navigation:** Understands natural language commands (e.g., "find the address of X on their website") and executes them.
* **Dual-Model AI Architecture:**
    * **Orchestrator (Gemini 2.5 Flash):** A fast, lightweight planner that creates high-level strategies.
    * **Specialist (Gemini 2.5 Pro):** A powerful vision model for detailed screen analysis and interaction.
* **Self-Correction Loop:** Automatically detects when an action fails (e.g., a button isn't found) and re-analyzes the page with more context (including HTML) to find a better solution.
* **Interactive Chat UI:** A clean, web-based interface for giving the agent tasks and receiving real-time summaries and final answers.
* **Human-in-the-Loop (HITL):** The agent will intelligently pause and ask for your help via the chat UI for tasks it cannot perform, such as solving CAPTCHAs or making subjective decisions.
* **Persistent Identity:** Saves browser session data (cookies, history) to a local profile, which significantly reduces the frequency of CAPTCHAs and security checks on subsequent runs.
* **Dynamic Search:** If the agent doesn't know the URL for a website, it will automatically perform a Google Search to find it.

---

## 🛠️ Setup and Execution

Follow these steps to get the agent up and running.

### **1. Prerequisites**
* **Node.js:** Ensure you have Node.js installed on your machine.
* **Google Gemini API Key:** You need a valid API key from the Google AI Studio.

### **2. Installation**
1.  Place all project files (`index.js`, `prompt.txt`, etc.) into a single folder.
2.  Open a terminal in the project folder and run the following command to install all required dependencies:
    ```bash
    npm install express ws puppeteer puppeteer-extra puppeteer-extra-plugin-stealth dotenv
    ```

### **3. Configuration**
1.  Create a file named `.env` in the root of the project folder.
2.  Add your Gemini API key to this file like so:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

### **4. Running the Agent**
1.  Run the server from your terminal:
    ```bash
    node index.js
    ```
    The terminal will show: `🚀 Server is ready. Please open http://localhost:3000 in your browser.`

2.  Open your Chrome browser and navigate to `http://localhost:3000`.

3.  The chat interface will appear, and you can begin giving the agent objectives.

> **Important First Run:** The first time you run the agent, it will create a folder named `my_browser_profile`. It is highly recommended to use this first session to manually log in to a Google account within the agent's browser window to establish a trusted session.

---

## 🏛️ System Architecture

The agent's intelligence is built on a two-tier model that separates high-level planning from low-level execution.

1.  **The Orchestrator (`gemini-2.5-flash`)**
    * **Role:** The "Project Manager."
    * **Inputs:** The user's objective, the action history, and a one-sentence summary of the current page state.
    * **Function:** It does **not** see the screen. Its job is to look at the overall progress and create a high-level plan by choosing the next "tool" to use (e.g., `analyze_screen`, `Google Search`, `request_credentials`). This makes it extremely fast and cost-effective.

2.  **The Specialist (`gemini-2.5-pro`)**
    * **Role:** The "Vision Expert" or "Hands."
    * **Inputs:** A specific task from the Orchestrator, a screenshot of the page, and (if necessary) the simplified HTML code.
    * **Function:** It performs detailed analysis of the visual layout and code to determine the precise action to take (e.g., `click`, `type`) and the exact CSS selector to use. It also provides a summary of the result of its action.

This separation prevents the agent from getting lost in details and allows it to form more robust, long-term plans while still acting with precision.

---

## 🗺️ Roadmap: Next Steps and Future Development

This project has a strong foundation, but there are many exciting improvements planned to elevate it to a world-class agent.

### **Next Steps (Immediate Improvements)**

1.  **Refined Error Handling:** Currently, if an action fails, the agent self-corrects by re-analyzing with HTML. We will enhance this by providing the specific error message (e.g., "Timeout waiting for selector") to the Orchestrator so it can make an even more informed plan, such as deciding the page is broken or a different approach is needed.
2.  **Multi-Tab Management:** Implement the ability for the agent to open links in new tabs, switch between tabs, and close them. This is essential for tasks that involve cross-referencing information or complex login flows that spawn new windows.
3.  **Enhanced Tool Integration:** Add more "tools" for the Orchestrator to use beyond `analyze_screen` and `Google Search`. This could include:
    * **`execute_code`:** A tool to run specific JavaScript snippets on the page for tasks that are difficult with simple clicks and types.
    * **`read_text_from_screen`:** A dedicated tool to extract all visible text from a page to answer questions without needing to parse the full HTML.

### **Future Development (Long-Term Vision)**

1.  **Long-Term Memory (Vector Database):**
    * Currently, the agent's memory is erased after each session. The next major leap is to integrate a **vector database** (like Pinecone or ChromaDB).
    * **Functionality:** Before starting any task, the agent would search this database for similar past objectives. If it finds a match, it can retrieve the entire successful action sequence and execute it, potentially completing a task in seconds that previously took minutes. It would learn from every interaction.
2.  **Advanced Human-in-the-Loop (HITL) for Ambiguity:**
    * Beyond simple CAPTCHAs, we will teach the agent to ask for help when it faces **ambiguity**.
    * **Example:** If it finds two buttons labeled "Download," it will pause and ask in the UI, "I see two 'Download' buttons. Which one should I click?" presenting a screenshot with the buttons highlighted.
3.  **Cross-Application Capabilities (Beyond the Browser):**
    * The ultimate vision is to evolve the agent beyond just controlling a web browser.
    * Using desktop automation libraries, the agent's architecture could be extended to understand screenshots of any application (Excel, Outlook, Slack) and interact with them, making it a true "Computer Use Agent" like the systems that inspired this project.
  
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss proposed changes.

## License

This project is licensed under the MIT License.

## Contact

**SAQIB SHERWANI**

[My GitHub](https://github.com/saqibcodes007)

[Email Me!](mailto:sherwanisaqib@gmail.com)

---
<p align="center">
    Developed by Saqib Sherwani
    <br>
    Copyright © 2025 • All Rights Reserved
</p>
