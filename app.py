import asyncio
import logging
import os
import threading
from queue import Queue
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.llm import ChatGoogle
from browser_use.browser.session import BrowserSession

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "potentia-ai-secret-key")

# Global variables for the persistent browser system
conversation_history = []
llm = None
persistent_browser = None
persistent_context = None
agent_task_queue = Queue()
agent_response_queue = Queue()
agent_thread = None
is_browser_initialized = False

class PersistentBrowserManager:
    """Manages a single persistent browser session that multiple agents can connect to"""
    
    def __init__(self, llm):
        self.llm = llm
        self.browser_session = None
        self.running = True
        
    async def initialize_browser(self):
        """Initialize the persistent browser session - this only happens once"""
        try:
            global is_browser_initialized
            
            if is_browser_initialized:
                logger.info("✅ Browser already initialized, reusing existing instance")
                return True
            
            # Create browser session with persistence (matches original script exactly)
            self.browser_session = BrowserSession(
                keep_alive=True,  # CRITICAL: This prevents browser from closing!
                headless=False,   # Keep visible like original script
                user_data_dir='~/.config/browseruse/profiles/default',
            )
            
            # Start the browser session (if needed by the API)
            # await self.browser_session.start()
            
            is_browser_initialized = True
            logger.info("✅ Persistent browser session initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser session: {e}")
            return False
    
    async def run_agent_task(self, task_context):
        """Run an agent task using the persistent browser session"""
        try:
            # Ensure browser is initialized
            if not is_browser_initialized:
                if not await self.initialize_browser():
                    return "Failed to initialize browser"
            
            logger.info(f"📥 Running agent task with persistent browser session")
            
            # Create agent that connects to existing browser session
            agent = Agent(
                task=task_context,
                llm=self.llm,
                browser_session=self.browser_session,  # Reuse same session!
            )
            
            # Run the agent task
            run_history = await agent.run()
            
            # Extract response from agent execution
            if run_history.history and len(run_history.history) > 0:
                last_event = run_history.history[-1]
                
                if hasattr(last_event, 'result') and last_event.result:
                    result_info = last_event.result[0]
                    agent_response = result_info.extracted_content
                    logger.info(f"✅ Agent task completed: {agent_response}")
                    return agent_response
                else:
                    return "I could not determine the next step. Please provide more instructions."
            else:
                return "I encountered an issue processing your request."
                
        except Exception as e:
            logger.error(f"❌ Error running agent task: {e}")
            return f"I encountered an error: {str(e)}. Please try again."
    
    async def run_forever(self):
        """Main loop that processes tasks from the queue"""
        logger.info("🚀 Persistent browser manager started and ready for tasks")
        
        while self.running:
            try:
                # Check for new tasks
                if not agent_task_queue.empty():
                    task_context = agent_task_queue.get()
                    logger.info(f"📥 Processing task: {task_context[:100]}...")
                    
                    # Run agent task with persistent browser
                    response = await self.run_agent_task(task_context)
                    agent_response_queue.put(response)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in browser manager main loop: {e}")
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop the persistent browser manager"""
        self.running = False
        if self.browser_session:
            try:
                asyncio.run(self.browser_session.close())
                logger.info("🔄 Browser session closed")
            except:
                pass

def agent_worker():
    """Worker thread that runs the persistent browser manager"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(persistent_browser.run_forever())
    except Exception as e:
        logger.error(f"❌ Browser manager worker thread error: {e}")

def initialize_services():
    """Initialize LLM and start persistent browser manager"""
    global llm, persistent_browser, agent_thread
    
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("TEMPERATURE", 0.1))
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        logger.info("--- Initializing LLM and Persistent Browser System ---")
        
        # Initialize LLM
        llm = ChatGoogle(model=model_name, temperature=temperature)
        
        # Create persistent browser manager
        persistent_browser = PersistentBrowserManager(llm)
        
        # Start browser manager in background thread
        agent_thread = threading.Thread(target=agent_worker, daemon=True)
        agent_thread.start()
        
        logger.info("✅ LLM and persistent browser system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/agent', methods=['POST'])
def agent_endpoint():
    """Handle agent requests using the persistent browser"""
    global conversation_history
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'history' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        # Update conversation history
        conversation_history = data['history']
        
        # Ensure services are initialized
        if not llm or not persistent_browser:
            logger.warning("⚠️ Services not initialized, attempting initialization")
            if not initialize_services():
                return jsonify({'response': 'Services initialization failed. Please try again.'}), 200
        
        # Format task context
        full_task_with_history = "\n".join(conversation_history)
        
        # Send task to persistent browser manager
        agent_task_queue.put(full_task_with_history)
        
        # Wait for response with timeout
        import time
        timeout = 60  # 60 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not agent_response_queue.empty():
                response = agent_response_queue.get()
                return jsonify({'response': response})
            time.sleep(0.1)
        
        # Timeout case
        return jsonify({'response': 'Request timed out. Please try again.'}), 200
            
    except Exception as e:
        logger.error(f"❌ Error in agent endpoint: {e}")
        return jsonify({'response': f'I encountered an error: {str(e)}. Please try again.'}), 200

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'llm_ready': llm is not None,
        'browser_ready': persistent_browser is not None and persistent_browser.running,
        'browser_initialized': is_browser_initialized
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Potentia AI server...")
    
    # Initialize services
    if initialize_services():
        logger.info("✅ Services initialized successfully")
    else:
        logger.warning("⚠️ Service initialization failed")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Cleanup on exit
        if persistent_browser:
            persistent_browser.stop()