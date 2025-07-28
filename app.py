import asyncio
import logging
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from browser_use import Agent, BrowserSession
from browser_use.llm import ChatGoogle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "potentia-ai-secret-key")

# Global variables for conversation and browser session
conversation_history = []
llm = None
browser_session = None

def initialize_browser_session():
    """Initialize the browser session and LLM - exactly like original main copy.py"""
    global browser_session, llm
    
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("TEMPERATURE", 0.1))
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize LLM
        llm = ChatGoogle(model=model_name, temperature=temperature)
        
        # Initialize ONE browser session that persists for the entire app lifecycle
        # This matches the original main copy.py behavior exactly
        browser_session = BrowserSession(keep_alive=True)
        logger.info("✅ LLM and persistent browser session initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize browser session: {e}")
        # Keep LLM even if browser fails
        if not llm:
            try:
                llm = ChatGoogle(model=model_name, temperature=temperature)
                logger.info("✅ LLM initialized, browser session unavailable")
            except:
                pass
        return False

def create_fresh_browser_session():
    """Only used as fallback if main session fails - try to recover"""
    try:
        global browser_session
        if browser_session:
            try:
                # Use asyncio to properly close the session
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(browser_session.close())
                loop.close()
            except:
                pass
        
        # Create new session with keep_alive=True to persist across interactions
        browser_session = BrowserSession(keep_alive=True)
        logger.info("✅ Recovery browser session created")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create recovery browser session: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/agent', methods=['POST'])
def agent_endpoint():
    """Handle agent requests from the frontend"""
    global conversation_history, llm, browser_session
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'history' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        # Update conversation history
        conversation_history = data['history']
        
        # Initialize LLM if not done
        if not llm:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return jsonify({'response': 'GEMINI_API_KEY is required for AI functionality.'}), 200
            
            model_name = os.getenv("MODEL", "gemini-2.5-flash")
            temperature = float(os.getenv("TEMPERATURE", 0.1))
            llm = ChatGoogle(model=model_name, temperature=temperature)
        
        # Try browser automation
        return handle_browser_automation()
            
    except Exception as e:
        logger.error(f"❌ Error in agent endpoint: {e}")
        return jsonify({'response': f'I encountered an error: {str(e)}. Please try again.'}), 200

def handle_browser_automation():
    """Handle requests with full browser automation - exactly like original main copy.py"""
    global conversation_history, llm, browser_session
    
    try:
        # Browser session should already exist from startup initialization
        if not browser_session:
            logger.warning("⚠️ Browser session missing, attempting recovery")
            if not create_fresh_browser_session():
                return jsonify({'response': 'Browser automation is not available. Please try again.'}), 200
        
        # Format the conversation history exactly like the original main copy.py
        full_task_with_history = "\n".join(conversation_history)
        
        # Run the agent asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create a new agent instance for each turn - exactly like original main copy.py
            # This is the key pattern: new Agent, same browser_session
            agent = Agent(
                task=full_task_with_history,
                llm=llm,
                browser_session=browser_session,
                interactive=True,
            )
            
            logger.info(f"🚀 Agent running with context: {full_task_with_history}")
            
            # Run the agent
            run_history = loop.run_until_complete(agent.run())
            
            # Extract the response exactly like the original
            if run_history.history and len(run_history.history) > 0:
                last_event = run_history.history[-1]
                
                if hasattr(last_event, 'result') and last_event.result:
                    result_info = last_event.result[0]
                    agent_response = result_info.extracted_content
                    
                    logger.info(f"✅ Agent response: {agent_response}")
                    return jsonify({'response': agent_response})
                else:
                    logger.warning("⚠️ No result found in agent response")
                    return jsonify({'response': 'I completed the task, but could not extract specific results.'})
            else:
                logger.warning("⚠️ No history found in agent run")
                return jsonify({'response': 'I encountered an issue processing your request.'})
                
        except Exception as e:
            logger.error(f"❌ Error running agent: {e}")
            # If browser session is corrupted, try to recover
            if 'BrowserContext' in str(e) or 'NoneType' in str(e):
                logger.warning("🔄 Browser session appears corrupted, attempting recovery")
                if create_fresh_browser_session():
                    return jsonify({'response': 'Browser session was recovered. Please try your request again.'})
            
            return jsonify({'response': f'I encountered an error: {str(e)}. Please try again.'})
        finally:
            # Clean up loop but keep browser session alive
            loop.close()
            
    except Exception as e:
        logger.error(f"❌ Error in browser automation: {e}")
        return jsonify({'response': f'I encountered an error: {str(e)}. Please try again.'})

def handle_llm_only_response():
    """Handle requests when browser automation is not available"""
    return jsonify({'response': 'Browser automation is not available. Please ensure Chrome or Firefox is installed and try again.'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'llm_ready': llm is not None, 'browser_ready': browser_session is not None})

@app.route('/reset-browser', methods=['POST'])
def reset_browser():
    """Reset browser session endpoint"""
    global browser_session
    try:
        if browser_session:
            # Close existing session
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(browser_session.close())
            loop.close()
            browser_session = None
            
        return jsonify({'status': 'Browser session reset successfully'})
    except Exception as e:
        logger.error(f"❌ Error resetting browser session: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize on startup
    logger.info("🚀 Starting Potentia AI server...")
    initialize_browser_session()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
