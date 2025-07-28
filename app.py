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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "potentia-ai-secret-key")

# Global variables for conversation and LLM
conversation_history = []
llm = None

def initialize_llm():
    """Initialize the LLM"""
    global llm

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False

    try:
        model_name = os.getenv("MODEL", "gemini-1.5-flash") # Updated model
        temperature = float(os.getenv("TEMPERATURE", 0.0)) # Set to 0 for more deterministic behavior
        llm = ChatGoogle(model=model_name, temperature=temperature)
        logger.info("✅ LLM initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/agent', methods=['POST'])
def agent_endpoint():
    """Handle agent requests from the frontend"""
    global conversation_history, llm

    try:
        data = request.get_json()
        if not data or 'history' not in data:
            return jsonify({'error': 'Invalid request format'}), 400

        conversation_history = data['history']

        if not llm:
            if not initialize_llm():
                 return jsonify({'response': 'Failed to initialize the AI model. Please check your API key.'}), 500

        return handle_browser_automation()

    except Exception as e:
        logger.error(f"❌ Error in agent endpoint: {e}")
        return jsonify({'response': f'I encountered an error: {str(e)}. Please try again.'}), 500

def handle_browser_automation():
    """Handle requests with full browser automation"""
    global conversation_history, llm
    browser_session = None
    try:
        # Create a new browser session for each request
        browser_session = BrowserSession()
        logger.info("✅ New browser session created")

        full_task_with_history = "\n".join(conversation_history)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            agent = Agent(
                task=full_task_with_history,
                llm=llm,
                browser_session=browser_session,
                interactive=False, # Set to False for non-interactive execution
            )

            logger.info(f"🚀 Agent running task: {full_task_with_history}")

            run_history = loop.run_until_complete(agent.run())

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
            logger.error(f"❌ Error running agent: {e}", exc_info=True)
            return jsonify({'response': f'I encountered an error during automation: {str(e)}. Please check the logs.'})
        finally:
            if loop.is_running():
                loop.close()


    except Exception as e:
        logger.error(f"❌ Error in browser automation setup: {e}", exc_info=True)
        return jsonify({'response': f'I encountered a setup error: {str(e)}. Please try again.'})
    finally:
        if browser_session:
            # Ensure the browser session is always closed
            try:
                final_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(final_loop)
                final_loop.run_until_complete(browser_session.close())
                final_loop.close()
                logger.info("✅ Browser session closed.")
            except Exception as e:
                logger.error(f"❌ Error closing browser session: {e}")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'llm_ready': llm is not None})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Potentia AI server...")
    initialize_llm()
    app.run(host='0.0.0.0', port=5000, debug=True)