# Potentia AI - Autonomous Web Agent

## Overview

Potentia AI is a Flask-based web application that provides an autonomous web agent interface. The application integrates with Google's Gemini AI model through the `browser_use` library to create an intelligent agent capable of performing web-based tasks autonomously. Users interact with the agent through a modern chat interface where they can request various tasks to be performed.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a client-server architecture with the following key characteristics:

### Frontend Architecture
- **Single Page Application (SPA)**: Modern web interface built with vanilla HTML, CSS, and JavaScript
- **Responsive Design**: Mobile-first approach using CSS Grid and Flexbox
- **Real-time Chat Interface**: Dynamic message handling with conversation history
- **Modern UI/UX**: Dark theme with gradient effects and Font Awesome icons

### Backend Architecture
- **Flask Web Framework**: Lightweight Python web server handling HTTP requests
- **Asynchronous Processing**: Uses asyncio for handling browser automation tasks
- **Session Management**: Persistent browser sessions for continuous agent interactions
- **Environment-based Configuration**: Secure credential management using dotenv

## Key Components

### Web Interface (`templates/index.html`, `static/`)
- Chat-based user interface with message bubbles
- Responsive design supporting desktop and mobile devices
- Real-time message updates and conversation history
- Input validation and user experience enhancements

### Backend Server (`app.py`, `main.py`)
- Flask application serving the web interface
- Browser session initialization and management
- LLM integration with Google Gemini models
- Conversation history tracking and management

### AI Integration
- **Browser Use Library**: Enables autonomous web browser control
- **Google Gemini LLM**: Provides natural language understanding and generation
- **Persistent Sessions**: Maintains context across multiple interactions

## Data Flow

1. **User Input**: Users enter tasks through the web interface
2. **Message Processing**: Frontend sends user messages to the Flask backend
3. **Agent Initialization**: Backend creates browser agents with conversation context
4. **Task Execution**: Agent performs autonomous web browsing and task completion
5. **Response Generation**: Agent generates responses based on task outcomes
6. **UI Updates**: Frontend displays agent responses and maintains conversation flow

## External Dependencies

### Core Libraries
- **Flask**: Web framework for serving the application
- **browser_use**: Browser automation and agent functionality
- **ChatGoogle**: Google Gemini AI integration
- **python-dotenv**: Environment variable management

### Frontend Dependencies
- **Google Fonts (Poppins)**: Typography and font styling
- **Font Awesome**: Icon library for UI elements
- **Vanilla JavaScript**: No additional frontend frameworks

### Environment Variables
- `GEMINI_API_KEY`: Required for Google AI authentication
- `MODEL`: Gemini model specification (default: gemini-2.5-flash)
- `TEMPERATURE`: AI response creativity level (default: 0.1)
- `SESSION_SECRET`: Flask session security key

## Deployment Strategy

### Development Environment
- Flask development server with debug logging
- Environment variable loading from `.env` file
- Hot reload capabilities for rapid development

### Production Considerations
- Browser session management with keep-alive functionality
- Error handling and logging for debugging
- Session persistence across multiple user interactions
- Secure credential management through environment variables

### Scalability Notes
- Single-threaded Flask application suitable for demonstration
- Browser session reuse for efficient resource management
- Conversation history maintained in memory (consider persistence for production)
- Asynchronous task handling for non-blocking operations

The application is designed as a proof-of-concept for autonomous web agents, prioritizing simplicity and functionality over enterprise-scale features. The architecture supports easy extension for additional AI models, enhanced browser capabilities, and improved user interface features.