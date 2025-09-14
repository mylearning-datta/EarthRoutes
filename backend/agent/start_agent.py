#!/usr/bin/env python3
"""
Startup script for the Travel Planning Agent
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import langgraph
        import langchain
        import openai
        import fastapi
        import uvicorn
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if environment file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Please copy env.example to .env and configure your API keys")
        return False
    print("✅ Environment file found")
    return True

def check_openai_key():
    """Check if OpenAI API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️  OpenAI API key not configured")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    print("✅ OpenAI API key configured")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Travel Planning Agent...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Check API key
    if not check_openai_key():
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All checks passed! Starting agent server...")
    print("📡 Agent will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        from main import app
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Agent server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
