#!/bin/bash
if [ -z "$GEMINI_API_KEY" ]; then
  echo "Error: GEMINI_API_KEY not set"
  echo "Usage: GEMINI_API_KEY=your_key ./start.sh"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

echo "Starting Takeoff AI server at http://localhost:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
