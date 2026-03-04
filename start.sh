#!/bin/bash
if [ -z "$GEMINI_API_KEY" ]; then
  echo "Error: GEMINI_API_KEY not set"
  echo "Usage: GEMINI_API_KEY=your_key ./start.sh"
  exit 1
fi

if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

uv sync

echo "Starting Takeoff AI server at http://localhost:8000"
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
