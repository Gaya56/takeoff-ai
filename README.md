# takeoff-ai

AI-powered construction takeoff tool. Analyzes architectural PDF floor plans to identify wall types, measure quantities, and generate material reports.

## Features

- Reads legend/assembly details to identify wall type codes (M1, M2, C1, etc.)
- Detects wall segments with pixel locations using Gemini Vision
- Draws colored overlays on the PDF per wall type
- Generates linear footage report per wall type, per floor
- Flags low-confidence detections for human review

## Quick Start

```bash
pip install -r requirements.txt
GEMINI_API_KEY=your_key uvicorn api.main:app --reload
# Open http://localhost:8000
```

## Running Tests

```bash
GEMINI_API_KEY=your_key python tests/test_pipeline.py
```

## Wall Type Color Map

| Code | Color  | Description         |
|------|--------|---------------------|
| M1   | Red    | Brick exterior      |
| M2   | Blue   | Siding              |
| M3   | Orange | Other masonry       |
| C1   | Green  | Load-bearing partition |
| C2   | Purple | Non-load-bearing    |
| C3   | Yellow | Other partition     |
| ?    | Pink   | Unknown / flagged   |
