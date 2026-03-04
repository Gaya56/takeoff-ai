# takeoff-ai

AI-powered construction takeoff tool for Genesis Open Developments Inc. Upload a PDF floor plan → get back an annotated PDF with color-coded wall segments + a material quantity report in JSON and Excel.

## Features

- Reads legend/assembly details to identify wall type codes (M1, M2, C1, etc.)
- Detects individual wall segments using Gemini 2.5 Flash Vision
- Draws colored overlays on the PDF per wall type
- Generates linear footage report per wall type, per floor
- Exports Excel report (.xlsx) with floor/type/footage breakdown
- Flags low-confidence detections (pink) for human review
- Parallel Gemini calls — 3-page PDF processes in ~2.5 min
- Persistent job storage (TinyDB) — jobs survive server restarts
- Job history dashboard in the UI with re-openable reports

## Quick Start

```bash
git clone https://github.com/Gaya56/takeoff-ai
cd takeoff-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
GEMINI_API_KEY=your_key uvicorn api.main:app --reload
# Open http://localhost:8000
```

Or use the helper:
```bash
GEMINI_API_KEY=your_key ./start.sh
```

## Running Tests

```bash
source .venv/bin/activate
GEMINI_API_KEY=your_key python tests/test_pipeline.py
# Outputs annotated PDFs to tests/output/
```

## Wall Type Color Map

| Code | Color  | Description            |
|------|--------|------------------------|
| M1   | Red    | Brick exterior         |
| M2   | Blue   | Siding exterior        |
| M3   | Orange | Other masonry          |
| C1   | Green  | Load-bearing partition |
| C2   | Purple | Stairway/fire partition|
| C3   | Yellow | Double/party wall      |
| DOOR | Cyan   | Door openings          |
| WIN  | Teal   | Window openings        |
| BEAM | Brown  | Structural beam        |
| COL  | Gray   | Column                 |
| ?    | Pink   | Unknown / flagged      |

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/upload` | Upload PDF, run pipeline, return report |
| GET | `/download/{job_id}` | Download annotated PDF |
| GET | `/download/{job_id}/excel` | Download Excel report |
| GET | `/jobs` | List all jobs (sorted newest first) |
| GET | `/jobs/{job_id}/report` | Get report JSON for a specific job |
| GET | `/health` | Health check |
