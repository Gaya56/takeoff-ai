# takeoff-ai

AI-powered construction takeoff from architectural PDF floor plans. Upload a PDF → get an annotated PDF with color-coded elements + material quantity report (JSON + Excel).

**Client:** Salem Al-Zahari, Genesis Open Developments Inc.

## How It Works

```
PDF Upload → Legend Extraction (Gemini 3 Flash)
           → Floor Plan Localization (Gemini 3 Flash)
           → Element Detection (Gemini 3.1 Pro)
           → Vector Line Snapping (scipy cKDTree + PyMuPDF)
           → Post-Detection Validation
           → Annotated PDF + Report (JSON + Excel)
```

### 3-Call Gemini Pipeline
1. **Legend** — Flash reads wall codes, scale, floor labels from legend panel
2. **Localize** — Flash finds floor plan bounding box + scale bar per page
3. **Detect** — Pro identifies every wall segment, door, window, structural element with coordinates

### Post-Processing
- **Vector snapping** — snaps AI coordinates to exact PDF vector lines (eliminates ~15pt error)
- **Validation** — duplicate removal, geometric axis snap, scale verification, wall type consistency
- **Length computation** — real-world linear footage from coordinates + scale ratio

## Quick Start

```bash
git clone https://github.com/Gaya56/takeoff-ai
cd takeoff-ai
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
GEMINI_API_KEY=your_key uv run uvicorn api.main:app --reload
# Open http://localhost:8000
```

Or: `GEMINI_API_KEY=your_key ./start.sh`

## Stack

| Layer | Tech | Role |
|-------|------|------|
| AI Vision | Gemini 3.1 Pro + 3 Flash (`google-genai` SDK) | Legend extraction, localization, element detection |
| PDF | PyMuPDF (fitz) | Render pages, extract vector lines, draw annotations |
| Spatial Index | scipy cKDTree | Snap AI coordinates to real PDF vector line endpoints |
| API | FastAPI + uvicorn | Backend REST API |
| Frontend | Single HTML (vanilla JS) | Drag-drop upload, engineer input, results display |
| Storage | TinyDB | Persistent job records |
| Excel | openpyxl | Material quantity report export (.xlsx) |
| Image | opencv-python-headless | HSV color-preserving preprocessing |
| Package mgr | uv | Dependency management |

<!-- TODO: Market search each layer for better alternatives or additions -->

## Wall Type Color Map

| Code | Color | Description |
|------|-------|-------------|
| M1 | Red | Brick exterior |
| M2 | Blue | Siding exterior |
| M3 | Orange | Other masonry |
| C1 | Green | Load-bearing partition |
| C2 | Purple | Stairway/fire partition |
| C3 | Yellow | Double/party wall |
| DOOR | Cyan | Door openings |
| WIN | Teal | Window openings |
| BEAM/COL | Brown/Gray | Structural |
| STAIR | Dark Yellow | Stairway |
| Flagged | Pink | Low confidence — needs review |

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/upload` | Upload PDF + engineer context, run pipeline |
| GET | `/download/{job_id}` | Download annotated PDF |
| GET | `/download/{job_id}/excel` | Download Excel report |
| GET | `/jobs` | List all jobs |
| GET | `/jobs/{job_id}/report` | Get report JSON |
| GET | `/health` | Health check |

## Engineer Input Fields

The upload form accepts structured context that improves accuracy:
- **Building Type** — Residential / Commercial / Industrial / Institutional
- **Number of Floors** — helps set page structure expectations
- **Drawing Scale** — 1/8" = 1'-0", 1/4" = 1'-0", etc.
- **Wall Codes** — e.g. `M1=Brick, C1=Load-bearing, C2=Fire-rated`
- **Special Instructions** — free-form notes for edge cases

## Test Results

See `samples/inputs/RATINGS.md` for detailed test run tracking.

| Test PDF | Pages | Vectors | Best Score | Status |
|----------|-------|---------|------------|--------|
| LA County Hospital | 4 | 189K | 5/10 | Active testing |
| Calgary Residential | 26 | N/A | N/A | Corrupted — removed |

## File Structure

```
pipeline/takeoff.py   ← Core AI pipeline (edit this most)
api/main.py           ← FastAPI routes
web/index.html        ← Frontend UI
samples/inputs/       ← Test PDFs + RATINGS.md
samples/reference/    ← Visual QA targets (don't feed to pipeline)
data/jobs/            ← Per-job output (gitignored)
```

Full architecture details in `CLAUDE.md`.
