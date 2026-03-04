# takeoff-ai — Build Instructions for Claude Code

## What to build
AI-powered construction takeoff tool. Analyzes architectural PDF floor plans using Gemini Vision.

## Pipeline
1. PDF → images (PyMuPDF)
2. Gemini Vision reads legend, scale, detects walls with pixel bounding boxes
3. PyMuPDF draws colored overlays on PDF per wall type
4. FastAPI serves results
5. Single HTML frontend

## Color map
M1=red, M2=blue, M3=orange, C1=green, C2=purple, C3=yellow, flagged/unknown=pink

## Files to create

### pipeline/__init__.py (empty)

### pipeline/takeoff.py
TakeoffPipeline class:
- ingest_pdf(path) → converts each PDF page to high-res JPEG bytes using PyMuPDF
- analyze_plan(page_images, page_count) → calls Gemini with each page image, returns parsed JSON
- annotate_pdf(pdf_path, detections, output_path) → uses PyMuPDF to draw colored lines + labels on PDF
- generate_report(detections) → returns {wall_type: {total_ft, per_floor: {floor: ft}}}
- run(pdf_path, output_path) → full pipeline, returns {annotated_pdf, report, flagged_count, wall_types}

Gemini prompt (use gemini-2.5-flash model, GEMINI_API_KEY from env):
Send each page as inline image data (base64). Ask Gemini to return ONLY valid JSON:
{
  "legend": {"M1": "Brick exterior wall", "M2": "Siding exterior wall", ...},
  "scale": "1/8 inch = 1 foot",
  "scale_factor": 8,
  "detections": [
    {"type_code": "M1", "page": 0, "x1": 100, "y1": 200, "x2": 400, "y2": 210, "real_length_ft": 25.0, "confidence": 0.9}
  ]
}
Coords are 0-1000 normalized. If no legend exists, infer wall types from context.

For annotation: convert 0-1000 coords to PDF page coords (page.rect.width/height), draw lines with fitz.

### api/__init__.py (empty)

### api/main.py
FastAPI app:
- Serve web/index.html at GET /
- POST /upload: saves PDF, runs TakeoffPipeline.run(), returns JSON result + job_id
- GET /download/{job_id}: streams annotated PDF file
- GET /health: returns {"status": "ok"}
- Job files in /tmp/takeoff_jobs/{job_id}/
- Mount static files if needed

### web/index.html
Clean dark minimal UI. No framework. Pure HTML/CSS/JS.
- Header: "Takeoff AI" with subtitle
- Drag-drop zone for PDF upload
- Submit button
- Loading spinner during processing
- Results section:
  - Color-coded wall type legend (colored squares + descriptions)
  - Table: Wall Type | Linear Footage | Floors
  - Flagged walls count with warning if > 0
  - Download annotated PDF button
- Error display
- Calls fetch('/upload', {method: 'POST', body: formData}) then fetch('/download/'+job_id)

### requirements.txt
```
google-generativeai>=0.8.0
PyMuPDF>=1.24.0
Pillow>=10.0.0
fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
```

### .env.example
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### README.md
```markdown
# Takeoff AI
AI-powered construction takeoff from PDF floor plans.

## Setup
pip install -r requirements.txt
export GEMINI_API_KEY=your_key

## Run
uvicorn api.main:app --reload
# Open http://localhost:8000
```

### tests/test_pipeline.py
- Import TakeoffPipeline
- Test 5 runs: samples/quadruplex_laval.pdf x3, samples/test_plan_1.pdf x1, samples/test_plan_2.pdf x1
- Each test: print run number, wall types found, total detections, flagged count, footage per type
- Save annotated PDFs to tests/output/run_N_annotated.pdf
- Print PASS if annotated PDF created and report has at least 1 wall type, FAIL otherwise
- Print final score: X/5 passed

## After building all files
1. Run: pip install -r requirements.txt
2. Run tests: GEMINI_API_KEY=$GEMINI_API_KEY python tests/test_pipeline.py
3. Fix any errors until all 5 pass
4. Verify server starts: GEMINI_API_KEY=$GEMINI_API_KEY uvicorn api.main:app --port 8000 &
5. git add -A && git commit -m "feat: initial working takeoff-ai MVP" && git push origin main
6. Notify: /home/ali/.config/nvm/versions/node/v22.15.0/bin/node /home/ali/Desktop/openclaw/openclaw.mjs system event --text "takeoff-ai build complete - all tests passed and pushed to GitHub" --mode now
