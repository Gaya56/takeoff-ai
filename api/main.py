"""FastAPI backend for takeoff-ai."""

import os
import uuid
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.takeoff import TakeoffPipeline

app = FastAPI(title="takeoff-ai", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/takeoff_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

WEB_DIR = Path(__file__).parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded PDF
    input_path = job_dir / "input.pdf"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    output_path = job_dir / "annotated.pdf"

    try:
        pipeline = TakeoffPipeline()
        result = pipeline.run(str(input_path), str(output_path))

        # Save report JSON
        report_path = job_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(result["report"], f, indent=2)

        return {
            "job_id": job_id,
            "report": result["report"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{job_id}")
async def download_annotated(job_id: str):
    pdf_path = JOBS_DIR / job_id / "annotated.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"takeoff_{job_id}_annotated.pdf",
    )
