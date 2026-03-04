"""FastAPI backend for takeoff-ai."""

import os
import uuid
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tinydb import TinyDB, Query

from pipeline.takeoff import TakeoffPipeline

app = FastAPI(title="takeoff-ai", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

db = TinyDB(DATA_DIR / "jobs.json")
jobs_table = db.table("jobs")
Job = Query()

WEB_DIR = Path(__file__).parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    notes: str = Form(default=""),
    building_type: str = Form(default=""),
    wall_codes: str = Form(default=""),
    scale: str = Form(default=""),
    num_floors: str = Form(default=""),
    special_instructions: str = Form(default=""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.pdf"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    output_path = job_dir / "annotated.pdf"

    # Build structured context from form fields
    user_context = {}
    if building_type:
        user_context["building_type"] = building_type
    if wall_codes:
        user_context["wall_codes"] = wall_codes
    if scale:
        user_context["scale"] = scale
    if num_floors:
        user_context["num_floors"] = num_floors
    if special_instructions:
        user_context["special_instructions"] = special_instructions

    jobs_table.insert({
        "job_id": job_id,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    })

    try:
        pipeline = TakeoffPipeline()
        result = pipeline.run(
            str(input_path), str(output_path),
            user_notes=notes,
            user_context=user_context if user_context else None,
        )

        report_path = job_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(result["report"], f, indent=2)

        jobs_table.update(
            {
                "status": "complete",
                "output_pdf": str(output_path),
                "total_segments": result["report"]["total_segments"],
                "total_linear_ft": result["report"]["total_linear_ft"],
                "scale": result["report"]["scale"],
                "flagged_count": result["report"].get("flagged_count", 0),
                "error_count": len(result["report"].get("errors", [])),
            },
            Job.job_id == job_id,
        )

        return {
            "job_id": job_id,
            "report": result["report"],
        }
    except Exception as e:
        jobs_table.update({"status": "failed", "error": str(e)}, Job.job_id == job_id)
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


@app.get("/download/{job_id}/excel")
async def download_excel(job_id: str):
    report_path = JOBS_DIR / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    with open(report_path) as f:
        report = json.load(f)

    excel_path = JOBS_DIR / job_id / "report.xlsx"
    if not excel_path.exists():
        pipeline = TakeoffPipeline()
        pipeline.export_excel(report, str(excel_path))

    return FileResponse(
        str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"takeoff_{job_id}_report.xlsx",
    )


@app.get("/jobs")
async def list_jobs():
    jobs = jobs_table.all()
    jobs.sort(key=lambda j: j.get("timestamp", ""), reverse=True)
    return {"jobs": jobs}


@app.get("/jobs/{job_id}/report")
async def get_report(job_id: str):
    report_path = JOBS_DIR / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with open(report_path) as f:
        return json.load(f)
