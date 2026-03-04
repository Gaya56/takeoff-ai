"""
TakeoffPipeline: AI-powered construction takeoff from architectural PDFs.
Uses Gemini Vision to detect wall types and segments from floor plans.
"""

import os
import re
import json
import base64
import fitz  # PyMuPDF
from google import genai
from google.genai import types


COLOR_MAP = {
    # Exterior walls
    "M1": (0.9, 0.1, 0.1),   # Red
    "M2": (0.1, 0.3, 0.9),   # Blue
    "M3": (0.9, 0.5, 0.0),   # Orange
    # Interior partitions
    "C1": (0.0, 0.6, 0.2),   # Green
    "C2": (0.5, 0.0, 0.7),   # Purple
    "C3": (0.8, 0.8, 0.0),   # Yellow
    # Doors / windows
    "DOOR": (0.0, 0.7, 0.8), # Cyan
    "WIN":  (0.0, 0.8, 0.6), # Teal
    # Structural
    "BEAM": (0.6, 0.3, 0.0), # Brown
    "COL":  (0.4, 0.4, 0.4), # Gray
    "STAIR":(0.7, 0.5, 0.0), # Dark yellow
    # Dimensions / rooms
    "DIM":  (0.5, 0.5, 0.5), # Gray
    "ROOM": (0.3, 0.6, 0.9), # Light blue
}

# Category fallback colors (when code not in COLOR_MAP)
CATEGORY_COLOR = {
    "wall":       (0.7, 0.2, 0.2),
    "door":       (0.0, 0.7, 0.8),
    "window":     (0.0, 0.8, 0.6),
    "structural": (0.5, 0.3, 0.0),
    "dimension":  (0.5, 0.5, 0.5),
    "room":       (0.3, 0.6, 0.9),
    "other":      (0.6, 0.6, 0.6),
}

FLAGGED_COLOR = (1, 0.4, 0.5)   # Pink/red for uncertain

CONFIDENCE_THRESHOLD = 0.65

GEMINI_PROMPT = """You are an expert construction estimator analyzing an architectural floor plan.

Analyze this floor plan image and return a JSON object with these fields:

1. **wall_types**: All element types from the legend/assembly table. Each:
   - "code": exact code (e.g. "M1", "C1", "DOOR", "WIN", "DIM")
   - "description": what it is (e.g. "brick exterior wall", "swing door", "window opening")
   - "category": one of "wall", "door", "window", "dimension", "structural", "room", "other"

2. **scale**: Drawing scale string (e.g. "1/8 inch = 1 foot"), or null.

3. **detections**: Every identifiable element on this floor plan page:
   - "type_code": code from wall_types, or standard codes: DOOR, WIN, BEAM, COL, STAIR, ROOM
   - "category": "wall" | "door" | "window" | "dimension" | "structural" | "room" | "other"
   - "page": 0
   - "x1": x pixel of element START point (left/top end of centerline)
   - "y1": y pixel of element START point
   - "x2": x pixel of element END point (right/bottom end of centerline)
   - "y2": y pixel of element END point
   - "real_length_ft": length in feet using scale. Estimate if no scale.
   - "label": short text label to show (e.g. "M1", "DOOR", "12.5ft")
   - "confidence": 0.0-1.0

CRITICAL INSTRUCTIONS:
- x1,y1 → x2,y2 are the TWO ENDPOINTS of the centerline (not bounding box corners)
- Horizontal elements: y1 ≈ y2, x1 < x2
- Vertical elements: x1 ≈ x2, y1 < y2
- Cover ALL elements: walls, doors, windows, dimension lines, room labels, stairs
- Match wall codes from the legend first (M1/M2/M3 = exterior, C1/C2/C3 = interior partitions)
- Doors: use DOOR + arc center at x1,y1; windows: use WIN
- If this is a legend/title/detail page with no floor plan, return empty detections array
- Aim for complete coverage — mark every visible element

Return ONLY valid JSON. No markdown, no explanation."""


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from Gemini response, handling common formatting issues."""
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON between first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try fixing trailing commas
    cleaned = re.sub(r',\s*([}\]])', r'\1', text[start:end] if start >= 0 else text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return {"wall_types": [], "scale": None, "detections": []}


class TakeoffPipeline:
    def __init__(self, gemini_api_key: str = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=self.api_key)
        self.pdf_path = None
        self.doc = None
        self.analysis = None

    def ingest_pdf(self, pdf_path: str) -> dict:
        """Load PDF and extract page info."""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        info = {
            "path": pdf_path,
            "page_count": len(self.doc),
            "pages": []
        }
        for i, page in enumerate(self.doc):
            rect = page.rect
            info["pages"].append({
                "page": i,
                "width": rect.width,
                "height": rect.height,
            })
        return info

    def analyze_plan(self) -> dict:
        """Send each page to Gemini Vision for wall detection."""
        if not self.doc:
            raise RuntimeError("Call ingest_pdf first")

        all_wall_types = []
        all_detections = []
        scale = None

        for page_idx in range(len(self.doc)):
            page = self.doc[page_idx]
            # Render page to high-res image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better detail
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            page_prompt = f"This is page {page_idx + 1} of {len(self.doc)} of an architectural floor plan.\n\n{GEMINI_PROMPT}"

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=img_bytes,
                                )
                            ),
                            types.Part(text=page_prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=65536,
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text.strip()
            result = _parse_json_safe(raw_text)

            # Collect wall types (handle both "wall_types" array and "legend" dict formats)
            wt_list = result.get("wall_types", [])
            legend = result.get("legend", {})
            if legend and not wt_list:
                for code, desc in legend.items():
                    wt_list.append({"code": code, "description": desc})
            for wt in wt_list:
                if wt not in all_wall_types:
                    all_wall_types.append(wt)

            # Set page index on detections
            for det in result.get("detections", []):
                det["page"] = page_idx
                all_detections.append(det)

            if result.get("scale") and not scale:
                scale = result["scale"]

        self.analysis = {
            "wall_types": all_wall_types,
            "scale": scale,
            "detections": all_detections,
        }
        return self.analysis

    def annotate_pdf(self, output_path: str) -> str:
        """Draw colored wall overlays on the PDF."""
        if not self.analysis or not self.doc:
            raise RuntimeError("Call analyze_plan first")

        # Reopen the doc fresh for annotation
        doc = fitz.open(self.pdf_path)
        zoom = 2.0  # Must match the zoom used in analyze_plan

        for det in self.analysis["detections"]:
            page_idx = det.get("page", 0)
            if page_idx >= len(doc):
                continue
            page = doc[page_idx]

            # Convert image pixel coords → PDF point coords
            x1 = det["x1"] / zoom
            y1 = det["y1"] / zoom
            x2 = det["x2"] / zoom
            y2 = det["y2"] / zoom

            # Smart centerline: if bounding box given, derive the wall centerline
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > h * 2:
                # Horizontal wall — snap to horizontal centerline
                mid_y = (y1 + y2) / 2
                draw_x1, draw_y1 = min(x1, x2), mid_y
                draw_x2, draw_y2 = max(x1, x2), mid_y
            elif h > w * 2:
                # Vertical wall — snap to vertical centerline
                mid_x = (x1 + x2) / 2
                draw_x1, draw_y1 = mid_x, min(y1, y2)
                draw_x2, draw_y2 = mid_x, max(y1, y2)
            else:
                # Diagonal or already a centerline — use as-is
                draw_x1, draw_y1 = x1, y1
                draw_x2, draw_y2 = x2, y2

            type_code = det.get("type_code", "UNKNOWN")
            confidence = det.get("confidence", 0.0)
            is_flagged = confidence < CONFIDENCE_THRESHOLD or type_code == "UNKNOWN"
            is_very_uncertain = confidence < 0.45 or type_code == "UNKNOWN"

            category = det.get("category", "wall")
            color = COLOR_MAP.get(type_code) or CATEGORY_COLOR.get(category, FLAGGED_COLOR)
            draw_color = FLAGGED_COLOR if is_flagged else color

            # Draw wall segment as thick colored line
            shape = page.new_shape()
            shape.draw_line(fitz.Point(draw_x1, draw_y1), fitz.Point(draw_x2, draw_y2))
            shape.finish(color=draw_color, width=3.0, stroke_opacity=0.85)
            shape.commit()

            # Add text label at midpoint
            label = type_code if not is_flagged else f"{type_code}?"
            mid_x = (draw_x1 + draw_x2) / 2
            mid_y = (draw_y1 + draw_y2) / 2
            page.insert_text(
                fitz.Point(mid_x + 2, mid_y - 4),
                label,
                fontsize=6,
                color=draw_color,
            )

            # Only circle walls that are very uncertain (not just moderately flagged)
            if is_very_uncertain:
                cx, cy = mid_x, mid_y
                radius = max(w, h) / 2 + 8
                radius = max(radius, 12)
                rect = fitz.Rect(cx - radius, cy - radius, cx + radius, cy + radius)
                shape2 = page.new_shape()
                shape2.draw_oval(rect)
                shape2.finish(color=FLAGGED_COLOR, width=1.0, stroke_opacity=0.6)
                shape2.commit()

        doc.save(output_path)
        doc.close()
        return output_path

    def generate_report(self) -> dict:
        """Generate material quantity report from analysis."""
        if not self.analysis:
            raise RuntimeError("Call analyze_plan first")

        detections = self.analysis["detections"]

        # Per-floor, per-type footage
        floor_data = {}
        flagged_count = 0
        total_footage = 0.0

        for det in detections:
            page = det.get("page", 0)
            type_code = det.get("type_code", "UNKNOWN")
            confidence = det.get("confidence", 0.0)
            length_ft = det.get("real_length_ft", 0.0)

            if confidence < CONFIDENCE_THRESHOLD or type_code == "UNKNOWN":
                flagged_count += 1

            floor_key = f"Floor {page + 1}"
            if floor_key not in floor_data:
                floor_data[floor_key] = {}

            if type_code not in floor_data[floor_key]:
                floor_data[floor_key][type_code] = {
                    "linear_ft": 0.0,
                    "segment_count": 0,
                }

            floor_data[floor_key][type_code]["linear_ft"] += length_ft
            floor_data[floor_key][type_code]["segment_count"] += 1
            total_footage += length_ft

        return {
            "wall_types": self.analysis["wall_types"],
            "scale": self.analysis["scale"],
            "floors": floor_data,
            "total_segments": len(detections),
            "total_linear_ft": round(total_footage, 2),
            "flagged_count": flagged_count,
        }

    def run(self, pdf_path: str, output_pdf_path: str = None) -> dict:
        """Full pipeline: ingest → analyze → annotate → report."""
        info = self.ingest_pdf(pdf_path)

        if output_pdf_path is None:
            base = os.path.splitext(pdf_path)[0]
            output_pdf_path = f"{base}_annotated.pdf"

        analysis = self.analyze_plan()
        self.annotate_pdf(output_pdf_path)
        report = self.generate_report()

        return {
            "pdf_info": info,
            "analysis": analysis,
            "annotated_pdf": output_pdf_path,
            "report": report,
        }
