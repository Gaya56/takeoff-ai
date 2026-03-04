# Test Ratings — takeoff-ai Pipeline

Tracking accuracy and quality across test runs on real architectural PDFs.

---

## Test File: `lacounty_hospital_floorplan.pdf`

**Project:** MLK Behavioral Health Center Renovation + LAC+USC Restorative Care Village + Olive View-UCLA Medical Center + Rancho Los Amigos National Rehabilitation Center
**Architect:** HOK / CannonDesign / SWA Architects / Gonzalez Goodale Architects
**Type:** Institutional healthcare (behavioral health, residential treatment, medical)
**Pages:** 4 (each 3024×2160pt / 42"×30")
**Scale:** 1/8" = 1'-0"
**Vector paths:** 189K total (32K + 75K + 30K + 51K per page)
**Difficulty:** Very Hard — multi-architect permit set, renovation (existing + new), multi-plan sheets, healthcare-specific partition types

### Run 1 — v2 pre-fix (2026-03-04)
| Metric | Value |
|--------|-------|
| **Rating** | **3/10** |
| Total segments | 479 |
| Total linear ft | 0.0 (broken) |
| Flagged | 351/479 (73%) |
| Localization | All 4 pages fell back to static crop |
| Downloads | Broken (returned JSON error) |

**Issues:** `real_length_ft` all zero (Gemini returned 0, no fallback computation). Confidence all default 0.5 < threshold 0.65 = everything flagged. Localization crop at 40%/92% was too aggressive. `thinking_config` + `response_schema` potentially incompatible.

### Run 2 — v2 post-fix-1 (2026-03-04)
| Metric | Value |
|--------|-------|
| **Rating** | **5/10** |
| Total segments | 572 |
| Total linear ft | 10,088.82 |
| Flagged | 572/572 (100%) |
| Localization | All 4 pages fell back — JSON truncation (max_output_tokens=512 too low) |
| Per-page detections | P1: 236, P2: 67, P3: 106, P4: 163 |

**Fixes applied:** (1) `FLOOR_PLAN_TOP_FRAC` 0.40→0.05, (2) logged localization exceptions, (3) added `_compute_real_lengths()` from coords+scale, (4) removed `thinking_config`.

**Results:** Linear footage now computed correctly (M1: 3,192ft, C1: 2,701ft, C2: 2,946ft — plausible for institutional). Still all flagged because confidence defaults to 0.5 and threshold was 0.65. Localization JSON truncated at 512 tokens. Page 2 underdetected (67 vs expected 150+ for 4-plan sheet). Annotations concentrated in legend/title areas on some pages.

**Breakdown by wall type:**
| Type | Linear Ft | Segments | Plausible? |
|------|-----------|----------|------------|
| M1 (Exterior) | 3,192 | 103 | Yes — large institutional perimeter |
| C1 (Interior partition) | 2,701 | 152 | Yes — heavy interior partitioning |
| C2 (Fire-rated) | 2,946 | 153 | Yes — healthcare corridors are fire-rated |
| DOOR | 613 | 100 | Reasonable count, footage seems high |
| WIN | 401 | 52 | Plausible |
| STAIR | 234 | 12 | Plausible |

### Run 3 — v2 post-fix-2 (2026-03-04)
**Fixes applied:** (1) Localization `max_output_tokens` 512→2048 (fix JSON truncation), (2) `CONFIDENCE_THRESHOLD` 0.65→0.40 (stop mass-flagging valid detections).

| Metric | Value |
|--------|-------|
| **Rating** | **pending** |

---

## Test File: `calgary_permit_drawing_residential.pdf` (CORRUPTED)

**Status:** File has broken zlib streams. All 26 pages render blank white. MuPDF, Poppler, and qpdf all fail to recover content. Needs re-download from source.

---

## Scoring Rubric

| Score | Meaning |
|-------|---------|
| 1-2 | Pipeline crashes or returns 0 detections |
| 3-4 | Detections exist but major data issues (0 lengths, all flagged, wrong locations) |
| 5-6 | Plausible numbers, some correct locations, but significant gaps or misplacements |
| 7-8 | Most walls correctly identified and located, minor gaps, usable for rough estimates |
| 9-10 | Near-production quality, matches manual takeoff within 10-15% |

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1 | 2026-03-04 | Initial MVP — 2-call pipeline, ~70% accuracy on residential |
| v2 | 2026-03-04 | 7-phase upgrade: vector snapping, 3-call pipeline, domain prompts, validation, color-preserving preprocessing, structured input, error handling |
| v2-fix1 | 2026-03-04 | Localization crop 0.05/0.95, compute lengths from coords, remove thinking_config, log exceptions |
| v2-fix2 | 2026-03-04 | Localization max_output_tokens 512→2048, confidence threshold 0.65→0.40 |
