---
name: test-assessment
description: Assess pipeline test results and identify bugs. Use this skill after running the takeoff pipeline on a test PDF. Triggers on "assess results", "review test", "what went wrong", "identify bugs", "analyze output", "rate the run", "check the report".
---

# Test Assessment — takeoff-ai Pipeline

You are a QA engineer specializing in construction takeoff accuracy. After each pipeline test run, you systematically assess the output, identify bugs, and produce an actionable fix list.

## Assessment Process

### 1. Gather Data

Collect all evidence from the test run:

- **Report JSON:** Read `data/jobs/{job_id}/report.json` — segments, linear ft, flagged count, per-page detections, errors
- **Annotated PDF:** Render pages from `data/jobs/{job_id}/annotated.pdf` to PNG and visually inspect
- **Server logs:** Check `/tmp/server.log` for exceptions or warnings
- **Job record:** Check `data/jobs.json` for job status
- **Input PDF analysis:** Compare vector count, page dimensions, text content against detection count

### 2. Score the Run

Rate on each dimension (1-10):

| Dimension | What to check |
|-----------|---------------|
| **Detection count** | Is it plausible? Hospital: 100-250/page. Residential: 40-150/page. 0 = total failure. |
| **Linear footage** | Non-zero? Plausible for building size? Hospital floor ~2000-5000ft total. Residential ~500-1500ft. |
| **Flagged ratio** | <20% flagged = good. >50% = confidence bug. 100% = threshold or default value issue. |
| **Localization** | Did it succeed or fall back? Check error messages for why. |
| **Wall type accuracy** | Do detected types match what's in the drawing? Wrong codes = legend extraction bug. |
| **Spatial accuracy** | Do annotations land on actual walls? Check rendered pages visually. |
| **Per-page balance** | Are detections roughly proportional to page complexity? One page with 10x fewer = localization or crop issue. |
| **Downloads** | Can annotated PDF and Excel be downloaded? Check file existence. |

### 3. Identify Bugs

For each issue found, classify it:

```
### Bug: [Short title]
- **Severity:** Critical / High / Medium / Low
- **Category:** detection | localization | coordinates | length | confidence | export | UI
- **Evidence:** [What data shows the bug]
- **Root cause:** [Best guess at why — reference specific code if possible]
- **Fix:** [Specific code change needed]
- **File:** pipeline/takeoff.py line XXX (or api/main.py, web/index.html)
```

### 4. Output Format

```markdown
## Test Assessment: [job_id]

**PDF:** [filename] | **Pages:** X | **Vectors:** Xk
**Date:** YYYY-MM-DD | **Pipeline version:** [commit hash or version]

### Overall Rating: X/10

| Dimension | Score | Notes |
|-----------|-------|-------|
| Detection count | X/10 | ... |
| Linear footage | X/10 | ... |
| ... | ... | ... |

### What's Working
- [bullet list of things that worked correctly]

### Bugs Found
[bug entries as above]

### Priority Fix Order
1. [highest impact fix first]
2. ...
3. ...

### Comparison to Previous Run
| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| Segments | X | Y | +/-Z |
| Linear ft | X | Y | +/-Z |
| Flagged | X% | Y% | +/-Z% |
```

### 5. Update RATINGS.md

After assessment, update `samples/inputs/RATINGS.md` with the new run data.

## Reference Values

### Expected detection counts (per page)
- Simple residential: 40-100 segments
- Complex residential (multi-unit): 80-150 segments
- Institutional (hospital/school): 100-250 segments
- Commercial (office): 60-120 segments

### Expected linear footage (per floor)
- Small residential floor: 200-600 ft
- Large residential floor: 500-1500 ft
- Institutional floor: 1500-5000 ft
- Commercial floor: 800-3000 ft

### Red flags
- 0 linear ft with >0 segments = length computation broken
- 100% flagged = confidence threshold or default value bug
- All pages "fell back to static crop" = localization model failing
- One page has <20 detections while others have >100 = crop/localization issue on that page
- Segments >200ft = scale parsing broken
- Door/window footage > wall footage = misclassification

## Key Files
- Pipeline: `pipeline/takeoff.py`
- API: `api/main.py`
- Frontend: `web/index.html`
- Constants: search for `CONFIDENCE_THRESHOLD`, `SNAP_THRESHOLD`, `FLOOR_PLAN_TOP_FRAC`
- Prompts: search for `SYSTEM_INSTRUCTION`, `GEMINI_PROMPT_V2`, `LOCALIZE_PROMPT`, `LEGEND_EXTRACTION_PROMPT`
