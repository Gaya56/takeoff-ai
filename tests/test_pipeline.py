"""Test the TakeoffPipeline with sample PDFs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.takeoff import TakeoffPipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")

TEST_RUNS = [
    ("samples/quadruplex_laval.pdf", "run_1"),
    ("samples/quadruplex_laval.pdf", "run_2"),
    ("samples/quadruplex_laval.pdf", "run_3"),
    ("samples/test_plan_1.pdf", "run_4"),
    ("samples/test_plan_2.pdf", "run_5"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    passed = 0
    total = len(TEST_RUNS)

    for i, (pdf_rel, run_name) in enumerate(TEST_RUNS, 1):
        pdf_path = os.path.join(os.path.dirname(__file__), "..", pdf_rel)
        output_path = os.path.join(OUTPUT_DIR, f"{run_name}_annotated.pdf")

        print(f"\n{'='*60}")
        print(f"Run {i}/{total}: {pdf_rel}")
        print(f"{'='*60}")

        try:
            pipeline = TakeoffPipeline()
            result = pipeline.run(pdf_path, output_path)
            report = result["report"]

            wall_types = [wt.get("code", str(wt)) for wt in report.get("wall_types", [])]
            total_detections = report.get("total_segments", 0)
            flagged_count = report.get("flagged_count", 0)

            print(f"  Wall types found: {wall_types}")
            print(f"  Total detections: {total_detections}")
            print(f"  Flagged count:    {flagged_count}")

            # Print footage per type
            for floor, types in report.get("floors", {}).items():
                for code, info in types.items():
                    print(f"  {floor} / {code}: {info['linear_ft']:.1f} ft ({info['segment_count']} segments)")

            # Check pass criteria
            annotated_exists = os.path.exists(output_path)
            has_wall_types = len(wall_types) >= 1 or total_detections >= 1

            if annotated_exists and has_wall_types:
                print(f"  ** PASS **")
                passed += 1
            else:
                print(f"  ** FAIL ** (annotated_exists={annotated_exists}, has_wall_types={has_wall_types})")

        except Exception as e:
            print(f"  ** FAIL ** Exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Final score: {passed}/{total} passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
