from __future__ import annotations

import argparse
import gc
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from clinical_inputs import (
    PIPELINE_NAME,
    PIPELINE_VERSION,
    clean_path,
    discover_packaged_cases,
    materialize_input,
    parse_branch_selection,
    parse_case_selection,
)
from clinical_outputs import atomic_csv, write_json
from clinical_workflow import analyze_case
from cta_legacy_backend_v1_0_15 import ensure_dir
import cta_legacy_backend_v1_0_15 as legacy
import stenosis_v2_core as core

DEFAULT_OUTPUT_ROOT = Path("cta_stenosis_v2_results")

DEFAULT_MIN_REPORTABLE_RATIO = 0.10

def run_batch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    input_path = clean_path(args.input)
    output_root = ensure_dir(clean_path(args.output))
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with materialize_input(input_path) as parent:
        cases = discover_packaged_cases(parent)
        cases = parse_case_selection(args.cases, cases)
        for case in cases:
            if (case.get("color_table") is not None):
                available_entries = legacy.parse_segmentation_color_table(Path(case["color_table"]))
                available_branches = [
                    legacy.normalize_packaged_branch_name(str(entry["name"]))
                    for value, entry in available_entries.items()
                    if ((int(value) > 0) and legacy.is_vessel_color_table_label(str(entry["name"])))
                ]
                selected = parse_branch_selection(args.branches, available_branches)
            else:
                requested = str((args.branches or "")).strip()
                selected = (
                    None
                    if (requested.lower() in {"", "all", "*"})
                    else [v for v in re.split(r"[,;\s]+", requested) if v]
                )
            try:
                summary = analyze_case(
                    case,
                    output_root,
                    selected,
                    bool(args.qa),
                    float(args.minimum_reportable_ratio),
                    float(args.centerline_step_mm),
                    float(args.cross_section_pixel_mm),
                )
                results.append(summary)
            except Exception as exc:
                errors.append(
                    {
                        "case_id": case["case_id"],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                gc.collect()
    if errors:
        atomic_csv(pd.DataFrame(errors), (output_root / "stenosis_v2_errors.csv"))
    write_json(
        (output_root / "batch_summary.json"),
        {
            "pipeline_name": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "successful_cases": int(len(results)),
            "failed_cases": int(len(errors)),
            "results": results,
            "errors": errors,
        },
    )
    if errors:
        raise SystemExit(1)
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "CTA stenosis v2: branch masks for navigation, POINT/parent rooted centerlines, "
            "longitudinally tracked CTA lumen segmentation, direct diameter/area stenosis, and a frozen one-pass raw CCTA Agatston output."
        )
    )
    parser.add_argument(
        "--input",
        type = str,
        required = True,
        help = "Folder or outer ZIP containing CTA ZIP + label or MRB; ColorTable optional.",
    )
    parser.add_argument("--output", type = str, default = str(DEFAULT_OUTPUT_ROOT), help = "Output root.")
    parser.add_argument(
        "--cases",
        type = str,
        default = None,
        help = "Case selection by comma-separated identifiers or numbers, e.g. 2,6-10.",
    )
    parser.add_argument(
        "--branches", type = str, default = None, help = "Optional branch selection, e.g. LAD,LCX,RCA."
    )
    parser.add_argument(
        "--minimum-reportable-ratio",
        type = float,
        default = DEFAULT_MIN_REPORTABLE_RATIO,
        help = "Minimum direct diameter-stenosis ratio retained as a candidate. Default 0.10.",
    )
    parser.add_argument(
        "--centerline-step-mm",
        type = float,
        default = core.DEFAULT_CENTERLINE_STEP_MM,
        help = "Root-to-tip centerline sampling interval in mm.",
    )
    parser.add_argument(
        "--cross-section-pixel-mm",
        type = float,
        default = core.DEFAULT_CROSS_SECTION_PIXEL_MM,
        help = "Orthogonal cross-section in-plane pixel size in mm.",
    )
    parser.add_argument(
        "--qa", dest = "qa", action = "store_true", default = True, help = "Save QA figures (default)."
    )
    parser.add_argument(
        "--no-qa", dest = "qa", action = "store_false", help = "Skip QA figures for faster runs."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        run_batch(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        output_root = ensure_dir(clean_path(args.output))
        write_json(
            (output_root / "batch_error.json"),
            {"error": str(exc), "error_type": type(exc).__name__},
        )
        raise SystemExit(2) from exc

if (__name__ == "__main__"):
    main()
