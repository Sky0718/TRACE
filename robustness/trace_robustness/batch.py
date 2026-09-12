from __future__ import annotations
from typing import Any
from typing import Dict
from concurrent.futures import FIRST_COMPLETED
from typing import List
from typing import Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
import argparse
import gc
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import wait
from . import runtime, models, extraction, perturbation, storage

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "CTA Concept Extraction Pipeline")
    mode = parser.add_mutually_exclusive_group(required = True)
    mode.add_argument(
        "--batch", type = str, default = None, help = "Input directory for batch analysis."
    )
    parser.add_argument(
        "--out",
        type = str,
        default = str(runtime.DATABASE_CASES_ROOT),
        help = "Output directory for per-case outputs.",
    )
    parser.add_argument(
        "--start", type = int, default = None, help = "Start case index for batch."
    )
    parser.add_argument("--end", type = int, default = None, help = "End case index for batch.")
    parser.add_argument(
        "--threshold",
        type = float,
        default = runtime.DEFAULT_CALCIUM_THRESHOLD_HU,
        help = "Calcium HU threshold.",
    )
    parser.add_argument(
        "--min-calc-vol",
        type = float,
        default = runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3,
        help = "Minimum calcium lesion volume mm^3.",
    )
    parser.add_argument(
        "--stenosis-threshold",
        type = float,
        default = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
        help = "Geometric narrowing reporting threshold 0-1.",
    )
    parser.add_argument(
        "--qa",
        dest = "qa",
        action = "store_true",
        default = runtime.DEFAULT_QA_FIGURE,
        help = "Save QA figures.",
    )
    parser.add_argument(
        "--no-qa", dest = "qa", action = "store_false", help = "Do not save QA figures."
    )
    parser.add_argument(
        "--workers",
        type = int,
        default = runtime.DEFAULT_PARALLEL_WORKERS,
        help = "Parallel worker count for batch analysis. Default: 5.",
    )
    parser.add_argument(
        "--skip-existing",
        dest = "skip_existing",
        action = "store_true",
        default = runtime.DEFAULT_SKIP_EXISTING,
        help = "Skip cases already completed with the same algorithm_version and parameter_hash.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest = "skip_existing",
        action = "store_false",
        help = "Do not skip existing cases.",
    )
    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help = "Delete and rerun existing per-case output folders for selected cases.",
    )
    parser.add_argument(
        "--only-failed",
        action = "store_true",
        help = "Run only existing failed / review-needed cases.",
    )
    parser.add_argument(
        "--robustness-sample",
        type = int,
        default = 0,
        help = "Optional mask-perturbation robustness analysis sample size after batch/rebuild. Use e.g. 100 for manuscript robustness.",
    )
    parser.add_argument(
        "--robustness-case-start",
        type = int,
        default = None,
        help = "Optional inclusive numeric case-id start for robustness analysis, e.g. 200.",
    )
    parser.add_argument(
        "--robustness-case-end",
        type = int,
        default = None,
        help = "Optional inclusive numeric case-id end for robustness analysis, e.g. 400.",
    )
    parser.add_argument(
        "--robustness-overwrite",
        action = "store_true",
        help = "For robustness analysis, overwrite existing checkpoint rows for the selected cases before running.",
    )
    parser.add_argument(
        "--visual-background",
        choices = ["dark", "white"],
        default = runtime.DEFAULT_VISUAL_BACKGROUND_MODE,
        help = "Background color for saved visual figures. Default keeps original dark background; choose 'white' for pure white manuscript figures.",
    )
    mode.add_argument(
        "--rebuild-master",
        nargs = "?",
        const = str(runtime.DATABASE_CASES_ROOT),
        default = None,
        help = "Scan an existing per-case output folder and rebuild all paper-ready master tables without rerunning NIfTI analysis.",
    )
    return parser.parse_args()

def _run_one_case_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    img = Path(payload["img"])
    lbl = Path(payload["lbl"])
    out_root = Path(payload["out_root"])
    pid = str(payload["pid"])
    case_out = storage.ensure_dir(extraction.case_expected_output_dir(out_root, pid))
    log_path = (case_out / "case_processing.log")
    try:
        runtime.set_visual_background_mode(
            payload.get("visual_background_mode", runtime.DEFAULT_VISUAL_BACKGROUND_MODE)
        )

        def _no_shared_database_update(
            case_row: models.CaseConcept, db_root: Path = runtime.DATABASE_ROOT
        ) -> Dict[str, str]:
            return {
                "csv": "disabled_in_parallel_worker",
                "xlsx": "disabled_in_parallel_worker",
                "sqlite": "disabled_in_parallel_worker",
            }

        previous_database_update = storage.update_concept_database
        storage.update_concept_database = _no_shared_database_update
        try:
            (case_row, branch_rows, meta) = extraction.analyze_single_case(
                img,
                lbl,
                out_root,
                threshold_hu = float(payload["threshold_hu"]),
                min_calc_vol_mm3 = float(payload["min_calc_vol_mm3"]),
                min_branch_voxels = int(payload["min_branch_voxels"]),
                stenosis_threshold = float(payload["stenosis_threshold"]),
                save_qa = bool(payload["save_qa"]),
                update_master = False,
            )
        finally:
            storage.update_concept_database = previous_database_update
        return {
            "case_id": pid,
            "status": "success",
            "elapsed_sec": float(meta.get("elapsed_sec", np.nan)),
            "output_dir": str(meta.get("output_dir", "")),
            "case_log": str(log_path),
            "max_geometric_narrowing_index": float(
                storage.safe_case_max_narrowing(case_row, 0.0)
            ),
        }
    except Exception as exc:
        error_traceback = traceback.format_exc()
        with open(log_path, "a", encoding = "utf-8") as case_log:
            case_log.write((((str(exc) + "\n") + error_traceback) + "\n"))
        return {
            "case_id": pid,
            "status": "error",
            "error": str(exc),
            "traceback": error_traceback,
            "case_log": str(log_path),
        }

def analyze_batch(
    input_dir: Path,
    out_root: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    threshold_hu: float = runtime.DEFAULT_CALCIUM_THRESHOLD_HU,
    min_calc_vol_mm3: float = runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
    stenosis_threshold: float = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
    save_qa: bool = runtime.DEFAULT_QA_FIGURE,
    workers: int = runtime.DEFAULT_PARALLEL_WORKERS,
    skip_existing: bool = runtime.DEFAULT_SKIP_EXISTING,
    overwrite: bool = False,
    only_failed: bool = False,
    robustness_sample: int = 0,
    visual_background_mode: Optional[str] = None,
) -> None:
    pairs = storage.find_pairs(Path(input_dir), start = start, end = end)
    if not pairs:
        raise ValueError("No valid .img.nii.gz / .label.nii.gz pairs were found.")
    out_root = storage.ensure_dir(out_root)
    if (visual_background_mode is not None):
        runtime.set_visual_background_mode(visual_background_mode)
    resolved_background_mode = runtime.visual_background_mode()
    params = extraction.canonical_parameter_dict(
        threshold_hu, min_calc_vol_mm3, min_branch_voxels, stenosis_threshold, save_qa
    )
    selected: List[Tuple[str, Path, Path]] = []
    skipped_rows = []
    for pid, img, lbl in pairs:
        cdir = extraction.case_expected_output_dir(out_root, pid)
        if (overwrite and cdir.exists()):
            import shutil

            shutil.rmtree(cdir, ignore_errors = True)
        if (only_failed and (not extraction.case_is_failed_or_review_needed(cdir))):
            skipped_rows.append({"case_id": pid, "reason": "not_failed_or_review_needed"})
            continue
        if (skip_existing and extraction.case_is_completed_same_version(cdir, params)):
            skipped_rows.append(
                {"case_id": pid, "reason": "completed_same_version_and_parameter_hash"}
            )
            continue
        selected.append((pid, img, lbl))
    workers = max(1, int((workers or 1)))
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            (Path(out_root) / "cta_batch_skipped_cases.csv"),
            index = False,
            encoding = "utf-8-sig",
        )
    if not selected:
        storage.rebuild_master_database(
            out_root, db_root = storage.infer_database_root_from_cases_root(out_root)
        )
        return
    results: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    payloads = []
    for pid, img, lbl in selected:
        payloads.append(
            {
                "pid": pid,
                "img": str(img),
                "lbl": str(lbl),
                "out_root": str(out_root),
                "threshold_hu": float(threshold_hu),
                "min_calc_vol_mm3": float(min_calc_vol_mm3),
                "min_branch_voxels": int(min_branch_voxels),
                "stenosis_threshold": float(stenosis_threshold),
                "save_qa": bool(save_qa),
                "visual_background_mode": resolved_background_mode,
            }
        )
    if (workers == 1):
        for payload in payloads:
            res = _run_one_case_worker(payload)
            results.append(res)
            if (res.get("status") == "error"):
                error_rows.append(res)
            gc.collect()
    else:
        with ProcessPoolExecutor(max_workers = workers) as ex:
            future_map = {ex.submit(_run_one_case_worker, p): p for p in payloads}
            pending = set(future_map.keys())
            while pending:
                (done_set, pending) = wait(
                    pending, timeout = 0.35, return_when = FIRST_COMPLETED
                )
                for fut in done_set:
                    p = future_map[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = {
                            "case_id": p.get("pid"),
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    results.append(res)
                    if (res.get("status") == "error"):
                        error_rows.append(res)
                    gc.collect()
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["_sort_num"] = result_df["case_id"].apply(
            lambda x: storage.case_sort_value(str(x))[0]
        )
        result_df["_sort_str"] = result_df["case_id"].astype(str)
        result_df.sort_values(["_sort_num", "_sort_str"], inplace = True)
        result_df.drop(columns = ["_sort_num", "_sort_str"], inplace = True)
        result_df.to_csv(
            (Path(out_root) / "cta_batch_run_log.csv"), index = False, encoding = "utf-8-sig"
        )
    if error_rows:
        err_path = (Path(out_root) / "cta_concept_dataset_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_path, index = False, encoding = "utf-8-sig")
    db_root = storage.infer_database_root_from_cases_root(out_root)
    storage.rebuild_master_database(out_root, db_root = db_root)
    if (int((robustness_sample or 0)) > 0):
        perturbation.run_robustness_analysis(
            out_root,
            db_root = db_root,
            sample_n = int(robustness_sample),
            threshold_hu = threshold_hu,
            min_calc_vol_mm3 = min_calc_vol_mm3,
            min_branch_voxels = min_branch_voxels,
            stenosis_threshold = stenosis_threshold,
        )
    if error_rows:
        raise RuntimeError(
            f"{len(error_rows)} case(s) failed; see cta_concept_dataset_errors.csv."
        )

def main() -> None:
    args = parse_args()
    if (args.rebuild_master is not None):
        cases_root = Path(storage.clean_user_path(args.rebuild_master))
        db_root = storage.infer_database_root_from_cases_root(cases_root)
        storage.rebuild_master_database(cases_root, db_root = db_root)
        if (int((getattr(args, "robustness_sample", 0) or 0)) > 0):
            perturbation.run_robustness_analysis(
                cases_root,
                db_root = db_root,
                sample_n = int(args.robustness_sample),
                threshold_hu = args.threshold,
                min_calc_vol_mm3 = args.min_calc_vol,
                stenosis_threshold = args.stenosis_threshold,
                workers = args.workers,
                robustness_case_start = args.robustness_case_start,
                robustness_case_end = args.robustness_case_end,
                robustness_overwrite = bool((args.robustness_overwrite or args.overwrite)),
            )
        return
    if args.batch:
        runtime.set_visual_background_mode(args.visual_background)
        analyze_batch(
            Path(storage.clean_user_path(args.batch)),
            storage.resolve_runtime_path(Path(storage.clean_user_path(args.out))),
            start = args.start,
            end = args.end,
            threshold_hu = args.threshold,
            min_calc_vol_mm3 = args.min_calc_vol,
            stenosis_threshold = args.stenosis_threshold,
            save_qa = args.qa,
            workers = args.workers,
            skip_existing = args.skip_existing,
            overwrite = args.overwrite,
            only_failed = args.only_failed,
            robustness_sample = args.robustness_sample,
            visual_background_mode = args.visual_background,
        )
        return
