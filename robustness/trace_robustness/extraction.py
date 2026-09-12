from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from typing import Tuple
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import platform
import sys
import time
from datetime import timezone
from . import (
    runtime,
    models,
    branch_tables,
    measurements,
    quality_tables,
    storage,
    topology,
    visual_outputs,
)

def extract_case_concepts(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    threshold_hu: float = runtime.DEFAULT_CALCIUM_THRESHOLD_HU,
    min_calc_vol_mm3: float = runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
    stenosis_threshold: float = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
    save_qa: bool = runtime.DEFAULT_QA_FIGURE,
    update_master: bool = True,
) -> Tuple[models.CaseConcept, List[models.BranchConcept], Dict[str, Any]]:
    pid = img_path.name.replace(".img.nii.gz", "")
    case_out = storage.ensure_dir((out_dir / storage.sanitize_case_id(pid)))
    start_t = time.time()
    (image, spacing) = storage.load_nifti(img_path)
    (label_vol, _) = storage.load_nifti(lbl_path)
    storage.ensure_same_shape(image, label_vol)
    vessel_mask = (label_vol > 0)
    voxel_vol = float(((spacing[0] * spacing[1]) * spacing[2]))
    calc_mask = measurements.extract_calcification(
        image, vessel_mask, threshold_hu, voxel_vol, min_calc_vol_mm3
    )
    total_vessel_volume = float((np.sum(vessel_mask) * voxel_vol))
    total_calc_volume = float((np.sum(calc_mask) * voxel_vol))
    calcium_burden = (
        float(((total_calc_volume / total_vessel_volume) * 100.0))
        if (total_vessel_volume > 0)
        else 0.0
    )
    total_lesions = measurements.count_lesions_3d(calc_mask)
    calcium_hu = image[calc_mask]
    total_mean_hu = float(np.mean(calcium_hu)) if calcium_hu.size else 0.0
    total_max_hu = float(np.max(calcium_hu)) if calcium_hu.size else 0.0
    agatston = measurements.compute_agatston_score(image, calc_mask, spacing)
    (grade, risk) = measurements.agatston_grade(agatston)
    (branches, branch_info) = topology.separate_branches(
        vessel_mask, min_branch_voxels = min_branch_voxels
    )
    branch_rows: List[models.BranchConcept] = []
    for bshort, bmask in branches.items():
        bcalc = (bmask & calc_mask)
        c = measurements.compute_centerline_concepts(
            bmask, spacing, stenosis_threshold = stenosis_threshold, calcium_mask = bcalc
        )
        bhu = image[bcalc]
        bvol = float((np.sum(bmask) * voxel_vol))
        bcalc_vol = float((np.sum(bcalc) * voxel_vol))
        burden = float(((bcalc_vol / bvol) * 100.0)) if (bvol > 0) else 0.0
        row = models.BranchConcept(
            case_id = pid,
            branch = bshort,
            branch_name = measurements.branch_long_name(bshort),
            branch_assignment_method = str(branch_info.get("method", "")),
            branch_voxels = int(np.sum(bmask)),
            vessel_volume_mm3 = float(bvol),
            vessel_length_mm = float(c.length_mm),
            vessel_length_method = str(c.method),
            vessel_diameter_mean_mm = float(c.mean_diameter_mm),
            vessel_diameter_median_mm = float(c.median_diameter_mm),
            vessel_diameter_min_mm = float(c.min_diameter_mm),
            vessel_diameter_max_mm = float(c.max_diameter_mm),
            stenosis_presence = int(c.stenosis_presence),
            stenosis_ratio = float(c.stenosis_ratio),
            stenosis_position_norm = float(c.stenosis_position_norm),
            stenosis_segment = str(c.stenosis_segment),
            lesion_length_mm = float(c.stenosis_length_mm),
            tortuosity = float(c.tortuosity),
            calcium_volume_mm3 = float(bcalc_vol),
            calcium_burden_pct = float(burden),
            calcium_lesion_count_3d = int(measurements.count_lesions_3d(bcalc)),
            calcium_mean_hu = float(np.mean(bhu)) if bhu.size else 0.0,
            calcium_max_hu = float(np.max(bhu)) if bhu.size else 0.0,
            centerline_point_count = int(c.point_count),
            centerline_quality_flag = str(c.quality_flag),
        )
        branch_rows.append(row)

    def branch_value(branch: str, attr: str, default: float = 0.0) -> float:
        for r in branch_rows:
            if (r.branch == branch):
                return float(getattr(r, attr))
        return float(default)

    torts = [float(r.tortuosity) for r in branch_rows if (float(r.tortuosity) > 0)]
    if branch_rows:
        best_sten = max(branch_rows, key = lambda r: float(r.stenosis_ratio))
        max_stenosis_ratio = float(best_sten.stenosis_ratio)
        max_stenosis_branch = str(best_sten.branch) if (max_stenosis_ratio > 0) else "none"
        sten_pos = (
            float(best_sten.stenosis_position_norm) if (max_stenosis_ratio > 0) else -1.0
        )
        sten_seg = str(best_sten.stenosis_segment) if (max_stenosis_ratio > 0) else "none"
    else:
        max_stenosis_ratio = 0.0
        max_stenosis_branch = "none"
        sten_pos = -1.0
        sten_seg = "none"
    warnings = []
    warnings.extend(branch_info.get("warnings", []))
    if branch_info.get("left_split", {}).get("warning"):
        warnings.append(branch_info.get("left_split", {}).get("warning"))
    case_row = models.CaseConcept(
        case_id = pid,
        source_img = str(img_path),
        source_label = str(lbl_path),
        spacing_x_mm = float(spacing[0]),
        spacing_y_mm = float(spacing[1]),
        spacing_z_mm = float(spacing[2]),
        total_vessel_volume_mm3 = float(total_vessel_volume),
        calcium_volume_mm3 = float(total_calc_volume),
        calcium_burden_pct = float(calcium_burden),
        calcium_lesion_count_3d = int(total_lesions),
        calcium_mean_hu = float(total_mean_hu),
        calcium_max_hu = float(total_max_hu),
        agatston_score = float(agatston),
        agatston_grade = str(grade),
        risk_level = str(risk),
        branch_count = int(len(branch_rows)),
        LAD_length_mm = branch_value("LAD", "vessel_length_mm"),
        RCA_length_mm = branch_value("RCA", "vessel_length_mm"),
        LAD_mean_diameter_mm = branch_value("LAD", "vessel_diameter_mean_mm"),
        RCA_mean_diameter_mm = branch_value("RCA", "vessel_diameter_mean_mm"),
        max_geometric_narrowing_index = float(max_stenosis_ratio),
        max_stenosis_branch = str(max_stenosis_branch),
        stenosis_position_norm = float(sten_pos),
        stenosis_segment = str(sten_seg),
        tortuosity_mean = float(np.mean(torts)) if torts else 0.0,
        tortuosity_max = float(np.max(torts)) if torts else 0.0,
        processing_status = "success",
        warning = "; ".join([str(w) for w in warnings if w]),
    )
    pd.DataFrame([asdict(case_row)]).to_csv(
        (case_out / "cta_concepts_case_level.csv"), index = False, encoding = "utf-8-sig"
    )
    pd.DataFrame([asdict(r) for r in branch_rows]).to_csv(
        (case_out / "cta_concepts_branch_level.csv"), index = False, encoding = "utf-8-sig"
    )
    storage.write_json((case_out / "cta_concepts_case_level.json"), asdict(case_row))
    storage.write_json((case_out / "branch_separation_info.json"), branch_info)
    storage.write_json(
        (case_out / "database_row.json"), storage.database_row_from_case(case_row)
    )
    visual_meta = {}
    if save_qa:
        visual_meta = visual_outputs.save_case_visual_outputs(
            pid, image, vessel_mask, calc_mask, branches, branch_rows, spacing, case_out
        )
    elapsed = (time.time() - start_t)
    paper_meta_paths = quality_tables.write_case_paperready_metadata(
        case_row, branch_rows, branch_info, visual_meta, case_out, elapsed
    )
    db_paths = storage.update_concept_database(case_row, runtime.DATABASE_ROOT)
    master_paths = {}
    if update_master:
        try:
            master_paths = storage.rebuild_master_database(
                out_dir, db_root = storage.infer_database_root_from_cases_root(out_dir)
            )
        except Exception:
            pass
    meta = {
        "case_id": pid,
        "elapsed_sec": float(elapsed),
        "output_dir": str(case_out),
        "branch_info": branch_info,
        "database": db_paths,
        "paperready_metadata": {k: str(v) for (k, v) in paper_meta_paths.items()},
        "master_database": master_paths,
        "visual_outputs": visual_meta,
    }
    del image, label_vol, vessel_mask, calc_mask
    gc.collect()
    return (case_row, branch_rows, meta)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond = 0).isoformat()

def canonical_parameter_dict(
    threshold_hu: float,
    min_calc_vol_mm3: float,
    min_branch_voxels: int,
    stenosis_threshold: float,
    save_qa: bool,
) -> Dict[str, Any]:
    return {
        "algorithm_version": runtime.ALGORITHM_VERSION,
        "threshold_hu": float(threshold_hu),
        "min_calc_vol_mm3": float(min_calc_vol_mm3),
        "min_branch_voxels": int(min_branch_voxels),
        "geometric_narrowing_threshold": float(stenosis_threshold),
        "save_qa": bool(save_qa),
        "visual_background_mode": runtime.visual_background_mode(),
        "high_hu_mode": "adaptive_peripheral_high_hu_candidate_v1",
        "narrowing_reliability_mode": "rule_based_multievidence_v1",
        "branch_confidence_mode": "heuristic_semantic_assignment_confidence_v1",
    }

def parameter_hash(params: Dict[str, Any]) -> str:
    compact = json.dumps(
        {k: params.get(k) for k in runtime.PARAMETER_HASH_KEYS},
        sort_keys = True,
        ensure_ascii = True,
        separators = (",", ":"),
    )
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]

def script_identity() -> Dict[str, Any]:
    try:
        script_path = Path(runtime.ENTRY_SCRIPT).resolve()
    except Exception:
        script_path = Path("concept_extraction_CTA.py")
    return {
        "script_filename": script_path.name,
        "script_path": str(script_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

def write_run_metadata(
    case_out: Path,
    case_id: str,
    img_path: Path,
    lbl_path: Path,
    params: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    meta = {
        "case_id": str(case_id),
        "algorithm_version": runtime.ALGORITHM_VERSION,
        "run_datetime_utc": now_utc_iso(),
        "parameter_hash": parameter_hash(params),
        "parameters": storage.json_safe(params),
        "input_dataset_root": str(Path(img_path).parent),
        "source_img": str(img_path),
        "source_label": str(lbl_path),
        "output_dir": str(case_out),
        "elapsed_sec": (
            float(elapsed_sec)
            if ((elapsed_sec is not None) and (not pd.isna(elapsed_sec)))
            else np.nan
        ),
        **script_identity(),
    }
    storage.write_json((case_out / "run_metadata.json"), meta)
    return meta

def read_run_metadata(case_dir: Path) -> Dict[str, Any]:
    return (storage.read_json_safe((Path(case_dir) / "run_metadata.json"), default = {}) or {})

def enrich_case_level_with_metadata(case_out: Path, meta: Dict[str, Any]) -> None:
    csv_path = (case_out / "cta_concepts_case_level.csv")
    json_path = (case_out / "cta_concepts_case_level.json")
    df = storage.read_csv_safe(csv_path)
    if not df.empty:
        for k in [
            "algorithm_version",
            "run_datetime_utc",
            "parameter_hash",
            "script_filename",
            "input_dataset_root",
        ]:
            df[k] = meta.get(k, "")
        if ("calcium_volume_mm3" in df.columns):
            df["high_hu_candidate_volume_mm3"] = pd.to_numeric(
                df["calcium_volume_mm3"], errors = "coerce"
            )
        if ("calcium_burden_pct" in df.columns):
            burden = pd.to_numeric(df["calcium_burden_pct"], errors = "coerce").fillna(0.0)
            df["high_hu_candidate_burden_pct"] = burden
            df["contrast_lumen_contamination_risk_flag"] = (burden > 5.0).astype(int)
        df["high_hu_interpretation_note"] = (
            "peripheral high-HU candidate in contrast CCTA; not non-contrast CAC ground truth"
        )
        df.to_csv(csv_path, index = False, encoding = "utf-8-sig")
        rec = df.iloc[0].to_dict()
        storage.write_json(json_path, rec)

def analyze_single_case(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    threshold_hu: float = runtime.DEFAULT_CALCIUM_THRESHOLD_HU,
    min_calc_vol_mm3: float = runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
    stenosis_threshold: float = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
    save_qa: bool = runtime.DEFAULT_QA_FIGURE,
    update_master: bool = True,
) -> Tuple[models.CaseConcept, List[models.BranchConcept], Dict[str, Any]]:
    params = canonical_parameter_dict(
        threshold_hu, min_calc_vol_mm3, min_branch_voxels, stenosis_threshold, save_qa
    )
    (case_row, branch_rows, meta) = extract_case_concepts(
        img_path,
        lbl_path,
        out_dir,
        threshold_hu = threshold_hu,
        min_calc_vol_mm3 = min_calc_vol_mm3,
        min_branch_voxels = min_branch_voxels,
        stenosis_threshold = stenosis_threshold,
        save_qa = save_qa,
        update_master = update_master,
    )
    case_out = Path(
        meta.get(
            "output_dir", (Path(out_dir) / storage.sanitize_case_id(str(case_row.case_id)))
        )
    )
    run_meta = write_run_metadata(
        case_out,
        str(case_row.case_id),
        Path(img_path),
        Path(lbl_path),
        params,
        elapsed_sec = meta.get("elapsed_sec"),
    )
    branch_info = (
        (storage.read_json_safe((case_out / "branch_separation_info.json"), default = {})
        or meta.get("branch_info", {}))
        or {}
    )
    enrich_case_level_with_metadata(case_out, run_meta)
    branch_tables.enrich_branch_level_outputs(case_out, branch_info)
    quality_tables.enrich_qa_with_metadata(case_out, run_meta)
    meta["run_metadata"] = run_meta
    return (case_row, branch_rows, meta)

def case_expected_output_dir(out_root: Path, case_id: str) -> Path:
    return (Path(out_root) / storage.sanitize_case_id(str(case_id)))

def case_is_completed_same_version(case_dir: Path, params: Dict[str, Any]) -> bool:
    if not case_dir.exists():
        return False
    meta = read_run_metadata(case_dir)
    if not meta:
        return False
    same_version = (str(meta.get("algorithm_version", "")) == runtime.ALGORITHM_VERSION)
    same_hash = (str(meta.get("parameter_hash", "")) == parameter_hash(params))
    case_exists = ((case_dir / "cta_concepts_case_level.csv").exists() and (
        case_dir / "cta_concepts_branch_level.csv"
    ).exists())
    if bool(params.get("save_qa", True)):
        visual_ok = (
            ((case_dir / "visual_outputs.json").exists()
            and (case_dir / "visuals").exists())
            and (case_dir / "videos").exists()
        )
    else:
        visual_ok = True
    return bool((((same_version and same_hash) and case_exists) and visual_ok))

def case_is_failed_or_review_needed(case_dir: Path) -> bool:
    q = storage.read_csv_safe((case_dir / "cta_concepts_qa.csv"))
    if q.empty:
        return True
    r = q.iloc[0]
    if (storage.to_int(r.get("processing_success", 1), 1) == 0):
        return True
    if (storage.to_int(r.get("manual_review_needed", 0), 0) == 1):
        return True
    if (storage.to_float(r.get("qa_score", 100.0), 100.0) < 90.0):
        return True
    return False
