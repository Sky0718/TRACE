from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from . import runtime, models, branch_tables, measurements, reliability, storage, topology

def _robustness_category_rank(label: Any) -> int:
    text = str((label or "")).lower()
    if (("70-99" in text) or ("99-100" in text)):
        return 3
    if ("50-69" in text):
        return 2
    if ("25-49" in text):
        return 1
    if ("1-24" in text):
        return 0
    return 0

def _robustness_numeric_summary(values: Any, prefix: str) -> Dict[str, Any]:
    arr = np.asarray(
        list(values) if isinstance(values, (list, tuple)) else values, dtype = np.float64
    )
    arr = arr[np.isfinite(arr)] if arr.size else np.asarray([], dtype = np.float64)
    if (arr.size == 0):
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q1": np.nan,
            f"{prefix}_q3": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_sd": float(np.std(arr, ddof = 1)) if (arr.size > 1) else 0.0,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q1": float(np.percentile(arr, 25)),
        f"{prefix}_q3": float(np.percentile(arr, 75)),
        f"{prefix}_p05": float(np.percentile(arr, 5)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }

def _robustness_mask_geometry(
    mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Dict[str, Any]:
    from scipy.ndimage import binary_erosion

    m = mask.astype(bool)
    sp = np.asarray(spacing, dtype = np.float64)
    voxel_vol = float(np.prod(sp))
    voxel_count = int(np.sum(m))
    out: Dict[str, Any] = {
        "mask_voxel_count": voxel_count,
        "mask_volume_mm3": float((voxel_count * voxel_vol)),
        "mask_component_count": (
            int(measurements.connected_components_3d(m)[1]) if (voxel_count > 0) else 0
        ),
        "surface_voxel_count": 0,
        "surface_volume_ratio": np.nan,
        "bbox_size_x_mm": np.nan,
        "bbox_size_y_mm": np.nan,
        "bbox_size_z_mm": np.nan,
        "bbox_volume_mm3": np.nan,
        "mask_compactness_proxy": np.nan,
    }
    if (voxel_count <= 0):
        return out
    try:
        surface = (m & ~binary_erosion(
            m, structure = np.ones((3, 3, 3), dtype = bool), iterations = 1
        ))
        surf_n = int(np.sum(surface))
        out["surface_voxel_count"] = surf_n
        out["surface_volume_ratio"] = float((surf_n / max(voxel_count, 1)))
        out["mask_compactness_proxy"] = float(
            (surf_n / max((voxel_count ** (2.0 / 3.0)), 1e-06))
        )
    except Exception:
        pass
    coords = np.argwhere(m)
    mins = coords.min(axis = 0)
    maxs = coords.max(axis = 0)
    size_mm = (((maxs - mins) + 1).astype(np.float64) * sp)
    out["bbox_size_x_mm"] = float(size_mm[0])
    out["bbox_size_y_mm"] = float(size_mm[1])
    out["bbox_size_z_mm"] = float(size_mm[2])
    out["bbox_volume_mm3"] = float(np.prod(size_mm))
    return out

def _robustness_profile_features(
    branch_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    calc_mask: Optional[np.ndarray],
    stenosis_threshold: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "profile_ok": 0,
        "profile_method": "",
        "profile_quality": "",
        "profile_point_count": 0,
        "profile_length_mm": 0.0,
        "peak_index": -1,
        "peak_distance_mm": np.nan,
        "peak_position_norm_profile": np.nan,
        "peak_geometric_narrowing_index_profile": 0.0,
        "diameter_at_peak_mm": np.nan,
        "reference_diameter_at_peak_mm": np.nan,
        "area_at_peak_mm2": np.nan,
        "area_reference_at_peak_mm2": np.nan,
        "area_evidence_at_peak_profile": np.nan,
        "high_hu_proximity_at_peak_profile": np.nan,
        "ratio_auc_mm": 0.0,
        "ratio_mean": np.nan,
        "ratio_median": np.nan,
        "ratio_p95": np.nan,
        "length_ratio_ge_25_mm": 0.0,
        "length_ratio_ge_50_mm": 0.0,
        "fraction_ratio_ge_25": 0.0,
        "fraction_ratio_ge_50": 0.0,
        "diameter_gradient_max_abs_mm_per_mm": np.nan,
        "diameter_gradient_mean_abs_mm_per_mm": np.nan,
    }
    try:
        prof = measurements.centerline_profile_arrays(
            branch_mask, spacing, calcium_mask = calc_mask
        )
        out["profile_method"] = str(prof.get("method", ""))
        out["profile_quality"] = str(prof.get("quality", ""))
        if not prof.get("ok"):
            return out
        cum = np.asarray(prof.get("cum", []), dtype = np.float64)
        diameter = np.asarray(prof.get("diameter", []), dtype = np.float64)
        ref = np.asarray(prof.get("reference", []), dtype = np.float64)
        ratio = np.asarray(prof.get("ratio", []), dtype = np.float64)
        area = np.asarray(prof.get("area", []), dtype = np.float64)
        area_ref = np.asarray(prof.get("area_reference", []), dtype = np.float64)
        area_ratio = np.asarray(
            prof.get("ratio_area", np.zeros_like(ratio)), dtype = np.float64
        )
        high_hu = np.asarray(
            prof.get("calcium_penalty", np.zeros_like(ratio)), dtype = np.float64
        )
        if ((cum.size == 0) or (ratio.size == 0)):
            return out
        n = int(min(cum.size, ratio.size))
        cum = cum[:n]
        ratio = ratio[:n]
        diameter = diameter[:n] if (diameter.size >= n) else np.full(n, np.nan)
        ref = ref[:n] if (ref.size >= n) else np.full(n, np.nan)
        area = area[:n] if (area.size >= n) else np.full(n, np.nan)
        area_ref = area_ref[:n] if (area_ref.size >= n) else np.full(n, np.nan)
        area_ratio = area_ratio[:n] if (area_ratio.size >= n) else np.full(n, np.nan)
        high_hu = high_hu[:n] if (high_hu.size >= n) else np.full(n, np.nan)
        idx = int(np.nanargmax(ratio)) if np.any(np.isfinite(ratio)) else 0
        total = float(cum[-1]) if cum.size else 0.0
        if (cum.size > 1):
            step = np.diff(cum)
            step = np.where((step > 0), step, np.nan)
            seg_mid_ratio = (0.5 * (ratio[:-1] + ratio[1:]))
            length_ge25 = float(np.nansum(step[(seg_mid_ratio >= 0.25)]))
            length_ge50 = float(np.nansum(step[(seg_mid_ratio >= 0.5)]))
            auc = float(np.nansum((step * seg_mid_ratio)))
            grad = (
                np.gradient(diameter, cum)
                if ((diameter.size == cum.size) and (total > 0))
                else np.asarray([], dtype = float)
            )
            grad_abs = (
                np.abs(grad[np.isfinite(grad)])
                if grad.size
                else np.asarray([], dtype = float)
            )
        else:
            length_ge25 = 0.0
            length_ge50 = 0.0
            auc = 0.0
            grad_abs = np.asarray([], dtype = float)
        out.update(
            {
                "profile_ok": 1,
                "profile_point_count": n,
                "profile_length_mm": float(total),
                "peak_index": int(idx),
                "peak_distance_mm": float(cum[idx]) if (idx < cum.size) else np.nan,
                "peak_position_norm_profile": (
                    float((cum[idx] / max(total, 1e-06)))
                    if ((idx < cum.size) and (total > 0))
                    else np.nan
                ),
                "peak_geometric_narrowing_index_profile": (
                    float(ratio[idx]) if (idx < ratio.size) else 0.0
                ),
                "diameter_at_peak_mm": (
                    float(diameter[idx]) if (idx < diameter.size) else np.nan
                ),
                "reference_diameter_at_peak_mm": (
                    float(ref[idx]) if (idx < ref.size) else np.nan
                ),
                "area_at_peak_mm2": float(area[idx]) if (idx < area.size) else np.nan,
                "area_reference_at_peak_mm2": (
                    float(area_ref[idx]) if (idx < area_ref.size) else np.nan
                ),
                "area_evidence_at_peak_profile": (
                    float(area_ratio[idx]) if (idx < area_ratio.size) else np.nan
                ),
                "high_hu_proximity_at_peak_profile": (
                    float(high_hu[idx]) if (idx < high_hu.size) else np.nan
                ),
                "ratio_auc_mm": float(auc),
                "ratio_mean": float(np.nanmean(ratio)),
                "ratio_median": float(np.nanmedian(ratio)),
                "ratio_p95": float(np.nanpercentile(ratio, 95)),
                "length_ratio_ge_25_mm": float(length_ge25),
                "length_ratio_ge_50_mm": float(length_ge50),
                "fraction_ratio_ge_25": float((length_ge25 / max(total, 1e-06))),
                "fraction_ratio_ge_50": float((length_ge50 / max(total, 1e-06))),
                "diameter_gradient_max_abs_mm_per_mm": (
                    float(np.nanmax(grad_abs)) if grad_abs.size else np.nan
                ),
                "diameter_gradient_mean_abs_mm_per_mm": (
                    float(np.nanmean(grad_abs)) if grad_abs.size else np.nan
                ),
            }
        )
        out.update(_robustness_numeric_summary(diameter, "diameter_profile_mm"))
        out.update(_robustness_numeric_summary(ref, "reference_profile_mm"))
        out.update(_robustness_numeric_summary(ratio, "narrowing_profile"))
    except Exception as exc:
        out["profile_error"] = f"{type(exc).__name__}: {exc}"
    return out

def _robustness_variant_tables(
    case_id: str,
    variant: str,
    image: Optional[np.ndarray],
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    threshold_hu: float,
    min_calc_vol_mm3: float,
    min_branch_voxels: int,
    stenosis_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    voxel_vol = float(np.prod(np.asarray(spacing, dtype = np.float64)))
    if ((image is not None) and (np.shape(image) == np.shape(mask))):
        calc_mask = measurements.extract_calcification(
            image, mask.astype(bool), threshold_hu, voxel_vol, min_calc_vol_mm3
        )
    else:
        calc_mask = np.zeros_like(mask, dtype = bool)
    (branches, branch_info) = topology.separate_branches(
        mask.astype(bool), min_branch_voxels = min_branch_voxels
    )
    branch_objects: List[models.BranchConcept] = []
    branch_extra: Dict[str, Dict[str, Any]] = {}
    for branch_name, bmask in branches.items():
        bcalc = (bmask & calc_mask)
        concept = measurements.compute_centerline_concepts(
            bmask, spacing, stenosis_threshold = stenosis_threshold, calcium_mask = bcalc
        )
        bvol = float((np.sum(bmask) * voxel_vol))
        bcalc_vol = float((np.sum(bcalc) * voxel_vol))
        bhu = (
            image[bcalc]
            if (((image is not None) and (np.shape(image) == np.shape(mask))) and np.any(bcalc))
            else np.asarray([], dtype = np.float32)
        )
        row = models.BranchConcept(
            case_id = case_id,
            branch = str(branch_name),
            branch_name = measurements.branch_long_name(str(branch_name)),
            branch_assignment_method = str(branch_info.get("method", "")),
            branch_voxels = int(np.sum(bmask)),
            vessel_volume_mm3 = float(bvol),
            vessel_length_mm = float(concept.length_mm),
            vessel_length_method = str(concept.method),
            vessel_diameter_mean_mm = float(concept.mean_diameter_mm),
            vessel_diameter_median_mm = float(concept.median_diameter_mm),
            vessel_diameter_min_mm = float(concept.min_diameter_mm),
            vessel_diameter_max_mm = float(concept.max_diameter_mm),
            stenosis_presence = int(concept.stenosis_presence),
            stenosis_ratio = float(concept.stenosis_ratio),
            stenosis_position_norm = float(concept.stenosis_position_norm),
            stenosis_segment = str(concept.stenosis_segment),
            lesion_length_mm = float(concept.stenosis_length_mm),
            tortuosity = float(concept.tortuosity),
            calcium_volume_mm3 = float(bcalc_vol),
            calcium_burden_pct = float(((bcalc_vol / bvol) * 100.0)) if (bvol > 0) else 0.0,
            calcium_lesion_count_3d = int(measurements.count_lesions_3d(bcalc)),
            calcium_mean_hu = float(np.mean(bhu)) if bhu.size else 0.0,
            calcium_max_hu = float(np.max(bhu)) if bhu.size else 0.0,
            centerline_point_count = int(concept.point_count),
            centerline_quality_flag = str(concept.quality_flag),
        )
        branch_objects.append(row)
        branch_extra[str(branch_name)] = _robustness_profile_features(
            bmask, spacing, bcalc, stenosis_threshold
        )
    branch_df = pd.DataFrame([asdict(x) for x in branch_objects])
    if not branch_df.empty:
        try:
            branch_df = branch_tables.compute_branch_assignment_confidence(
                branch_df, branch_info
            )
        except Exception:
            branch_df["branch_assignment_confidence"] = "not_available"
            branch_df["branch_assignment_confidence_score"] = np.nan
        try:
            rel_df = reliability.compute_case_narrowing_reliability(
                case_id, branches, branch_objects, calc_mask, spacing
            )
            if ((rel_df is not None) and (not rel_df.empty)):
                branch_df = branch_df.merge(rel_df, on = ["case_id", "branch"], how = "left")
        except Exception as exc:
            branch_df["candidate_reliability_error"] = f"{type(exc).__name__}: {exc}"
        try:
            branch_df = reliability._add_primary_statistics_flags_to_branch_df(branch_df)
        except Exception:
            if ("candidate_reliable_for_primary_statistics" not in branch_df.columns):
                branch_df["candidate_reliable_for_primary_statistics"] = 0
            if ("candidate_primary_exclusion_reason" not in branch_df.columns):
                branch_df["candidate_primary_exclusion_reason"] = "not_evaluated"
        for col in ["stenosis_ratio", "geometric_narrowing_index"]:
            if (col not in branch_df.columns):
                branch_df[col] = pd.to_numeric(
                    branch_df.get("stenosis_ratio", 0), errors = "coerce"
                ).fillna(0.0)
        branch_df["geometric_narrowing_index"] = pd.to_numeric(
            branch_df.get(
                "geometric_narrowing_index", branch_df.get("stenosis_ratio", 0.0)
            ),
            errors = "coerce",
        ).fillna(0.0)
        branch_df["geometric_narrowing_category"] = branch_df[
            "geometric_narrowing_index"
        ].apply(reliability.stenosis_interval_label)
    branch_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for _, r in branch_df.iterrows() if not branch_df.empty else []:
        bname = str(r.get("branch", ""))
        rec = {str(k): storage.json_safe(v) for (k, v) in r.to_dict().items()}
        rec["variant"] = str(variant)
        rec["robustness_analysis_version"] = runtime.ROBUSTNESS_ANALYSIS_VERSION
        rec.update(branch_extra.get(bname, {}))
        branch_rows.append(rec)
        candidate_present = int(
            storage.to_int(rec.get("candidate_present", rec.get("stenosis_presence", 0)), 0)
        )
        gni = storage.to_float(
            rec.get("geometric_narrowing_index", rec.get("stenosis_ratio", 0.0)), 0.0
        )
        if ((candidate_present > 0) or (gni > 0.0)):
            cand = {k: rec.get(k) for k in rec.keys()}
            cand["candidate_key"] = f"{case_id}|{bname}"
            cand["geometric_narrowing_index"] = float(gni)
            cand["geometric_narrowing_category"] = reliability.stenosis_interval_label(gni)
            cand["geometric_narrowing_category_rank"] = _robustness_category_rank(
                cand["geometric_narrowing_category"]
            )
            candidate_rows.append(cand)
    lengths = (
        pd.to_numeric(
            branch_df.get("vessel_length_mm", pd.Series(dtype = float)), errors = "coerce"
        )
        if not branch_df.empty
        else pd.Series(dtype = float)
    )
    torts = (
        pd.to_numeric(branch_df.get("tortuosity", pd.Series(dtype = float)), errors = "coerce")
        if not branch_df.empty
        else pd.Series(dtype = float)
    )
    diams = (
        pd.to_numeric(
            branch_df.get("vessel_diameter_mean_mm", pd.Series(dtype = float)),
            errors = "coerce",
        )
        if not branch_df.empty
        else pd.Series(dtype = float)
    )
    narrows = (
        pd.to_numeric(
            branch_df.get("geometric_narrowing_index", pd.Series(dtype = float)),
            errors = "coerce",
        )
        if not branch_df.empty
        else pd.Series(dtype = float)
    )
    primary = (
        pd.to_numeric(
            branch_df.get(
                "candidate_reliable_for_primary_statistics", pd.Series(dtype = int)
            ),
            errors = "coerce",
        )
        .fillna(0)
        .astype(int)
        if not branch_df.empty
        else pd.Series(dtype = int)
    )
    broad = (
        pd.to_numeric(
            branch_df.get(
                "candidate_reliable_for_broad_statistics",
                branch_df.get("candidate_reliable_for_statistics", pd.Series(dtype = int)),
            ),
            errors = "coerce",
        )
        .fillna(0)
        .astype(int)
        if not branch_df.empty
        else pd.Series(dtype = int)
    )
    present = (
        pd.to_numeric(
            branch_df.get(
                "candidate_present",
                (narrows > 0).astype(int) if len(narrows) else pd.Series(dtype = int),
            ),
            errors = "coerce",
        )
        .fillna(0)
        .astype(int)
        if not branch_df.empty
        else pd.Series(dtype = int)
    )
    max_gni = (
        float(np.nanmax(narrows)) if (len(narrows) and np.any(np.isfinite(narrows))) else 0.0
    )
    mask_geom = _robustness_mask_geometry(mask.astype(bool), spacing)
    case_row: Dict[str, Any] = {
        "case_id": str(case_id),
        "variant": str(variant),
        "status": "success",
        "algorithm_version": runtime.ALGORITHM_VERSION,
        "robustness_analysis_version": runtime.ROBUSTNESS_ANALYSIS_VERSION,
        **mask_geom,
        "branch_count": int(len(branches)),
        "valid_branch_count": int(np.sum((lengths.fillna(0) > 0))) if len(lengths) else 0,
        "branches_detected": "|".join(sorted([str(x) for x in branches.keys()])),
        "LM_detected": int(("LM" in branches)),
        "LAD_detected": int(("LAD" in branches)),
        "LCX_detected": int(("LCX" in branches)),
        "RCA_detected": int(("RCA" in branches)),
        "OTHER_branch_count": int(
            sum((1 for b in branches.keys() if str(b).startswith("OTHER")))
        ),
        "total_branch_length_mm": float(np.nansum(lengths)) if len(lengths) else 0.0,
        "mean_branch_length_mm": float(np.nanmean(lengths)) if len(lengths) else np.nan,
        "median_branch_length_mm": float(np.nanmedian(lengths)) if len(lengths) else np.nan,
        "mean_branch_diameter_mm": float(np.nanmean(diams)) if len(diams) else np.nan,
        "median_branch_diameter_mm": float(np.nanmedian(diams)) if len(diams) else np.nan,
        "mean_tortuosity": float(np.nanmean(torts)) if len(torts) else np.nan,
        "max_tortuosity": float(np.nanmax(torts)) if len(torts) else np.nan,
        "max_geometric_narrowing_index": float(max_gni),
        "max_geometric_narrowing_category": reliability.stenosis_interval_label(max_gni),
        "max_geometric_narrowing_category_rank": _robustness_category_rank(
            reliability.stenosis_interval_label(max_gni)
        ),
        "candidate_count": int(np.sum(present)) if len(present) else 0,
        "broad_reliable_candidate_count": (
            int(np.sum(((present > 0) & (broad > 0)))) if (len(present) and len(broad)) else 0
        ),
        "primary_reliable_candidate_count": (
            int(np.sum(((present > 0) & (primary > 0))))
            if (len(present) and len(primary))
            else 0
        ),
        "primary_excluded_candidate_count": (
            int(np.sum(((present > 0) & (primary <= 0))))
            if (len(present) and len(primary))
            else 0
        ),
        "calcium_volume_mm3": float((np.sum(calc_mask) * voxel_vol)),
        "calcium_burden_pct": float(((np.sum(calc_mask) / max(np.sum(mask), 1)) * 100.0)),
        "calcium_component_count": int(measurements.count_lesions_3d(calc_mask)),
        "branch_info_json": json.dumps(
            storage.json_safe(branch_info), ensure_ascii = False, sort_keys = True
        ),
    }
    for b in ["LM", "LAD", "LCX", "RCA"]:
        sub = (
            branch_df[(branch_df["branch"].astype(str) == b)]
            if (not branch_df.empty and ("branch" in branch_df.columns))
            else pd.DataFrame()
        )
        case_row[f"{b}_present"] = int(not sub.empty)
        for src, dst in [
            ("vessel_length_mm", "length_mm"),
            ("vessel_diameter_mean_mm", "mean_diameter_mm"),
            ("vessel_diameter_min_mm", "min_diameter_mm"),
            ("tortuosity", "tortuosity"),
            ("geometric_narrowing_index", "geometric_narrowing_index"),
            ("candidate_reliability_score", "candidate_reliability_score"),
            ("candidate_reliable_for_primary_statistics", "primary_candidate"),
        ]:
            case_row[f"{b}_{dst}"] = (
                storage.to_float(sub.iloc[0][src], np.nan)
                if (not sub.empty and (src in sub.columns))
                else np.nan
            )
        case_row[f"{b}_narrowing_category"] = (
            reliability.stenosis_interval_label(
                storage.to_float(case_row.get(f"{b}_geometric_narrowing_index", 0.0), 0.0)
            )
            if not sub.empty
            else "not_detected"
        )
        case_row[f"{b}_assignment_confidence"] = (
            str(sub.iloc[0].get("branch_assignment_confidence", "not_detected"))
            if not sub.empty
            else "not_detected"
        )
    return (case_row, branch_rows, candidate_rows)

def _robustness_delta_record(
    base: Dict[str, Any], pert: Dict[str, Any], id_fields: List[str], metrics: List[str]
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {k: pert.get(k, base.get(k, "")) for k in id_fields}
    rec["original_variant"] = "original"
    rec["perturbation_variant"] = str(pert.get("variant", ""))
    for k in metrics:
        ov = storage.to_float(base.get(k, np.nan), np.nan)
        pv = storage.to_float(pert.get(k, np.nan), np.nan)
        rec[f"original_{k}"] = ov
        rec[f"perturbed_{k}"] = pv
        rec[f"delta_{k}"] = (
            float((pv - ov)) if (np.isfinite(ov) and np.isfinite(pv)) else np.nan
        )
        rec[f"abs_delta_{k}"] = (
            float(abs((pv - ov))) if (np.isfinite(ov) and np.isfinite(pv)) else np.nan
        )
        rec[f"relative_delta_{k}"] = (
            float(((pv - ov) / max(abs(ov), 1e-06)))
            if (np.isfinite(ov) and np.isfinite(pv))
            else np.nan
        )
        rec[f"abs_relative_delta_{k}"] = (
            float((abs((pv - ov)) / max(abs(ov), 1e-06)))
            if (np.isfinite(ov) and np.isfinite(pv))
            else np.nan
        )
    return rec
