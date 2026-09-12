from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import cta_legacy_backend_v1_0_15 as legacy
import stenosis_v2_core as core

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return (value if np.isfinite(value) else None)
    if isinstance(obj, float):
        return (obj if math.isfinite(obj) else None)
    return obj

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii = False, indent = 2), encoding = "utf-8")

def add_patient_coordinates(
    df: pd.DataFrame, voxel_to_patient_affine: Optional[Sequence[Sequence[float]]]
) -> pd.DataFrame:
    if (df.empty or (voxel_to_patient_affine is None)):
        return df
    try:
        affine = np.asarray(voxel_to_patient_affine, dtype = np.float64)
        if (affine.shape != (4, 4)):
            return df
    except Exception:
        return df
    out = df.copy()
    for prefix in ("start", "peak", "end"):
        cols = [f"{prefix}_voxel_x", f"{prefix}_voxel_y", f"{prefix}_voxel_z"]
        if not all((c in out.columns) for c in cols):
            continue
        xyz = out[cols].apply(pd.to_numeric, errors = "coerce").to_numpy(dtype = np.float64)
        finite = np.all(np.isfinite(xyz), axis = 1)
        patient = np.full_like(xyz, np.nan, dtype = np.float64)
        if np.any(finite):
            hom = np.column_stack([xyz[finite], np.ones(np.sum(finite), dtype = np.float64)])
            patient[finite] = (hom @ affine.T)[:, :3]
        out[f"{prefix}_patient_x_mm"] = patient[:, 0]
        out[f"{prefix}_patient_y_mm"] = patient[:, 1]
        out[f"{prefix}_patient_z_mm"] = patient[:, 2]
    return out

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    temp = path.with_suffix((path.suffix + ".tmp"))
    df.to_csv(temp, index = False, encoding = "utf-8-sig")
    temp.replace(path)

def append_replace_case(
    master_path: Path, new_df: pd.DataFrame, case_id: str, sort_columns: Sequence[str]
) -> None:
    if master_path.exists():
        try:
            old = pd.read_csv(master_path)
        except Exception:
            old = pd.DataFrame()
        if ("case_id" in old.columns):
            old = old[(old["case_id"].astype(str) != str(case_id))]
        merged = pd.concat([old, new_df], ignore_index = True, sort = False)
    else:
        merged = new_df.copy()
    usable = [c for c in sort_columns if (c in merged.columns)]
    if usable:
        merged = merged.sort_values(usable, kind = "mergesort")
    atomic_csv(merged, master_path)

FORMAL_CANDIDATE_STATUSES = {"formal", "formal_limited_branch_coverage"}

def partition_candidate_outputs(
    candidate_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if candidate_df.empty:
        empty = candidate_df.copy()
        report = {
            "candidate_count": 0,
            "formal_count": 0,
            "rejected_count": 0,
            "not_assessable_count": 0,
            "partition_complete": 1,
            "partition_mutually_exclusive": 1,
        }
        return empty, empty, empty, empty, report
    df = candidate_df.copy()
    for col, default in (
        ("accepted", 0),
        ("assessment_status", "assessable"),
        ("clinical_output_status", ""),
        ("final_decision", ""),
        ("rejection_reasons", ""),
        ("not_assessable_reason", ""),
        ("coverage_scope", "whole_branch"),
        ("status_consistency_checked", 0),
        ("final_state_version", "2.0.10"),
    ):
        if (col not in df.columns):
            df[col] = default
    accepted = pd.to_numeric(df["accepted"], errors = "coerce").fillna(0).astype(int)
    status = df["clinical_output_status"].fillna("").astype(str)
    assessment = df["assessment_status"].fillna("assessable").astype(str)
    not_reason = df["not_assessable_reason"].fillna("").astype(str)

    formal_mask = accepted.eq(1)
    local_mask = (formal_mask & (
        (
            status.eq("formal_limited_branch_coverage") | assessment.eq("locally_assessable_in_incomplete_branch")
        ) | df["coverage_scope"].fillna("").astype(str).eq("local_only")
    ))
    df.loc[(formal_mask & ~local_mask), "clinical_output_status"] = "formal"
    df.loc[(formal_mask & ~local_mask), "assessment_status"] = "assessable"
    df.loc[(formal_mask & ~local_mask), "coverage_scope"] = "whole_branch"
    df.loc[local_mask, "clinical_output_status"] = "formal_limited_branch_coverage"
    df.loc[local_mask, "assessment_status"] = "locally_assessable_in_incomplete_branch"
    df.loc[local_mask, "coverage_scope"] = "local_only"
    df.loc[formal_mask, "rejection_reasons"] = ""
    df.loc[formal_mask, "not_assessable_reason"] = ""

    nonformal = ~formal_mask
    not_assessable_mask = (nonformal & (
        (assessment.str.startswith("not_assessable") | status.eq("not_assessable")) | not_reason.str.len().gt(0)
    ))
    rejected_mask = (nonformal & ~not_assessable_mask)
    df.loc[not_assessable_mask, "clinical_output_status"] = "not_assessable"
    df.loc[not_assessable_mask, "assessment_status"] = "not_assessable"
    df.loc[not_assessable_mask, "coverage_scope"] = "insufficient"
    df.loc[
        (not_assessable_mask & df["final_decision"].fillna("").astype(str).eq("")), "final_decision"
    ] = "not_assessable_v2_0_10"
    df.loc[rejected_mask, "clinical_output_status"] = "rejected"
    df.loc[rejected_mask, "assessment_status"] = "assessable"
    df.loc[
        (rejected_mask & df["final_decision"].fillna("").astype(str).eq("")), "final_decision"
    ] = "rejected_v2_0_10"
    df["output_partition"] = df["clinical_output_status"].astype(str)
    df["status_consistency_checked"] = 1
    df["final_state_version"] = "2.0.10"

    formal_df = df[
        (
            df["clinical_output_status"].isin(FORMAL_CANDIDATE_STATUSES) & pd.to_numeric(df["accepted"], errors = "coerce").fillna(0).astype(int).eq(1)
        )
    ].copy()
    not_assessable_df = df[df["clinical_output_status"].eq("not_assessable")].copy()
    rejected_df = df[df["clinical_output_status"].eq("rejected")].copy()
    partition_total = ((len(formal_df) + len(not_assessable_df)) + len(rejected_df))
    report = {
        "candidate_count": int(len(df)),
        "formal_count": int(len(formal_df)),
        "formal_whole_branch_count": int(np.sum(formal_df["clinical_output_status"].eq("formal"))),
        "formal_local_only_count": int(
            np.sum(formal_df["clinical_output_status"].eq("formal_limited_branch_coverage"))
        ),
        "rejected_count": int(len(rejected_df)),
        "not_assessable_count": int(len(not_assessable_df)),
        "partition_complete": int((partition_total == len(df))),
        "partition_mutually_exclusive": 1,
        "partition_total": int(partition_total),
    }
    if (partition_total != len(df)):
        raise RuntimeError(
            f"Candidate output partition mismatch: candidates={len(df)}, partitioned={partition_total}"
        )
    return df, formal_df, rejected_df, not_assessable_df, report

def build_stenosis_review_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        out = candidate_df.copy()
        out["review_category"] = pd.Series(dtype = str)
        return out
    df = candidate_df.copy()
    for col, default in (
        ("accepted", 0),
        ("borderline_review_candidate", 0),
        ("segment_review_required", 0),
        ("resolution_uncertainty_applied", 0),
        ("blooming_overestimate_flag", 0),
        ("severity_uncertainty_reason", ""),
        ("clinical_review_priority", "routine"),
        ("borderline_review_reason", ""),
        ("additional_lesion_review_required", 0),
        ("additional_lesion_review_candidate", 0),
        ("additional_lesion_review_reason", ""),
        ("adjacent_severe_shoulder_flag", 0),
        ("localization_audit_status", ""),
    ):
        if (col not in df.columns):
            df[col] = default
    accepted = pd.to_numeric(df["accepted"], errors = "coerce").fillna(0).astype(int).eq(1)
    borderline = (
        pd.to_numeric(df["borderline_review_candidate"], errors = "coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    additional_candidate = (
        pd.to_numeric(df["additional_lesion_review_candidate"], errors = "coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    formal_uncertain = (accepted & (
        (
            (
                (
                    (
                        pd.to_numeric(df["segment_review_required"], errors = "coerce")
                        .fillna(0)
                        .astype(int)
                        .eq(1) | pd.to_numeric(df["resolution_uncertainty_applied"], errors = "coerce")
                        .fillna(0)
                        .astype(int)
                        .eq(1)
                    ) | pd.to_numeric(df["blooming_overestimate_flag"], errors = "coerce")
                    .fillna(0)
                    .astype(int)
                    .eq(1)
                ) | df["severity_uncertainty_reason"].fillna("").astype(str).str.len().gt(0)
            ) | ~df["clinical_review_priority"].fillna("routine").astype(str).eq("routine")
        ) | pd.to_numeric(df["additional_lesion_review_required"], errors = "coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    ))
    review = df[((borderline | additional_candidate) | formal_uncertain)].copy()
    if review.empty:
        review["review_category"] = pd.Series(dtype = str)
        return review
    categories: List[str] = []
    for _, row in review.iterrows():
        is_formal = (int((pd.to_numeric(row.get("accepted", 0), errors = "coerce") or 0)) == 1)
        if (not is_formal and (
            int(
                (
                    pd.to_numeric(row.get("additional_lesion_review_candidate", 0), errors = "coerce")
                    or 0
                )
            )
            == 1
        )):
            categories.append("additional_possible_lesion_review")
        elif (not is_formal and (
            int((pd.to_numeric(row.get("adjacent_severe_shoulder_flag", 0), errors = "coerce") or 0))
            == 1
        )):
            categories.append("adjacent_mild_shoulder_review_only")
        elif not is_formal:
            categories.append(
                (
                    str(row.get("borderline_review_reason", "borderline_nonformal_candidate"))
                    or "borderline_nonformal_candidate"
                )
            )
        elif (
            int((pd.to_numeric(row.get("blooming_overestimate_flag", 0), errors = "coerce") or 0))
            == 1
        ):
            categories.append("formal_blooming_adjusted")
        elif (
            int((pd.to_numeric(row.get("resolution_uncertainty_applied", 0), errors = "coerce") or 0))
            == 1
        ):
            categories.append("formal_resolution_uncertainty")
        elif (
            int(
                (
                    pd.to_numeric(row.get("additional_lesion_review_required", 0), errors = "coerce")
                    or 0
                )
            )
            == 1
        ):
            categories.append("formal_additional_lesion_review")
        elif (int((pd.to_numeric(row.get("segment_review_required", 0), errors = "coerce") or 0)) == 1):
            categories.append("formal_localization_review")
        else:
            categories.append("formal_review_requested")
    review["review_category"] = categories
    sort_cols = [
        col for col in ("case_id", "branch", "peak_distance_mm") if (col in review.columns)
    ]
    if sort_cols:
        review = review.sort_values(sort_cols, kind = "mergesort")
    return review

def synchronize_branch_results_with_formal_lesions(
    branch_df: pd.DataFrame, formal_df: pd.DataFrame
) -> pd.DataFrame:
    if branch_df.empty:
        return branch_df
    out = branch_df.copy()
    if ("lesion_count" in out.columns):
        out["detector_lesion_count_before_export_sync"] = (
            pd.to_numeric(out["lesion_count"], errors = "coerce").fillna(0).astype(int)
        )
    else:
        out["detector_lesion_count_before_export_sync"] = 0
    for idx, row in out.iterrows():
        branch = str(row.get("branch", ""))
        sub = (
            formal_df[formal_df.get("branch", pd.Series(dtype = str)).astype(str).eq(branch)]
            if not formal_df.empty
            else pd.DataFrame()
        )
        out.at[idx, "lesion_count"] = int(len(sub))
        out.at[idx, "formal_lesion_count"] = int(len(sub))
        local_count = (
            int(
                np.sum(
                    sub.get("clinical_output_status", pd.Series(dtype = str))
                    .astype(str)
                    .eq("formal_limited_branch_coverage")
                )
            )
            if not sub.empty
            else 0
        )
        out.at[idx, "local_only_lesion_count"] = local_count
        out.at[idx, "formal_segment_review_count"] = (
            int(
                pd.to_numeric(
                    sub.get("segment_review_required", pd.Series(dtype = float)), errors = "coerce"
                )
                .fillna(0)
                .astype(int)
                .sum()
            )
            if not sub.empty
            else 0
        )
        out.at[idx, "additional_lesion_review_count"] = (
            int(
                pd.to_numeric(
                    sub.get("additional_lesion_review_required", pd.Series(dtype = float)),
                    errors = "coerce",
                )
                .fillna(0)
                .astype(int)
                .sum()
            )
            if not sub.empty
            else 0
        )
        out.at[idx, "localization_review_lesion_count"] = (
            int(
                np.sum(
                    (
                        pd.to_numeric(
                            sub.get("segment_review_required", pd.Series(dtype = float)),
                            errors = "coerce",
                        )
                        .fillna(0)
                        .astype(int)
                        .eq(1) | pd.to_numeric(
                            sub.get("additional_lesion_review_required", pd.Series(dtype = float)),
                            errors = "coerce",
                        )
                        .fillna(0)
                        .astype(int)
                        .eq(1)
                    )
                )
            )
            if not sub.empty
            else 0
        )
        if (local_count > 0):
            out.at[idx, "coverage_scope"] = "local_only"
            out.at[idx, "whole_branch_negative_interpretation_valid"] = 0
        elif (str(row.get("branch_assessability", "")) == "assessable"):
            out.at[idx, "coverage_scope"] = "whole_branch"
            out.at[idx, "whole_branch_negative_interpretation_valid"] = 1
        else:
            out.at[idx, "coverage_scope"] = "insufficient"
            out.at[idx, "whole_branch_negative_interpretation_valid"] = 0
        out.at[idx, "final_state_version"] = "2.0.10"
        if sub.empty:
            out.at[idx, "maximum_stenosis_ratio"] = 0.0
            out.at[idx, "maximum_stenosis_interval"] = "0%"
            out.at[idx, "maximum_stenosis_segment"] = "none"
            out.at[idx, "maximum_stenosis_segment_span"] = "none"
            out.at[idx, "maximum_stenosis_segment_status"] = ""
            out.at[idx, "maximum_stenosis_position_norm"] = -1.0
        else:
            ratios = pd.to_numeric(sub["stenosis_ratio"], errors = "coerce")
            best = sub.loc[ratios.idxmax()]
            out.at[idx, "maximum_stenosis_ratio"] = float(best.get("stenosis_ratio", 0.0))
            out.at[idx, "maximum_stenosis_interval"] = str(best.get("stenosis_interval", ""))
            out.at[idx, "maximum_stenosis_segment"] = str(
                best.get("reported_segment", best.get("peak_segment", "unknown"))
            )
            out.at[idx, "maximum_stenosis_segment_span"] = str(
                best.get("reported_segment_span", best.get("segment_span", "unknown"))
            )
            out.at[idx, "maximum_stenosis_segment_status"] = str(
                best.get("reported_segment_status", "")
            )
            out.at[idx, "maximum_stenosis_position_norm"] = float(
                best.get("peak_position_norm", best.get("position_norm", -1.0))
            )
    return out

def centerline_dataframe(centerline: core.CenterlineResult) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i in range(centerline.path_voxel.shape[0]):
        rows.append(
            {
                "index": int(i),
                "distance_mm": float(centerline.cumulative_mm[i]),
                "voxel_x": float(centerline.path_voxel[i, 0]),
                "voxel_y": float(centerline.path_voxel[i, 1]),
                "voxel_z": float(centerline.path_voxel[i, 2]),
                "physical_x_mm": float(centerline.path_physical_mm[i, 0]),
                "physical_y_mm": float(centerline.path_physical_mm[i, 1]),
                "physical_z_mm": float(centerline.path_physical_mm[i, 2]),
                "tangent_x": float(centerline.tangent[i, 0]),
                "tangent_y": float(centerline.tangent[i, 1]),
                "tangent_z": float(centerline.tangent[i, 2]),
            }
        )
    return pd.DataFrame(rows)

def save_root_overview(
    case_id: str,
    aligned_label: np.ndarray,
    label_map: Dict[str, int],
    roots: Dict[str, core.RootAssignment],
    output_path: Path,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rgb = legacy.label_projection_rgb(aligned_label, label_map, axis = 2, background = (0, 0, 0))
    fig, ax = plt.subplots(1, 1, figsize = (8.5, 8.0), facecolor = "white")
    ax.imshow(rgb)
    for branch, root in roots.items():
        x = float(root.root_voxel_x)
        y = float(((aligned_label.shape[1] - 1) - root.root_voxel_y))
        color = (np.asarray(legacy.branch_display_color(branch), dtype = np.float32) / 255.0)
        ax.scatter([x], [y], s = 70, marker = "o", edgecolors = "white", linewidths = 1.5, color = [color])
        ax.text(
            (x + 4),
            (y + 4),
            f"{branch}\n{root.source}",
            color = "white",
            fontsize = 8,
            bbox = dict(boxstyle = "round,pad=0.18", fc = "black", ec = color, alpha = 0.82),
        )
    ax.set_title(f"{case_id} root assignments", fontsize = 16, fontweight = "bold")
    ax.axis("off")
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(output_path, dpi = 180, bbox_inches = "tight", facecolor = "white")
    plt.close(fig)
    return output_path

def save_overall_stenosis_mips(
    case_id: str,
    image: np.ndarray,
    aligned_label: np.ndarray,
    label_map: Dict[str, int],
    centerlines: Dict[str, core.CenterlineResult],
    all_lesions: Sequence[core.LesionResult],
    spacing: Sequence[float],
    output_dir: Path,
) -> List[Path]:
    accepted_by_branch: Dict[str, List[core.LesionResult]] = {}
    for lesion in all_lesions:
        if lesion.accepted:
            accepted_by_branch.setdefault(lesion.branch, []).append(lesion)
    branch_profiles: Dict[str, Dict[str, Any]] = {}
    paths: Dict[str, np.ndarray] = {}
    for branch, centerline in centerlines.items():
        paths[branch] = np.rint(centerline.path_voxel).astype(np.int32)
        branch_profiles[branch] = {
            "accepted_candidates": [
                {
                    "peak_idx": int(lesion.peak_index),
                    "interval_label": str(lesion.stenosis_interval),
                    "report_ratio": float(lesion.stenosis_ratio),
                }
                for lesion in accepted_by_branch.get(branch, [])
            ]
        }
    union = np.isin(aligned_label, np.asarray(list(label_map.values()), dtype = aligned_label.dtype))
    try:
        return legacy.save_stenosis_debug_all_mips(
            case_id,
            image,
            union,
            paths,
            branch_profiles,
            output_dir,
            aligned_label = aligned_label,
            label_map = label_map,
            spacing = tuple(float(v) for v in spacing),
        )
    finally:
        del union
