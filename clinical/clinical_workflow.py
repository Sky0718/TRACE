from __future__ import annotations

import gc
import tempfile
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import cta_legacy_backend_v1_0_15 as legacy
import stenosis_v2_core as core
from mrb_input import load_mrb_segmentation
from agatston_v2 import AGATSTON_ALGORITHM_VERSION, AgatstonResult, compute_agatston
from clinical_inputs import (
    LabelMaskMapping,
    PIPELINE_NAME,
    PIPELINE_VERSION,
    ensure_dependencies,
    infer_branch_labels_without_colortable,
    load_dicom_volume_preallocated,
    load_label_uint16,
    mask_geometry,
    parse_branch_selection,
    release_large_arrays,
    sanitize_case_id,
)
from clinical_outputs import (
    add_patient_coordinates,
    append_replace_case,
    atomic_csv,
    build_stenosis_review_candidates,
    centerline_dataframe,
    partition_candidate_outputs,
    save_overall_stenosis_mips,
    save_root_overview,
    synchronize_branch_results_with_formal_lesions,
    write_json,
)
from cta_legacy_backend_v1_0_15 import ensure_dir

def analyze_case(
    case: Dict[str, Any],
    out_root: Path,
    selected_branches: Optional[Sequence[str]],
    save_qa: bool,
    minimum_reportable_ratio: float,
    centerline_step_mm: float,
    cross_section_pixel_mm: float,
) -> Dict[str, Any]:
    ensure_dependencies()
    case_id = str(case["case_id"])
    case_out = ensure_dir((Path(out_root) / case_id))
    visual_dir = ensure_dir((case_out / "visuals"))
    profile_dir = ensure_dir((case_out / "profiles"))
    centerline_dir = ensure_dir((case_out / "centerlines"))
    start_time = time.time()

    color_table_path = case.get("color_table")
    label_mrb_path = (case.get("label_mrb") if (case.get("label_nifti") is None) else None)
    input_provenance: Dict[str, Any] = {}
    color_entries = (
        legacy.parse_segmentation_color_table(Path(color_table_path))
        if (color_table_path is not None)
        else {}
    )
    if (label_mrb_path is not None):
        label_raw, label_header_spacing, color_entries, input_provenance = load_mrb_segmentation(
            Path(label_mrb_path)
        )
    vessel_entries = {
        int(value): entry
        for value, entry in color_entries.items()
        if ((int(value) > 0) and legacy.is_vessel_color_table_label(str(entry["name"])))
    }
    if (((color_table_path is not None) or (label_mrb_path is not None)) and not vessel_entries):
        raise ValueError("No vessel labels were found in ColorTable.")

    if (label_mrb_path is None):
        label_raw, label_header_spacing = load_label_uint16(Path(case["label_nifti"]))
    alignment_values = list(vessel_entries.keys())
    if ((color_table_path is None) and (label_mrb_path is None)):
        alignment_values = [int(v) for v in np.unique(label_raw) if (v > 0)]
        if not alignment_values:
            raise ValueError("No positive labels were found in the segmentation.")

    with tempfile.TemporaryDirectory(prefix = f"{case_id}_dicom_") as tmp:
        legacy.safe_extract_zip(Path(case["dicom_zip"]), Path(tmp))
        image, spacing, dicom_meta = load_dicom_volume_preallocated(Path(tmp))

    aligned_label, alignment_method, alignment_score = legacy.align_multilabel_to_dicom(
        label_raw, image, alignment_values
    )
    if not np.issubdtype(aligned_label.dtype, np.integer):
        aligned_label = np.rint(aligned_label).astype(np.uint16)
    del label_raw
    gc.collect()

    branch_specs: List[Dict[str, Any]] = []
    if ((color_table_path is None) and (label_mrb_path is None)):
        aligned_label, branch_specs, input_provenance = infer_branch_labels_without_colortable(
            aligned_label
        )
    duplicate_counts: Dict[str, int] = {}
    for value, entry in vessel_entries.items():
        if not np.any((aligned_label == int(value))):
            continue
        base = legacy.normalize_packaged_branch_name(str(entry["name"]))
        duplicate_counts[base] = (duplicate_counts.get(base, 0) + 1)
        branch = (base if (duplicate_counts[base] == 1) else f"{base}_{duplicate_counts[base]}")
        branch_specs.append(
            {
                "branch": str(branch),
                "label_value": int(value),
                "original_name": str(entry.get("original_name", entry["name"])),
                "rgba": tuple(int(v) for v in entry["rgba"]),
            }
        )
    branch_specs.sort(key = lambda x: legacy.branch_sort_key(x["branch"]))
    label_map = {str(spec["branch"]): int(spec["label_value"]) for spec in branch_specs}
    original_name_map = {str(spec["branch"]): str(spec["original_name"]) for spec in branch_specs}
    rgba_map = {str(spec["branch"]): tuple(spec["rgba"]) for spec in branch_specs}
    branch_names = parse_branch_selection(
        (",".join(selected_branches) if selected_branches else None),
        list(label_map),
    )
    if not branch_names:
        raise ValueError("No requested branches exist in this case.")

    lazy_masks = LabelMaskMapping(aligned_label, label_map)
    point_values = [
        int(value)
        for value, entry in color_entries.items()
        if ("POINT" in str(entry["name"]).upper())
    ]
    roots = core.assign_branch_roots_from_labels(aligned_label, label_map, point_values, spacing)
    release_large_arrays()

    agatston: AgatstonResult = compute_agatston(
        image,
        aligned_label,
        branch_specs,
        spacing,
        dicom_meta = dicom_meta,
        calibration = None,
    )
    release_large_arrays()

    branch_rows: List[Dict[str, Any]] = []
    all_candidates: List[core.LesionResult] = []
    all_accepted: List[core.LesionResult] = []
    root_rows: List[Dict[str, Any]] = []
    centerlines: Dict[str, core.CenterlineResult] = {}

    for branch in branch_names:
        branch_mask = lazy_masks[branch]
        root = roots.get(branch)
        branch_base = core.base_branch_name(branch)
        short_branch = (branch_base in ("LM", "D1", "D2", "RAMUS", "RAD"))
        effective_centerline_step_mm = (
            min(float(centerline_step_mm), 0.35) if short_branch else float(centerline_step_mm)
        )
        effective_cross_section_pixel_mm = (
            min(float(cross_section_pixel_mm), 0.12)
            if (branch_base in ("D1", "D2", "RAMUS", "RAD"))
            else (
                min(float(cross_section_pixel_mm), 0.14)
                if (branch_base == "LM")
                else float(cross_section_pixel_mm)
            )
        )
        initial_centerline = core.extract_rooted_centerline(
            branch,
            branch_mask,
            spacing,
            root_assignment = root,
            step_mm = effective_centerline_step_mm,
        )

        centerline, profile, patches, lumen_qc = core.analyze_lumen_cross_sections(
            image,
            branch_mask,
            initial_centerline,
            spacing,
            calc_threshold_hu = float(agatston.calcium_threshold_hu),
            pixel_mm = effective_cross_section_pixel_mm,
            keep_patches = bool(save_qa),
        )
        centerlines[branch] = centerline
        if (root is not None):
            root_rows.append(
                {
                    **asdict(root),
                    **{f"topology_{k}": v for k, v in centerline.topology.items()},
                    **{f"lumen_qc_{k}": v for k, v in lumen_qc.items()},
                }
            )
        else:
            root_rows.append(
                {
                    "branch": branch,
                    "source": "fallback_largest_radius_endpoint",
                    "root_voxel_x": (
                        float(centerline.path_voxel[0, 0]) if centerline.ok else np.nan
                    ),
                    "root_voxel_y": (
                        float(centerline.path_voxel[0, 1]) if centerline.ok else np.nan
                    ),
                    "root_voxel_z": (
                        float(centerline.path_voxel[0, 2]) if centerline.ok else np.nan
                    ),
                    "nearest_distance_mm": np.nan,
                    "point_component_id": -1,
                    "parent_branch": "",
                    "seam_exclusion_mm": 0.0,
                    **{f"topology_{k}": v for k, v in centerline.topology.items()},
                    **{f"lumen_qc_{k}": v for k, v in lumen_qc.items()},
                }
            )
        atomic_csv(
            centerline_dataframe(centerline),
            (centerline_dir / f"{sanitize_case_id(branch)}_root_to_tip_centerline.csv"),
        )

        lesions, profile = core.detect_and_measure_lesions(
            case_id,
            branch,
            profile,
            centerline,
            lazy_masks,
            spacing,
            minimum_reportable_ratio = float(minimum_reportable_ratio),
            branch_roots = roots,
        )
        formal_statuses = {"formal", "formal_limited_branch_coverage"}
        accepted = [
            lesion
            for lesion in lesions
            if (
                (int(lesion.accepted) == 1)
                and (str(lesion.clinical_output_status) in formal_statuses)
            )
        ]
        all_candidates.extend(lesions)
        all_accepted.extend(accepted)
        atomic_csv(profile, (profile_dir / f"{sanitize_case_id(branch)}_cta_lumen_profile.csv"))

        summary = asdict(
            core.summarize_branch(
                case_id, branch, int(label_map[branch]), centerline, profile, lesions
            )
        )
        summary.update(mask_geometry(branch_mask, centerline, spacing))
        summary.update({f"lumen_qc_{k}": v for k, v in lumen_qc.items()})
        summary["effective_centerline_step_mm"] = float(effective_centerline_step_mm)
        summary["effective_cross_section_pixel_mm"] = float(effective_cross_section_pixel_mm)
        summary["short_branch_high_resolution_sampling"] = int(short_branch)
        rgba = rgba_map[branch]
        branch_ag = agatston.branch_results.get(branch, {})
        summary.update(
            {
                "case_group": int(case["case_group"]),
                "case_number": int(case["case_number"]),
                "pipeline_version": PIPELINE_VERSION,
                "stenosis_algorithm_version": core.ALGORITHM_VERSION,
                "agatston_algorithm_version": AGATSTON_ALGORITHM_VERSION,
                "branch_original_name": original_name_map[branch],
                "label_color_r": int(rgba[0]),
                "label_color_g": int(rgba[1]),
                "label_color_b": int(rgba[2]),
                "label_color_a": int(rgba[3]),
                "spacing_x_mm": float(spacing[0]),
                "spacing_y_mm": float(spacing[1]),
                "spacing_z_mm": float(spacing[2]),
                "label_header_spacing_x_mm": float(label_header_spacing[0]),
                "label_header_spacing_y_mm": float(label_header_spacing[1]),
                "label_header_spacing_z_mm": float(label_header_spacing[2]),
                "segmentation_alignment_method": str(alignment_method),
                "segmentation_alignment_hu_score": float(alignment_score),
                "point_label_available": int(
                    any(("POINT" in str(e["name"]).upper()) for e in color_entries.values())
                ),
                "branch_agatston_score": float(branch_ag.get("score", 0.0)),
                "branch_agatston_raw_score": float(
                    branch_ag.get("raw_volume_conversion_score", branch_ag.get("score", 0.0))
                ),
                "branch_agatston_direct_score": float(branch_ag.get("direct_density_score", 0.0)),
                "branch_calcium_volume_mm3": float(branch_ag.get("calcium_volume_mm3", 0.0)),
                "branch_agatston_lesion_count": int(branch_ag.get("lesion_count", 0)),
                "branch_agatston_area_mm2": float(branch_ag.get("accepted_area_mm2", 0.0)),
                "branch_agatston_peak_hu": float(branch_ag.get("maximum_hu", 0.0)),
                "branch_calcium_shell_threshold_hu": float(
                    branch_ag.get("shell_threshold_hu", np.nan)
                ),
                "branch_calcium_core_threshold_hu": float(
                    branch_ag.get("core_threshold_hu", np.nan)
                ),
                "agatston_score": float(agatston.agatston_score),
                "agatston_score_raw": float(agatston.agatston_score_raw),
                "agatston_score_direct": float(agatston.agatston_score_direct),
                "agatston_score_legacy": float(agatston.agatston_score_legacy),
                "agatston_grade": str(agatston.agatston_grade),
                "risk_category": str(agatston.risk_category),
                "calcium_threshold_hu": float(agatston.calcium_threshold_hu),
                "agatston_calibration_applied": 0,
                "agatston_frozen": 1,
                "agatston_frozen_from_pipeline_version": "2.0.5",
                "source_dicom_zip": str(case["dicom_zip"]),
                "source_color_table": (
                    str(color_table_path) if (color_table_path is not None) else ""
                ),
                "source_label_nifti": (
                    str(case["label_nifti"]) if (case.get("label_nifti") is not None) else ""
                ),
                **({"source_mrb": str(label_mrb_path)} if (label_mrb_path is not None) else {}),
            }
        )
        branch_rows.append(summary)

        if (save_qa and centerline.ok):
            cpr_path = (visual_dir / f"stenosis_v2_CPR_{sanitize_case_id(branch)}.png")
            core.save_cpr_figure(
                case_id, branch, image, centerline, profile, lesions, spacing, cpr_path
            )
            cpr_u, cpr_offsets = core.sample_cpr(image, centerline, spacing, "u")
            cpr_v, _ = core.sample_cpr(image, centerline, spacing, "v")
            np.savez_compressed(
                (profile_dir / f"{sanitize_case_id(branch)}_CPR_arrays.npz"),
                cpr_u = cpr_u.astype(np.float32),
                cpr_v = cpr_v.astype(np.float32),
                offsets_mm = cpr_offsets.astype(np.float32),
                distance_mm = centerline.cumulative_mm.astype(np.float32),
                path_voxel = centerline.path_voxel.astype(np.float32),
                path_physical_mm = centerline.path_physical_mm.astype(np.float32),
                spacing_mm = np.asarray(spacing, dtype = np.float32),
            )
            del cpr_u, cpr_v, cpr_offsets
            core.save_section_stack_npz(
                (profile_dir / f"{sanitize_case_id(branch)}_section_stack.npz"),
                patches,
                centerline,
            )
            for lesion in accepted:
                core.save_lesion_cross_section_figure(
                    case_id,
                    branch,
                    lesion,
                    patches,
                    (
                        visual_dir / f"stenosis_v2_{sanitize_case_id(branch)}_lesion_{int(lesion.lesion_index):02d}_sections.png"
                    ),
                )
        del branch_mask, profile, patches
        release_large_arrays()

    branch_df = pd.DataFrame(branch_rows)
    lesion_columns = [field.name for field in fields(core.LesionResult)]
    candidate_df = (
        core.dataframe_from_dataclasses(all_candidates)
        if all_candidates
        else pd.DataFrame(columns = lesion_columns)
    )
    candidate_df = add_patient_coordinates(candidate_df, dicom_meta.get("voxel_to_patient_affine"))
    candidate_df, accepted_df, rejected_df, not_assessable_df, partition_report = (
        partition_candidate_outputs(candidate_df)
    )
    review_df = build_stenosis_review_candidates(candidate_df)
    partition_report["review_candidate_count"] = int(len(review_df))
    partition_report["borderline_nonformal_review_count"] = (
        int(
            np.sum(
                pd.to_numeric(review_df.get("accepted", pd.Series(dtype = float)), errors = "coerce")
                .fillna(0)
                .astype(int)
                .eq(0)
            )
        )
        if not review_df.empty
        else 0
    )
    partition_report["formal_uncertainty_review_count"] = (
        int(
            np.sum(
                pd.to_numeric(review_df.get("accepted", pd.Series(dtype = float)), errors = "coerce")
                .fillna(0)
                .astype(int)
                .eq(1)
            )
        )
        if not review_df.empty
        else 0
    )
    branch_df = synchronize_branch_results_with_formal_lesions(branch_df, accepted_df)
    formal_keys = {
        (str(r.get("branch", "")), round(float(r.get("peak_distance_mm", -1.0)), 4))
        for _, r in accepted_df.iterrows()
    }
    all_accepted = [
        lesion
        for lesion in all_candidates
        if (
            ((str(lesion.branch), round(float(lesion.peak_distance_mm), 4)) in formal_keys)
            and (int(lesion.accepted) == 1)
        )
    ]
    root_columns = [field.name for field in fields(core.RootAssignment)]
    root_df = (pd.DataFrame(root_rows) if root_rows else pd.DataFrame(columns = root_columns))
    agatston_dict = agatston.to_dict()
    agatston_row = {k: v for k, v in agatston_dict.items() if (k != "branch_results")}
    agatston_row.update(
        {
            "case_id": case_id,
            "case_group": int(case["case_group"]),
            "case_number": int(case["case_number"]),
            "pipeline_version": PIPELINE_VERSION,
            "agatston_frozen": 1,
            "agatston_frozen_from_pipeline_version": "2.0.5",
            "calibration_applied": 0,
            "calibration_description": "frozen_raw_v2_0_5_algorithm_no_calibration_in_v2_0_10",
        }
    )
    for branch, values in agatston.branch_results.items():
        agatston_row[f"{branch}_agatston_score"] = float(values.get("score", 0.0))
        agatston_row[f"{branch}_agatston_lesion_count"] = int(values.get("lesion_count", 0))
    agatston_df = pd.DataFrame([agatston_row])

    pre_verifier_df = (
        candidate_df[
            (
                pd.to_numeric(candidate_df.get("pre_verifier_accepted", 0), errors = "coerce")
                .fillna(0)
                .astype(int)
                == 1
            )
        ].copy()
        if (not candidate_df.empty and ("pre_verifier_accepted" in candidate_df.columns))
        else pd.DataFrame(columns = candidate_df.columns)
    )
    write_json((case_out / "stenosis_output_consistency.json"), partition_report)
    atomic_csv(branch_df, (case_out / "branch_mask_dataset.csv"))
    atomic_csv(candidate_df, (case_out / "stenosis_candidates_all.csv"))
    atomic_csv(pre_verifier_df, (case_out / "stenosis_lesions_pre_verifier.csv"))
    atomic_csv(accepted_df, (case_out / "stenosis_lesions.csv"))
    atomic_csv(review_df, (case_out / "stenosis_review_candidates.csv"))
    atomic_csv(not_assessable_df, (case_out / "stenosis_not_assessable.csv"))
    atomic_csv(rejected_df, (case_out / "stenosis_rejected_candidates.csv"))
    atomic_csv(root_df, (case_out / "root_assignments.csv"))
    atomic_csv(agatston_df, (case_out / "agatston_summary.csv"))

    if save_qa:
        legacy.save_branch_mask_mip_figure(
            case_id,
            aligned_label,
            label_map,
            visual_dir,
            spacing = tuple(float(v) for v in spacing),
        )
        save_root_overview(
            case_id,
            aligned_label,
            label_map,
            roots,
            (visual_dir / "stenosis_v2_root_assignments.png"),
        )
        save_overall_stenosis_mips(
            case_id,
            image,
            aligned_label,
            label_map,
            centerlines,
            all_accepted,
            spacing,
            visual_dir,
        )

    elapsed = float((time.time() - start_time))
    case_summary = {
        "case_id": case_id,
        "case_group": int(case["case_group"]),
        "case_number": int(case["case_number"]),
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "stenosis_algorithm_version": core.ALGORITHM_VERSION,
        "agatston_algorithm_version": AGATSTON_ALGORITHM_VERSION,
        "branch_count_available": int(len(label_map)),
        "branch_count_processed": int(len(branch_rows)),
        "processed_branches": branch_names,
        "accepted_lesion_count": int(len(accepted_df)),
        "candidate_count": int(len(candidate_df)),
        "rejected_assessable_candidate_count": int(len(rejected_df)),
        "not_assessable_candidate_count": int(len(not_assessable_df)),
        "review_candidate_count": int(len(review_df)),
        "borderline_nonformal_review_count": int(
            partition_report.get("borderline_nonformal_review_count", 0)
        ),
        "formal_uncertainty_review_count": int(
            partition_report.get("formal_uncertainty_review_count", 0)
        ),
        "formal_whole_branch_lesion_count": int(
            partition_report.get("formal_whole_branch_count", 0)
        ),
        "formal_local_only_lesion_count": int(partition_report.get("formal_local_only_count", 0)),
        "candidate_partition_complete": int(partition_report.get("partition_complete", 0)),
        "candidate_partition_mutually_exclusive": int(
            partition_report.get("partition_mutually_exclusive", 0)
        ),
        "assessable_branch_count": (
            int(
                np.sum(
                    (
                        branch_df.get("branch_assessability", pd.Series(dtype = str)).astype(str)
                        == "assessable"
                    )
                )
            )
            if not branch_df.empty
            else 0
        ),
        "limited_or_not_assessable_branch_count": (
            int(
                np.sum(
                    (
                        branch_df.get("branch_assessability", pd.Series(dtype = str)).astype(str)
                        != "assessable"
                    )
                )
            )
            if not branch_df.empty
            else 0
        ),
        "agatston_score": float(agatston.agatston_score),
        "agatston_score_raw": float(agatston.agatston_score_raw),
        "agatston_score_direct": float(agatston.agatston_score_direct),
        "agatston_score_legacy": float(agatston.agatston_score_legacy),
        "agatston_grade": str(agatston.agatston_grade),
        "risk_category": str(agatston.risk_category),
        "calcium_threshold_hu": float(agatston.calcium_threshold_hu),
        "calcium_volume_mm3": float(agatston.calcium_volume_mm3),
        "agatston_calibration_applied": 0,
        "agatston_calibration_description": "frozen_raw_v2_0_5_algorithm_no_calibration_in_v2_0_10",
        "agatston_frozen": 1,
        "agatston_frozen_from_pipeline_version": "2.0.5",
        "spacing_mm": [float(v) for v in spacing],
        "dicom_meta": dicom_meta,
        "segmentation_alignment_method": alignment_method,
        "segmentation_alignment_hu_score": float(alignment_score),
        "elapsed_seconds": elapsed,
        "important_interpretation": (
            "Branch labels are used for navigation/search. The centerline is recentered with CTA support, "
            "lumen contours are tracked longitudinally on orthogonal CTA planes, invalid sections are excluded, "
            "and severity is measured directly from valid diameter/area measurements. Evidence scores do not scale the stenosis percentage."
        ),
        "clinical_status": "research_output_requires_clinical_validation",
    }
    case_summary.update(input_provenance)
    write_json((case_out / "case_summary.json"), case_summary)
    write_json(
        (case_out / "label_mapping.json"),
        {
            branch: {
                "label_value": int(label_map[branch]),
                "original_name": original_name_map[branch],
                "rgba": list(rgba_map[branch]),
            }
            for branch in label_map
        },
    )
    write_json(
        (case_out / "run_manifest.json"),
        {
            **case_summary,
            "input_files": {
                "dicom_zip": str(case["dicom_zip"]),
                "color_table": (str(color_table_path) if (color_table_path is not None) else None),
                "label_nifti": (
                    str(case["label_nifti"]) if (case.get("label_nifti") is not None) else None
                ),
                **({"label_mrb": str(label_mrb_path)} if (label_mrb_path is not None) else {}),
            },
            "outputs": {
                "branch_dataset": str((case_out / "branch_mask_dataset.csv")),
                "accepted_lesions": str((case_out / "stenosis_lesions.csv")),
                "review_candidates": str((case_out / "stenosis_review_candidates.csv")),
                "pre_verifier_lesions": str((case_out / "stenosis_lesions_pre_verifier.csv")),
                "all_candidates": str((case_out / "stenosis_candidates_all.csv")),
                "not_assessable_candidates": str((case_out / "stenosis_not_assessable.csv")),
                "rejected_candidates": str((case_out / "stenosis_rejected_candidates.csv")),
                "output_consistency": str((case_out / "stenosis_output_consistency.json")),
                "root_assignments": str((case_out / "root_assignments.csv")),
                "agatston_summary": str((case_out / "agatston_summary.csv")),
                "profiles": str(profile_dir),
                "centerlines": str(centerline_dir),
                "visuals": str(visual_dir),
            },
        },
    )

    append_replace_case(
        (Path(out_root) / "branch_mask_dataset.csv"),
        branch_df,
        case_id,
        ["case_group", "case_number", "branch"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_candidates_all.csv"),
        candidate_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_lesions_pre_verifier.csv"),
        pre_verifier_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_lesions.csv"),
        accepted_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_review_candidates.csv"),
        review_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_not_assessable.csv"),
        not_assessable_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "stenosis_rejected_candidates.csv"),
        rejected_df,
        case_id,
        ["case_id", "branch", "peak_distance_mm"],
    )
    append_replace_case(
        (Path(out_root) / "agatston_scores.csv"),
        agatston_df,
        case_id,
        ["case_group", "case_number"],
    )

    del image, aligned_label
    gc.collect()
    return case_summary
