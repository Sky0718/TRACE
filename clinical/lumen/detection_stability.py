from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter

from . import detection_base as detection_base_ops
from . import profiles as profiles_ops
from . import sections as sections_ops

from .models import (
    BranchResult,
    CenterlineResult,
    DEFAULT_CROSS_SECTION_PIXEL_MM,
    DEFAULT_MIN_REPORTABLE_RATIO,
    LesionResult,
    RootAssignment,
    base_branch_name,
    severity_interval,
)

def detect_stable_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    results, df = detection_base_ops.detect_base_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    if (df.empty or not results):
        return results, df
    step = (
        float(np.median(np.diff(centerline.cumulative_mm)))
        if (centerline.cumulative_mm.size >= 2)
        else 0.55
    )
    valid = (pd.to_numeric(df.get("valid", 0), errors = "coerce").fillna(0).to_numpy(int) > 0)
    ds = pd.to_numeric(
        df.get("smoothed_min_diameter_mm", df.get("min_diameter_mm")), errors = "coerce"
    ).to_numpy(float)
    dref = pd.to_numeric(df.get("reference_diameter_mm", np.nan), errors = "coerce").to_numpy(float)
    identity_arr = (
        pd.to_numeric(df.get("lumen_identity_score", 0.0), errors = "coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    edge_arr = (
        pd.to_numeric(df.get("gradient_boundary_support", 0.0), errors = "coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    contour_arr = (
        pd.to_numeric(df.get("contour_component_agreement", 0.0), errors = "coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    tracking_overlap_arr = pd.to_numeric(
        df.get("tracking_overlap", np.nan), errors = "coerce"
    ).to_numpy(float)
    center_inside_arr = (
        pd.to_numeric(df.get("center_inside_component", 0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    radial_cv_arr = pd.to_numeric(df.get("radial_cv", np.nan), errors = "coerce").to_numpy(float)
    tracking_score_arr = (
        pd.to_numeric(df.get("tracking_score", 0.0), errors = "coerce").fillna(0.0).to_numpy(float)
    )
    center_shift_arr = pd.to_numeric(df.get("center_shift_mm", np.nan), errors = "coerce").to_numpy(
        float
    )
    diameter_ratio_arr = (
        pd.to_numeric(df.get("diameter_stenosis_ratio", 0.0), errors = "coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    noise_ratio = sections_ops._robust_profile_noise_ratio(ds, dref, valid, step)
    df["profile_noise_ratio"] = float(noise_ratio)
    navigation_d = pd.to_numeric(
        df.get("navigation_mask_equivalent_diameter_mm", np.nan), errors = "coerce"
    ).to_numpy(float)
    navigation_valid = (np.isfinite(navigation_d) & (navigation_d > 0))
    if (int(np.sum(navigation_valid)) >= 5):
        nav_fill = navigation_d.copy()
        x_all = np.arange(nav_fill.size)
        nav_fill[~navigation_valid] = np.interp(
            x_all[~navigation_valid], x_all[navigation_valid], nav_fill[navigation_valid]
        )
        nav_window = max(5, int(round((10.0 / max(step, 0.2)))))
        if ((nav_window % 2) == 0):
            nav_window += 1
        nav_ref = percentile_filter(nav_fill, percentile = 80.0, size = nav_window, mode = "nearest")
        nav_ref = gaussian_filter1d(
            np.maximum(nav_ref, nav_fill), sigma = max(0.7, (0.8 / max(step, 0.2))), mode = "nearest"
        )
        navigation_ratio_profile = np.clip((1.0 - (nav_fill / np.maximum(nav_ref, 1e-6))), 0.0, 1.0)
    else:
        navigation_ratio_profile = np.full(len(df), np.nan, dtype = float)
    df["navigation_mask_stenosis_ratio"] = navigation_ratio_profile
    branch_base = base_branch_name(branch)
    main_branch = (branch_base in ("RCA", "LAD", "LCX"))
    small_branch = (branch_base in ("D1", "D2", "RAMUS", "RAD"))
    topology = dict((centerline.topology or {}))
    topology_length = float(
        topology.get(
            "length_mm", (centerline.cumulative_mm[-1] if centerline.cumulative_mm.size else 0.0)
        )
    )
    topology_tortuosity = float(topology.get("tortuosity", 0.0))
    topology_endpoints = int(round(float(topology.get("skeleton_endpoint_count", 0.0))))
    topology_path_fraction = float(topology.get("path_skeleton_fraction", 1.0))
    topology_risky = bool(
        (
            (
                (branch_base == "RCA")
                and (
                    (topology_length > 210.0)
                    or (topology_tortuosity > 2.55)
                    or (topology_endpoints > 16)
                )
            )
            or (topology_tortuosity > 3.25)
            or ((topology_endpoints > 14) and (topology_path_fraction < 0.18))
        )
    )
    topology_risk_label = (
        f"length={topology_length:.1f};tortuosity={topology_tortuosity:.2f};"
        f"endpoints={topology_endpoints};path_fraction={topology_path_fraction:.3f}"
        if topology_risky
        else ""
    )

    for lesion in results:
        lesion.pre_verifier_accepted = int(lesion.accepted)
        start = max(0, int(lesion.start_index))
        end = min((len(df) - 1), int(lesion.end_index))
        idx = (np.arange(start, (end + 1), dtype = int) if (end >= start) else np.array([], dtype = int))
        good_idx = (idx[valid[idx]] if idx.size else np.array([], dtype = int))
        if good_idx.size:
            identity_score = float(np.nanmedian(identity_arr[good_idx]))
            edge_support = float(np.nanmedian(edge_arr[good_idx]))
            contour_agreement = float(np.nanmedian(contour_arr[good_idx]))
        else:
            identity_score = edge_support = contour_agreement = 0.0
        ratio = (float(lesion.stenosis_ratio) if np.isfinite(lesion.stenosis_ratio) else 0.0)
        critical_threshold = max(0.25, (0.65 * max(ratio, 0.0)))
        critical_idx = (
            good_idx[(diameter_ratio_arr[good_idx] >= critical_threshold)]
            if good_idx.size
            else np.array([], dtype = int)
        )
        if (critical_idx.size < 2):
            critical_idx = good_idx

        def _finite_median(values: np.ndarray, indices: np.ndarray, default: float = 0.0) -> float:
            if (indices.size == 0):
                return float(default)
            arr = np.asarray(values, dtype = float)[indices]
            arr = arr[np.isfinite(arr)]
            return (float(np.median(arr)) if arr.size else float(default))

        def _finite_fraction(values: np.ndarray, indices: np.ndarray, threshold: float) -> float:
            if (indices.size == 0):
                return 0.0
            arr = np.asarray(values, dtype = float)[indices]
            arr = arr[np.isfinite(arr)]
            return (float(np.mean((arr >= float(threshold)))) if arr.size else 0.0)

        tracking_overlap_median = _finite_median(tracking_overlap_arr, critical_idx, 0.0)
        tracking_supported_fraction = _finite_fraction(tracking_overlap_arr, critical_idx, 0.45)
        center_inside_fraction = _finite_fraction(center_inside_arr, critical_idx, 0.5)
        radial_cv_median = _finite_median(radial_cv_arr, critical_idx, 1.0)
        tracking_score_median = _finite_median(tracking_score_arr, critical_idx, 0.0)
        center_shift_median = _finite_median(center_shift_arr, critical_idx, 99.0)
        expected_area = float(np.clip((1.0 - ((1.0 - ratio) ** 2)), 0.0, 1.0))
        area_ratio = (
            float(lesion.area_stenosis_ratio)
            if np.isfinite(lesion.area_stenosis_ratio)
            else float("nan")
        )
        area_agreement = (
            float(np.clip((1.0 - abs((area_ratio - expected_area))), 0.0, 1.0))
            if np.isfinite(area_ratio)
            else 0.0
        )
        if (good_idx.size and np.any(np.isfinite(navigation_ratio_profile[good_idx]))):
            navigation_ratio = float(np.nanmedian(navigation_ratio_profile[good_idx]))
        else:
            navigation_ratio = float("nan")
        cta_mask_concordance = (
            float(np.clip((1.0 - abs((navigation_ratio - ratio))), 0.0, 1.0))
            if np.isfinite(navigation_ratio)
            else float("nan")
        )
        signal_z = float((lesion.local_prominence / max(noise_ratio, 0.025)))
        lesion.lumen_identity_score = float(identity_score)
        lesion.gradient_boundary_support = float(edge_support)
        lesion.profile_noise_ratio = float(noise_ratio)
        lesion.profile_signal_z = float(signal_z)
        lesion.diameter_area_agreement = float(area_agreement)
        lesion.navigation_mask_stenosis_ratio = float(navigation_ratio)
        lesion.cta_mask_concordance = float(cta_mask_concordance)
        lesion.tracking_overlap_median = float(tracking_overlap_median)
        lesion.tracking_supported_fraction = float(tracking_supported_fraction)
        lesion.center_inside_fraction = float(center_inside_fraction)
        lesion.radial_cv_median = float(radial_cv_median)
        lesion.tracking_score_median = float(tracking_score_median)
        lesion.center_shift_median_mm = float(center_shift_median)
        lesion.topology_risk_flag = str(topology_risk_label)
        lesion.verifier_version = "deterministic_identity_noise_topology_v2_0_5"

        if ((int(lesion.accepted) != 1) or (str(lesion.assessment_status) != "assessable")):
            continue
        reasons: List[str] = []
        warnings = [x for x in str(lesion.warning_flags).split(";") if x]
        min_identity = (0.48 if (ratio >= 0.70) else (0.52 if (ratio >= 0.50) else 0.57))
        min_edge = (0.20 if (ratio >= 0.70) else (0.24 if (ratio >= 0.50) else 0.29))
        if (identity_score < min_identity):
            reasons.append("low_target_lumen_identity")
        if (edge_support < min_edge):
            reasons.append("weak_true_CTA_boundary_gradient")
        if (contour_agreement < (0.30 if (ratio >= 0.70) else 0.40)):
            reasons.append("weak_component_contour_agreement")

        min_z = (1.35 if (ratio >= 0.70) else (1.75 if (ratio >= 0.50) else 2.25))
        if (signal_z < min_z):
            reasons.append("profile_change_not_above_tracking_noise")
        minimum_length = (1.4 if small_branch else (1.8 if main_branch else 1.5))
        if (float(lesion.lesion_length_mm) < minimum_length):
            reasons.append("insufficient_physical_lesion_length")
        if (float(lesion.peak_valid_run_mm) < max(1.1, (0.35 * float(lesion.lesion_length_mm)))):
            reasons.append("insufficient_sustained_valid_lumen_tracking")
        min_ref_fraction = (0.34 if (ratio >= 0.70) else (0.45 if (ratio >= 0.50) else 0.55))
        if (float(lesion.reference_valid_fraction) < min_ref_fraction):
            reasons.append("weak_reference_section_coverage")
        min_area_agreement = (0.28 if (ratio >= 0.70) else (0.40 if (ratio >= 0.50) else 0.55))
        if (area_agreement < min_area_agreement):
            reasons.append("diameter_area_severity_disagreement")

        minimum_critical_sections = (2 if (ratio >= 0.65) else (3 if main_branch else 2))
        if (critical_idx.size < minimum_critical_sections):
            reasons.append("too_few_identity_supported_narrowed_sections")
        if (tracking_overlap_median < (0.42 if (ratio >= 0.70) else 0.48)):
            reasons.append("weak_longitudinal_component_overlap")
        if (tracking_supported_fraction < (0.55 if (ratio >= 0.70) else 0.65)):
            reasons.append("insufficient_longitudinal_tracking_support")
        if (tracking_score_median < (0.56 if (ratio >= 0.70) else 0.64)):
            reasons.append("weak_longitudinal_tracking_score")
        if (center_shift_median > (1.10 if (ratio >= 0.70) else 0.90)):
            reasons.append("unstable_lumen_center_tracking")
        radial_limit = (
            0.76 if ((ratio >= 0.65) or (float(lesion.calcification_fraction) >= 0.25)) else 0.62
        )
        if (radial_cv_median > radial_limit):
            reasons.append("unstable_radial_lumen_boundary")

        bilateral = ((float(lesion.proximal_recovery) >= 0.12) and (
            float(lesion.distal_recovery) >= 0.12
        ))
        near_root = (float(lesion.position_norm) <= 0.14)
        ostial_rescue = (near_root and (ratio >= 0.50) and (float(lesion.distal_recovery) >= 0.16))
        severe_rescue = ((ratio >= 0.70) and (
            max(float(lesion.proximal_recovery), float(lesion.distal_recovery)) >= 0.18
        ))
        if not (bilateral or ostial_rescue or severe_rescue):
            reasons.append("insufficient_sustained_flank_recovery")
        if (("one_sided_reference" in warnings) and not (ostial_rescue or severe_rescue)):
            reasons.append("unreliable_one_sided_reference")
        calc_fraction = float(lesion.calcification_fraction)
        if ((calc_fraction >= 0.45) and (identity_score < 0.64)):
            reasons.append("calcification_blooming_without_stable_lumen_identity")
        severe_root_rescue = bool(
            (
                (float(lesion.position_norm) < 0.15)
                and (ratio >= 0.80)
                and (identity_score >= 0.76)
                and (center_inside_fraction >= 0.80)
                and (tracking_supported_fraction >= 0.60)
                and (float(lesion.distal_recovery) >= 0.25)
            )
        )
        if ((calc_fraction >= 0.35) and (ratio >= 0.50)):
            weak_mask_support = bool(
                (
                    np.isfinite(navigation_ratio)
                    and np.isfinite(cta_mask_concordance)
                    and (navigation_ratio < 0.24)
                    and (cta_mask_concordance < 0.50)
                )
            )
            unstable_calcified_identity = bool(
                ((center_inside_fraction < 0.80) or (tracking_supported_fraction < 0.70))
            )
            if ((weak_mask_support or unstable_calcified_identity) and not severe_root_rescue):
                reasons.append("calcified_CTA_severity_not_reliably_supported")
        seam_exclusion = (
            float(centerline.root_assignment.seam_exclusion_mm)
            if centerline.root_assignment
            else 0.0
        )
        if (float(lesion.peak_distance_mm) < max(2.0, (seam_exclusion + 0.8))):
            root_rescue = (
                (ratio >= 0.75)
                and (float(lesion.distal_recovery) >= 0.25)
                and (identity_score >= 0.75)
                and (float(lesion.calcification_fraction) < 0.30)
            )
            if not root_rescue:
                reasons.append("root_or_label_seam_measurement_unreliable")
        broad_limit = (28.0 if main_branch else 18.0)
        if ((float(lesion.lesion_length_mm) > broad_limit) and (ratio < 0.60)):
            reasons.append("broad_change_more_consistent_with_taper_or_tracking")
        distance_to_tip = (
            float((centerline.cumulative_mm[-1] - lesion.end_distance_mm))
            if centerline.cumulative_mm.size
            else 0.0
        )
        terminal_margin = (4.5 if main_branch else 3.5)
        if ((distance_to_tip < terminal_margin) and (ratio < 0.60)):
            reasons.append("lesion_reaches_unreliable_terminal_endpoint")
        if ((float(lesion.position_norm) > (0.88 if main_branch else 0.80)) and (ratio < 0.50)):
            reasons.append("distal_taper_dominant_candidate")
        if (
            main_branch
            and (float(lesion.position_norm) > 0.72)
            and (ratio < 0.50)
            and np.isfinite(navigation_ratio)
            and (navigation_ratio < 0.12)
        ):
            reasons.append("weakly_supported_distal_mild_candidate")
        if (
            topology_risky
            and (float(lesion.position_norm) > 0.55)
            and np.isfinite(navigation_ratio)
            and (navigation_ratio < 0.18)
            and (not np.isfinite(cta_mask_concordance) or (cta_mask_concordance < 0.58))
        ):
            reasons.append("distal_candidate_on_topologically_unreliable_main_path")

        if reasons:
            lesion.accepted = 0
            prior = [x for x in str(lesion.rejection_reasons).split(";") if x]
            lesion.rejection_reasons = ";".join(dict.fromkeys((prior + reasons)))
            lesion.warning_flags = ";".join(
                dict.fromkeys((warnings + ["v2_0_5_verifier_rejected"]))
            )
        else:
            lesion.stenosis_ratio_uncorrected = float(ratio)
            lesion.partial_volume_correction_mm = 0.0
            lesion.severity_method = "direct_valid_cross_sections"
            peak_dist = float(lesion.peak_distance_mm)
            peak_window = np.where(
                (
                    (centerline.cumulative_mm >= (peak_dist - 1.1)) & (centerline.cumulative_mm <= (peak_dist + 1.1))
                )
            )[0]
            trusted_peak = (
                peak_window[
                    (
                        (valid[peak_window] & (identity_arr[peak_window] >= 0.60)) & (edge_arr[peak_window] >= 0.30)
                    )
                ]
                if peak_window.size
                else np.array([], dtype = int)
            )
            if (
                (trusted_peak.size >= 3)
                and (ratio >= 0.42)
                and (identity_score >= 0.70)
                and (edge_support >= 0.45)
            ):
                raw_peak_d = pd.to_numeric(
                    df.get("min_diameter_mm", np.nan), errors = "coerce"
                ).to_numpy(float)
                peak_values = raw_peak_d[trusted_peak]
                peak_values = peak_values[(np.isfinite(peak_values) & (peak_values > 0))]
                if (peak_values.size >= 3):
                    robust_peak_min = float(np.percentile(peak_values, 15.0))
                    resolution_mm = float(
                        (
                            0.5 * max(
                                float(spacing[0]), float(spacing[1]), DEFAULT_CROSS_SECTION_PIXEL_MM
                            )
                        )
                    )
                    corrected_min = max(0.10, (robust_peak_min - resolution_mm))
                    severity_ref = float(lesion.reference_diameter_mm)
                    if (
                        np.isfinite(lesion.proximal_reference_diameter_mm)
                        and (lesion.proximal_reference_diameter_mm > 0)
                        and (
                            not np.isfinite(lesion.distal_reference_diameter_mm)
                            or (
                                lesion.proximal_reference_diameter_mm
                                > (1.20 * max(lesion.distal_reference_diameter_mm, 1e-6))
                            )
                        )
                        and (float(lesion.position_norm) < 0.72)
                    ):
                        severity_ref = max(
                            severity_ref, (0.95 * float(lesion.proximal_reference_diameter_mm))
                        )
                    corrected_ratio = float(
                        np.clip((1.0 - (corrected_min / max(severity_ref, 1e-6))), 0.0, 0.99)
                    )
                    if (float(lesion.calcification_fraction) >= 0.25):
                        maximum_addition = 0.07
                    elif (ratio < 0.60):
                        maximum_addition = 0.18
                    else:
                        maximum_addition = 0.15
                    corrected_ratio = min(corrected_ratio, (ratio + maximum_addition))
                    if (corrected_ratio > (ratio + 0.025)):
                        lesion.minimum_lumen_diameter_mm = float(corrected_min)
                        lesion.reference_diameter_mm = float(severity_ref)
                        lesion.stenosis_ratio = float(corrected_ratio)
                        lesion.stenosis_interval = severity_interval(corrected_ratio)
                        lesion.partial_volume_correction_mm = float(resolution_mm)
                        lesion.severity_method = (
                            "identity_verified_partial_volume_corrected_short_axis"
                        )
            verified_score = ((
                (
                    (
                        (
                            (
                                (0.24 * np.clip(identity_score, 0.0, 1.0)) + (0.16 * np.clip((edge_support / 0.65), 0.0, 1.0))
                            ) + (0.15 * np.clip((signal_z / 4.0), 0.0, 1.0))
                        ) + (0.13 * np.clip(area_agreement, 0.0, 1.0))
                    ) + (0.12 * np.clip(float(lesion.reference_valid_fraction), 0.0, 1.0))
                ) + (
                    0.10 * np.clip(
                        (
                            max(float(lesion.proximal_recovery), float(lesion.distal_recovery)) / 0.35
                        ),
                        0.0,
                        1.0,
                    )
                )
            ) + (0.10 * np.clip(float(lesion.contour_quality), 0.0, 1.0)))
            lesion.evidence_score = float(np.clip(verified_score, 0.0, 0.98))
            lesion.rejection_reasons = ""

    accepted = sorted(
        [
            r
            for r in results
            if ((int(r.accepted) == 1) and (str(r.assessment_status) == "assessable"))
        ],
        key = lambda r: r.start_distance_mm,
    )
    root_severe = next(
        (
            r
            for r in accepted
            if ((float(r.position_norm) < 0.15) and (float(r.stenosis_ratio) >= 0.75))
        ),
        None,
    )
    if (root_severe is not None):
        filtered_accepted: List[LesionResult] = []
        for r in accepted:
            nav = (
                float(r.navigation_mask_stenosis_ratio)
                if np.isfinite(r.navigation_mask_stenosis_ratio)
                else float("nan")
            )
            suppress = bool(
                (
                    (r is not root_severe)
                    and (float(r.position_norm) < 0.50)
                    and (float(r.stenosis_ratio) < 0.70)
                    and np.isfinite(nav)
                    and (nav < 0.16)
                )
            )
            if suppress:
                r.accepted = 0
                r.rejection_reasons = ";".join(
                    dict.fromkeys(
                        [
                            x
                            for x in (
                                str(r.rejection_reasons) + ";proximal_echo_after_verified_root_severe_lesion"
                            ).split(";")
                            if x
                        ]
                    )
                )
                r.warning_flags = ";".join(
                    dict.fromkeys(
                        [
                            x
                            for x in (str(r.warning_flags) + ";v2_0_5_verifier_rejected").split(";")
                            if x
                        ]
                    )
                )
            else:
                filtered_accepted.append(r)
        accepted = filtered_accepted
    consolidated: List[LesionResult] = []
    cum = centerline.cumulative_mm
    for lesion in accepted:
        if not consolidated:
            consolidated.append(lesion)
            continue
        prev = consolidated[-1]
        gap_start = (int(prev.end_index) + 1)
        gap_end = (int(lesion.start_index) - 1)
        physical_gap_mm = float((lesion.start_distance_mm - prev.end_distance_mm))
        merge = (physical_gap_mm <= 5.0)
        recovery_value = float("nan")
        recovery_span = 0.0
        if ((gap_start <= gap_end) and (gap_end < len(df))):
            gap = np.arange(gap_start, (gap_end + 1), dtype = int)
            caliber = (ds[gap] / np.maximum(dref[gap], 1e-6))
            trusted = ((
                (valid[gap] & (identity_arr[gap] >= 0.48)) & (edge_arr[gap] >= 0.20)
            ) & np.isfinite(caliber))
            if np.any(trusted):
                recovery_value = float(np.nanmedian(caliber[trusted]))
                trusted_indices = gap[trusted]
                recovery_span = (
                    float((cum[trusted_indices[-1]] - cum[trusted_indices[0]]))
                    if (trusted_indices.size >= 2)
                    else 0.0
                )
            if ((recovery_span < 2.2) or not np.isfinite(recovery_value) or (recovery_value < 0.90)):
                merge = True
        else:
            merge = True
        if merge:
            keeper = (
                lesion
                if (
                    (lesion.stenosis_ratio, lesion.evidence_score)
                    > (prev.stenosis_ratio, prev.evidence_score)
                )
                else prev
            )
            keeper.start_index = min(prev.start_index, lesion.start_index)
            keeper.end_index = max(prev.end_index, lesion.end_index)
            keeper.start_distance_mm = float(cum[keeper.start_index])
            keeper.end_distance_mm = float(cum[keeper.end_index])
            keeper.lesion_length_mm = float((keeper.end_distance_mm - keeper.start_distance_mm))
            keeper.start_position_norm = (keeper.start_distance_mm / max(float(cum[-1]), 1e-6))
            keeper.end_position_norm = (keeper.end_distance_mm / max(float(cum[-1]), 1e-6))
            keeper.start_segment = (
                profiles_ops._segment_for_index(
                    keeper.start_index,
                    branch,
                    {
                        "boundaries": [],
                        "method": keeper.segment_method,
                        "confidence": keeper.segment_confidence,
                    },
                )
                if (keeper.start_segment == "unknown")
                else keeper.start_segment
            )
            keeper.recovery_between_adjacent_lesions = recovery_value
            keeper.warning_flags = ";".join(
                dict.fromkeys(
                    [
                        x
                        for x in (
                            ((str(prev.warning_flags) + ";") + str(lesion.warning_flags)) + ";v2_0_5_consolidated"
                        ).split(";")
                        if x
                    ]
                )
            )
            consolidated[-1] = keeper
        else:
            lesion.recovery_between_adjacent_lesions = recovery_value
            consolidated.append(lesion)

    retained_ids = {id(x) for x in consolidated}
    for lesion in results:
        if ((int(lesion.accepted) == 1) and (id(lesion) not in retained_ids)):
            lesion.accepted = 0
            lesion.rejection_reasons = ";".join(
                dict.fromkeys(
                    [
                        x
                        for x in (
                            str(lesion.rejection_reasons) + ";merged_into_same_recovery_process"
                        ).split(";")
                        if x
                    ]
                )
            )
    for i, lesion in enumerate(consolidated, start = 1):
        lesion.lesion_index = int(i)
    rejected = [r for r in results if (int(r.accepted) != 1)]
    return (consolidated + rejected), df

def summarize_stable_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = detection_base_ops.summarize_base_branch(
        case_id, branch, label_value, centerline, profile, lesions
    )
    length = float(result.centerline_length_mm)
    base = base_branch_name(branch)
    if (base == "LM"):
        required_run = min(5.0, max(1.8, (0.50 * length)))
        minimum_fraction = 0.45
    elif (base in ("D1", "D2", "RAMUS", "RAD")):
        required_run = min(8.0, max(3.0, (0.35 * length)))
        minimum_fraction = 0.42
    else:
        required_run = min(12.0, max(5.0, (0.25 * length)))
        minimum_fraction = 0.40
    vf = float(result.valid_cross_section_fraction)
    longest = float(result.longest_valid_run_mm)
    if not centerline.ok:
        result.branch_assessability = "not_assessable_centerline_failed"
        result.branch_assessability_detail = "root_to_tip_centerline_unavailable"
    elif ((vf < minimum_fraction) or (longest < required_run)):
        result.branch_assessability = "not_assessable_CTA_lumen_continuity"
        result.branch_assessability_detail = (
            f"valid_fraction={vf:.3f};longest_valid_run_mm={longest:.2f};"
            f"required_run_mm={required_run:.2f};branch_length_mm={length:.2f}"
        )
    elif (vf < 0.58):
        result.branch_assessability = "limited_cross_section_quality"
        result.branch_assessability_detail = f"valid_fraction={vf:.3f};adaptive_rule_v2_0_5"
    else:
        result.branch_assessability = "assessable"
        result.branch_assessability_detail = "pass_adaptive_length_rule_v2_0_5"
    return result
