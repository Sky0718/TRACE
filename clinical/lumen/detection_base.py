from __future__ import annotations

import math
from typing import List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter

from . import profiles as profiles_ops
from . import sections as sections_ops
from . import topology as topology_ops

from .models import (
    BranchResult,
    CenterlineResult,
    DEFAULT_MIN_REPORTABLE_RATIO,
    LesionResult,
    RootAssignment,
    base_branch_name,
    severity_interval,
)

def detect_base_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    if (profile.empty or not centerline.ok):
        return [], profile
    df = profile.copy()
    cum = pd.to_numeric(df["distance_mm"], errors = "coerce").to_numpy(float)
    d_raw = pd.to_numeric(df["min_diameter_mm"], errors = "coerce").to_numpy(float)
    a_raw = pd.to_numeric(df["lumen_area_mm2"], errors = "coerce").to_numpy(float)
    quality = pd.to_numeric(df["quality_score"], errors = "coerce").fillna(0.0).to_numpy(float)
    valid = (pd.to_numeric(df["valid"], errors = "coerce").fillna(0).to_numpy(int) > 0)
    boundary = pd.to_numeric(df["boundary_support"], errors = "coerce").fillna(0.0).to_numpy(float)
    calc_frac = (
        pd.to_numeric(df["calcification_fraction"], errors = "coerce").fillna(0.0).to_numpy(float)
    )
    unresolved = (
        pd.to_numeric(df.get("unresolved_lumen", pd.Series(0, index = df.index)), errors = "coerce")
        .fillna(0)
        .to_numpy(int)
        > 0
    )
    outer_fraction = (
        pd.to_numeric(
            df.get("outer_mask_fraction", pd.Series(0.0, index = df.index)), errors = "coerce"
        )
        .fillna(0.0)
        .to_numpy(float)
    )
    step = (float(np.median(np.diff(cum))) if (cum.size >= 2) else 0.55)

    d, interp_d = profiles_ops._interpolate_small_gaps_physical(
        np.where(valid, d_raw, np.nan), valid, cum, maximum_gap_mm = 2.2
    )
    a, interp_a = profiles_ops._interpolate_small_gaps_physical(
        np.where(valid, a_raw, np.nan), valid, cum, maximum_gap_mm = 2.2
    )
    interpolated = (interp_d | interp_a)
    df["interpolated_for_detection"] = interpolated.astype(int)
    detection_valid = (valid | interpolated)
    if (np.sum((np.isfinite(d) & (d > 0))) < 8):
        return [], df

    dfill = d.copy()
    good_d = (np.isfinite(dfill) & (dfill > 0))
    dfill[~good_d] = float(np.nanmedian(dfill[good_d]))
    afill = a.copy()
    good_a = (np.isfinite(afill) & (afill > 0))
    afill[~good_a] = float(np.nanmedian(afill[good_a]))
    med_size = max(3, int(round((1.2 / max(step, 0.2)))))
    if ((med_size % 2) == 0):
        med_size += 1
    ds = gaussian_filter1d(
        median_filter(dfill, size = med_size, mode = "nearest"),
        sigma = max(0.7, (0.65 / max(step, 0.2))),
        mode = "nearest",
    )
    ars = gaussian_filter1d(
        median_filter(afill, size = med_size, mode = "nearest"),
        sigma = max(0.7, (0.65 / max(step, 0.2))),
        mode = "nearest",
    )
    win = max(7, int(round((18.0 / max(step, 0.2)))))
    if ((win % 2) == 0):
        win += 1
    dref = gaussian_filter1d(
        np.maximum(percentile_filter(ds, 82, size = win, mode = "nearest"), ds),
        sigma = max(1.0, (1.2 / max(step, 0.2))),
        mode = "nearest",
    )
    aref = gaussian_filter1d(
        np.maximum(percentile_filter(ars, 82, size = win, mode = "nearest"), ars),
        sigma = max(1.0, (1.2 / max(step, 0.2))),
        mode = "nearest",
    )
    dratio = np.clip((1.0 - (ds / np.maximum(dref, 1e-6))), 0.0, 1.0)
    area_diam_ratio = np.clip(
        (1.0 - np.sqrt((np.maximum(ars, 0.0) / np.maximum(aref, 1e-6)))), 0.0, 1.0
    )
    aratio = np.clip((1.0 - (ars / np.maximum(aref, 1e-6))), 0.0, 1.0)
    signal = gaussian_filter1d(
        ((0.68 * dratio) + (0.32 * area_diam_ratio)),
        sigma = max(0.65, (0.55 / max(step, 0.2))),
        mode = "nearest",
    )
    signal[~detection_valid] = 0.0
    for name, values in (
        ("smoothed_min_diameter_mm", ds),
        ("reference_diameter_mm", dref),
        ("diameter_stenosis_ratio", dratio),
        ("smoothed_lumen_area_mm2", ars),
        ("reference_area_mm2", aref),
        ("area_stenosis_ratio", aratio),
        ("area_equivalent_diameter_stenosis_ratio", area_diam_ratio),
        ("candidate_signal", signal),
    ):
        df[name] = values

    candidate_mask = ((
        signal >= max(0.065, (0.65 * float(minimum_reportable_ratio)))
    ) & detection_valid)
    candidate_mask = profiles_ops._close_small_1d_gaps(candidate_mask, cum, maximum_gap_mm = 1.5)
    raw_runs = profiles_ops._runs_from_mask(candidate_mask)
    intervals: List[Tuple[int, int]] = []
    for start, end in raw_runs:
        length = (float((cum[end] - cum[start])) if (end > start) else 0.0)
        peak_signal = float(np.max(signal[start : (end + 1)]))
        if ((length < 1.2) and (peak_signal < 0.50)):
            continue
        intervals.extend(profiles_ops._split_recovery_intervals(start, end, signal, ds, dref, cum, step))

    unresolved_runs = profiles_ops._runs_from_mask((unresolved & (outer_fraction >= 0.03)))
    unresolved_intervals: List[Tuple[int, int]] = []
    for start, end in unresolved_runs:
        span = (float((cum[end] - cum[start])) if (end > start) else 0.0)
        left_valid = np.any(
            valid[max(0, (start - max(2, int(round((3.0 / max(step, 0.2))))))) : start]
        )
        right_valid = np.any(
            valid[
                (end + 1) : min(
                    valid.size, ((end + 1) + max(2, int(round((3.0 / max(step, 0.2))))))
                )
            ]
        )
        if ((span >= 0.8) and left_valid and right_valid):
            unresolved_intervals.append((start, end))

    results: List[LesionResult] = []
    total = (float(cum[-1]) if cum.size else 0.0)
    seam = (
        float(centerline.root_assignment.seam_exclusion_mm) if centerline.root_assignment else 0.0
    )
    base = base_branch_name(branch)
    segment_info = topology_ops._segment_boundaries(branch, centerline, branch_masks, spacing, branch_roots)
    df["anatomic_segment"] = [profiles_ops._segment_for_index(i, branch, segment_info) for i in range(len(df))]
    df["segment_method"] = str(segment_info.get("method", "normalized_length_fallback"))
    df["segment_confidence"] = float(segment_info.get("confidence", 0.0))

    def make_result(
        start: int,
        peak: int,
        end: int,
        provisional: int,
        assessable: bool,
        not_reason: str,
        suspected_near_occlusion: int = 0,
    ) -> LesionResult:
        peak_dist = float(cum[peak])
        lesion_len = (float((cum[end] - cum[start])) if (end >= start) else 0.0)
        start_pos = float((cum[start] / max(total, 1e-6)))
        peak_pos = float((peak_dist / max(total, 1e-6)))
        end_pos = float((cum[end] / max(total, 1e-6)))
        start_seg = profiles_ops._segment_for_index(start, branch, segment_info)
        peak_seg = profiles_ops._segment_for_index(peak, branch, segment_info)
        end_seg = profiles_ops._segment_for_index(end, branch, segment_info)
        span_label = profiles_ops._segment_span(start_seg, peak_seg, end_seg)
        start_vox, start_phys = profiles_ops._lesion_coordinates(centerline, start)
        peak_vox, peak_phys = profiles_ops._lesion_coordinates(centerline, peak)
        end_vox, end_phys = profiles_ops._lesion_coordinates(centerline, end)

        valid_count, valid_span, peak_run_idx = profiles_ops._valid_run_around_peak(valid, peak, cum)
        peak_neigh = profiles_ops._window_indices(cum, (peak_dist - 1.0), (peak_dist + 1.0))
        peak_valid = (
            peak_neigh[
                ((valid[peak_neigh] & np.isfinite(d_raw[peak_neigh])) & (d_raw[peak_neigh] > 0))
            ]
            if peak_neigh.size
            else np.array([], dtype = int)
        )
        minimum_peak_count = max(3, int(round((1.0 / max(step, 0.2)))))
        if (peak_valid.size >= minimum_peak_count):
            ordered_d = np.sort(d_raw[peak_valid])
            ordered_a = np.sort(
                a_raw[peak_valid][(np.isfinite(a_raw[peak_valid]) & (a_raw[peak_valid] > 0))]
            )
            use_count = min(max(3, int(round((1.1 / max(step, 0.2))))), ordered_d.size)
            min_d = float(np.median(ordered_d[:use_count]))
            min_a = (
                float(np.median(ordered_a[: min(use_count, ordered_a.size)]))
                if ordered_a.size
                else float("nan")
            )
        else:
            min_d = min_a = float("nan")
            assessable = False
            not_reason = (not_reason or "insufficient_continuous_valid_peak_sections")

        prox_idx = profiles_ops._window_indices(cum, (peak_dist - 18.0), (peak_dist - 3.0))
        dist_idx = profiles_ops._window_indices(cum, (peak_dist + 3.0), (peak_dist + 18.0))
        pD, p_count, p_frac = profiles_ops._robust_reference_with_validity(
            ds, quality, valid, prox_idx, percentile = 75.0
        )
        dD, d_count, d_frac = profiles_ops._robust_reference_with_validity(
            ds, quality, valid, dist_idx, percentile = 75.0
        )
        pA, _, _ = profiles_ops._robust_reference_with_validity(ars, quality, valid, prox_idx, percentile = 75.0)
        dA, _, _ = profiles_ops._robust_reference_with_validity(ars, quality, valid, dist_idx, percentile = 75.0)
        avail_d = [x for x in (pD, dD) if (np.isfinite(x) and (x > 0))]
        avail_a = [x for x in (pA, dA) if (np.isfinite(x) and (x > 0))]
        reference_count = int((p_count + d_count))
        reference_fraction = float(((p_frac + d_frac) / 2.0))
        if (len(avail_d) == 2):
            pc = (
                float(np.median(cum[prox_idx[valid[prox_idx]]]))
                if np.any(valid[prox_idx])
                else (peak_dist - 8.0)
            )
            dc = (
                float(np.median(cum[dist_idx[valid[dist_idx]]]))
                if np.any(valid[dist_idx])
                else (peak_dist + 8.0)
            )
            alpha = float(np.clip(((peak_dist - pc) / max((dc - pc), 1e-6)), 0.0, 1.0))
            ref_d = float((((1.0 - alpha) * pD) + (alpha * dD)))
        elif (len(avail_d) == 1):
            ref_d = float(avail_d[0])
        else:
            ref_d = float("nan")
            assessable = False
            not_reason = (not_reason or "insufficient_valid_reference_sections")
        if (len(avail_a) == 2):
            ref_a = float((0.5 * (avail_a[0] + avail_a[1])))
        elif (len(avail_a) == 1):
            ref_a = float(avail_a[0])
        elif np.isfinite(ref_d):
            ref_a = float((math.pi * ((ref_d / 2.0) ** 2)))
        else:
            ref_a = float("nan")

        if (assessable and np.isfinite(min_d) and np.isfinite(ref_d) and (ref_d > 0)):
            ratio = float(np.clip((1.0 - (min_d / ref_d)), 0.0, 1.0))
            area_ratio = (
                float(np.clip((1.0 - (min_a / ref_a)), 0.0, 1.0))
                if (np.isfinite(min_a) and np.isfinite(ref_a) and (ref_a > 0))
                else float("nan")
            )
        else:
            ratio = area_ratio = float("nan")

        p_rec = (
            float(np.clip(((pD - min_d) / max(ref_d, 1e-6)), 0.0, 1.0))
            if (np.isfinite(pD) and np.isfinite(min_d) and np.isfinite(ref_d))
            else 0.0
        )
        d_rec = (
            float(np.clip(((dD - min_d) / max(ref_d, 1e-6)), 0.0, 1.0))
            if (np.isfinite(dD) and np.isfinite(min_d) and np.isfinite(ref_d))
            else 0.0
        )
        idx = np.arange(start, (end + 1), dtype = int)
        q = (float(np.nanmedian(quality[idx])) if idx.size else 0.0)
        bq = (float(np.nanmedian(boundary[idx])) if idx.size else 0.0)
        cf = (float(np.nanmax(calc_frac[idx])) if idx.size else 0.0)
        local_prom = (
            float(
                max(
                    (
                        np.nanmax(signal[idx]) - max(
                            (float(signal[(start - 1)]) if (start > 0) else 0.0),
                            (float(signal[(end + 1)]) if ((end + 1) < signal.size) else 0.0),
                        )
                    ),
                    0.0,
                )
            )
            if idx.size
            else 0.0
        )

        rejection: List[str] = []
        warnings: List[str] = []
        if not assessable:
            rejection.append((not_reason or "not_assessable"))
        if assessable:
            terminal_hard = (8.0 if (base in ("RCA", "LAD", "LCX")) else 5.0)
            near_root = (peak_dist < max(seam, 2.5))
            near_tip = ((total - peak_dist) < max((terminal_hard + 2.0), 10.0))
            if (ratio < float(minimum_reportable_ratio)):
                rejection.append("below_minimum_reportable_ratio")
            if (q < 0.52):
                rejection.append("low_cross_section_quality")
            if (bq < 0.52):
                rejection.append("weak_lumen_boundary_support")
            if (lesion_len < 1.5):
                rejection.append("too_short")
            if ((valid_count < minimum_peak_count) or (valid_span < 0.8)):
                rejection.append("insufficient_continuous_valid_peak_sections")
            if ((reference_count < 4) or (reference_fraction < 0.22)):
                rejection.append("insufficient_valid_reference_sections")
            min_prom = (0.16 if (ratio < 0.50) else 0.075)
            if (local_prom < min_prom):
                rejection.append("insufficient_local_prominence")
            if ((lesion_len > 30.0) and (ratio < 0.50)):
                rejection.append("broad_taper_or_segmentation_change")
            if (near_root and (ratio < 0.50)):
                rejection.append("root_or_label_seam_zone")
            if ((total - peak_dist) < terminal_hard):
                rejection.append("terminal_endpoint_unreliable")
            elif (near_tip and (ratio < 0.50)):
                rejection.append("terminal_taper_zone")
            bilateral = ((p_rec >= 0.08) and (d_rec >= 0.08))
            unilateral = (max(p_rec, d_rec) >= 0.12)
            if not bilateral:
                if ((ratio >= 0.50) and unilateral):
                    warnings.append("one_sided_reference")
                elif (near_root and (ratio >= 0.50) and (d_rec >= 0.10)):
                    warnings.append("possible_ostial_lesion")
                else:
                    rejection.append("insufficient_flank_recovery")
            expected_area = float(np.clip((1.0 - ((1.0 - ratio) ** 2)), 0.0, 1.0))
            area_agreement = (
                (1.0 - abs((float(area_ratio) - expected_area))) if np.isfinite(area_ratio) else 0.0
            )
            if ((area_agreement < 0.42) and (ratio < 0.70)):
                rejection.append("diameter_area_disagreement")
            if (cf > 0.30):
                warnings.append("calcification_blooming_risk")
            if ("cta_lumen_continuity_low" in str(centerline.quality_flag)):
                rejection.append("centerline_CTA_continuity_failed")
            elif (("ambiguous" in str(centerline.quality_flag)) or (
                "failed" in str(centerline.quality_flag)
            )):
                warnings.append("centerline_qc_warning")
            score = ((
                (
                    (
                        (
                            (
                                (
                                    (0.24 * np.clip((ratio / 0.70), 0.0, 1.0)) + (0.16 * np.clip(area_agreement, 0.0, 1.0))
                                ) + (0.15 * np.clip(q, 0.0, 1.0))
                            ) + (0.14 * np.clip(bq, 0.0, 1.0))
                        ) + (0.10 * np.clip((max(p_rec, d_rec) / 0.20), 0.0, 1.0))
                    ) + (0.08 * np.clip((local_prom / 0.20), 0.0, 1.0))
                ) + (0.07 * np.clip((valid_span / 3.0), 0.0, 1.0))
            ) + (0.06 * np.clip((reference_fraction / 0.70), 0.0, 1.0)))
        else:
            area_agreement = 0.0
            score = 0.0
            if suspected_near_occlusion:
                warnings.append("suspected_near_occlusion_requires_review")

        accepted = int((assessable and (len(rejection) == 0)))
        if (accepted and (ratio < 0.25)):
            strong_mild = (
                (p_rec >= 0.15)
                and (d_rec >= 0.15)
                and (local_prom >= 0.18)
                and (q >= 0.65)
                and (bq >= 0.65)
                and (area_agreement >= 0.70)
            )
            if not strong_mild:
                accepted = 0
                rejection.append("weak_mild_lesion_evidence")

        return LesionResult(
            case_id = str(case_id),
            branch = str(branch),
            lesion_index = int(provisional),
            start_index = int(start),
            peak_index = int(peak),
            end_index = int(end),
            start_distance_mm = float(cum[start]),
            peak_distance_mm = float(peak_dist),
            end_distance_mm = float(cum[end]),
            lesion_length_mm = float(lesion_len),
            position_norm = float(peak_pos),
            segment = str(peak_seg),
            stenosis_ratio = float(ratio),
            stenosis_interval = (
                severity_interval(ratio) if np.isfinite(ratio) else "not_assessable"
            ),
            area_stenosis_ratio = float(area_ratio),
            minimum_lumen_diameter_mm = float(min_d),
            reference_diameter_mm = float(ref_d),
            minimum_lumen_area_mm2 = float(min_a),
            reference_area_mm2 = float(ref_a),
            proximal_reference_diameter_mm = float(pD),
            distal_reference_diameter_mm = float(dD),
            proximal_recovery = float(p_rec),
            distal_recovery = float(d_rec),
            local_prominence = float(local_prom),
            candidate_signal = (float(signal[peak]) if np.isfinite(signal[peak]) else 0.0),
            boundary_support = float(bq),
            contour_quality = float(q),
            calcification_fraction = float(cf),
            evidence_score = float(np.clip(score, 0.0, 0.99)),
            accepted = int(accepted),
            rejection_reasons = ";".join(dict.fromkeys(rejection)),
            warning_flags = ";".join(dict.fromkeys(warnings)),
            root_source = (
                centerline.root_assignment.source if centerline.root_assignment else "fallback"
            ),
            centerline_quality = str(centerline.quality_flag),
            assessment_status = ("assessable" if assessable else "not_assessable"),
            not_assessable_reason = ("" if assessable else str(not_reason)),
            start_position_norm = float(start_pos),
            peak_position_norm = float(peak_pos),
            end_position_norm = float(end_pos),
            start_segment = str(start_seg),
            peak_segment = str(peak_seg),
            end_segment = str(end_seg),
            segment_span = str(span_label),
            segment_method = str(segment_info.get("method", "unknown")),
            segment_confidence = float(segment_info.get("confidence", 0.0)),
            start_voxel_x = float(start_vox[0]),
            start_voxel_y = float(start_vox[1]),
            start_voxel_z = float(start_vox[2]),
            peak_voxel_x = float(peak_vox[0]),
            peak_voxel_y = float(peak_vox[1]),
            peak_voxel_z = float(peak_vox[2]),
            end_voxel_x = float(end_vox[0]),
            end_voxel_y = float(end_vox[1]),
            end_voxel_z = float(end_vox[2]),
            start_x_mm = float(start_phys[0]),
            start_y_mm = float(start_phys[1]),
            start_z_mm = float(start_phys[2]),
            peak_x_mm = float(peak_phys[0]),
            peak_y_mm = float(peak_phys[1]),
            peak_z_mm = float(peak_phys[2]),
            end_x_mm = float(end_phys[0]),
            end_y_mm = float(end_phys[1]),
            end_z_mm = float(end_phys[2]),
            peak_valid_run_count = int(valid_count),
            peak_valid_run_mm = float(valid_span),
            peak_section_valid = (int(valid[peak]) if (0 <= peak < valid.size) else 0),
            reference_valid_count = int(reference_count),
            reference_valid_fraction = float(reference_fraction),
            suspected_near_occlusion = int(suspected_near_occlusion),
        )

    provisional = 1
    for start, end in intervals:
        valid_idx = np.arange(start, (end + 1), dtype = int)
        valid_idx = valid_idx[valid[valid_idx]]
        if valid_idx.size:
            peak = int(valid_idx[np.argmax(dratio[valid_idx])])
            result = make_result(start, peak, end, provisional, True, "")
        else:
            peak = int(((start + end) // 2))
            result = make_result(
                start, peak, end, provisional, False, "no_valid_sections_in_candidate", 1
            )
        results.append(result)
        provisional += 1
    for start, end in unresolved_intervals:
        peak = int(((start + end) // 2))
        results.append(
            make_result(start, peak, end, provisional, False, "not_assessable_unresolved_lumen", 1)
        )
        provisional += 1

    assessable = sorted([r for r in results if r.accepted], key = lambda r: r.start_distance_mm)
    consolidated: List[LesionResult] = []
    for lesion in assessable:
        if not consolidated:
            consolidated.append(lesion)
            continue
        prev = consolidated[-1]
        gap_start, gap_end = (prev.end_index + 1), (lesion.start_index - 1)
        merge = (lesion.start_distance_mm <= prev.end_distance_mm)
        recovery_value = float("nan")
        if (gap_start <= gap_end):
            gap_idx = np.arange(gap_start, (gap_end + 1), dtype = int)
            recovery_values = (ds[gap_idx] / np.maximum(dref[gap_idx], 1e-6))
            recovery_value = (
                float(np.nanmedian(recovery_values)) if recovery_values.size else float("nan")
            )
            recovery_span = (float((cum[gap_end] - cum[gap_start])) if (gap_end > gap_start) else 0.0)
            if ((recovery_span < 1.2) or not np.isfinite(recovery_value) or (recovery_value < 0.82)):
                merge = True
        elif ((lesion.start_distance_mm - prev.end_distance_mm) <= 1.5):
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
            keeper.lesion_length_mm = (keeper.end_distance_mm - keeper.start_distance_mm)
            keeper.start_position_norm = (keeper.start_distance_mm / max(total, 1e-6))
            keeper.end_position_norm = (keeper.end_distance_mm / max(total, 1e-6))
            keeper.start_segment = profiles_ops._segment_for_index(keeper.start_index, branch, segment_info)
            keeper.end_segment = profiles_ops._segment_for_index(keeper.end_index, branch, segment_info)
            keeper.segment_span = profiles_ops._segment_span(
                keeper.start_segment, keeper.peak_segment, keeper.end_segment
            )
            sv, sp = profiles_ops._lesion_coordinates(centerline, keeper.start_index)
            ev, ep = profiles_ops._lesion_coordinates(centerline, keeper.end_index)
            keeper.start_voxel_x, keeper.start_voxel_y, keeper.start_voxel_z = map(float, sv)
            keeper.end_voxel_x, keeper.end_voxel_y, keeper.end_voxel_z = map(float, ev)
            keeper.start_x_mm, keeper.start_y_mm, keeper.start_z_mm = map(float, sp)
            keeper.end_x_mm, keeper.end_y_mm, keeper.end_z_mm = map(float, ep)
            keeper.recovery_between_adjacent_lesions = recovery_value
            consolidated[-1] = keeper
        else:
            lesion.recovery_between_adjacent_lesions = recovery_value
            consolidated.append(lesion)

    for i, lesion in enumerate(consolidated, start = 1):
        lesion.lesion_index = int(i)
    rejected_or_unassessable = [r for r in results if not r.accepted]
    return (consolidated + rejected_or_unassessable), df

def summarize_base_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    accepted = [r for r in lesions if (int(r.accepted) == 1)]
    valid = (
        pd.to_numeric(profile.get("valid", pd.Series(dtype = float)), errors = "coerce")
        .fillna(0)
        .to_numpy(int)
        > 0
    )
    d = pd.to_numeric(
        profile.get("min_diameter_mm", pd.Series(dtype = float)), errors = "coerce"
    ).to_numpy(float)
    a = pd.to_numeric(
        profile.get("lumen_area_mm2", pd.Series(dtype = float)), errors = "coerce"
    ).to_numpy(float)
    dv = d[((valid & np.isfinite(d)) & (d > 0))]
    av = a[((valid & np.isfinite(a)) & (a > 0))]
    length = (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0)
    endpoint = (
        float(np.linalg.norm((centerline.path_physical_mm[-1] - centerline.path_physical_mm[0])))
        if centerline.ok
        else 0.0
    )
    maximum = (max(accepted, key = lambda r: r.stenosis_ratio) if accepted else None)
    vf = (float(np.mean(valid)) if valid.size else 0.0)
    step = (
        float(np.median(np.diff(centerline.cumulative_mm)))
        if (centerline.cumulative_mm.size >= 2)
        else 0.55
    )
    _, longest_mm = sections_ops._longest_true_run(valid, step)
    if not centerline.ok:
        assess = "not_assessable_centerline_failed"
        detail = "root_to_tip_centerline_unavailable"
    elif (
        ("cta_lumen_continuity_low" in str(centerline.quality_flag))
        or (vf < 0.35)
        or (longest_mm < 12.0)
    ):
        assess = "not_assessable_CTA_lumen_continuity"
        detail = f"valid_fraction={vf:.3f};longest_valid_run_mm={longest_mm:.2f}"
    elif (vf < 0.55):
        assess = "limited_cross_section_quality"
        detail = f"valid_fraction={vf:.3f}"
    else:
        assess = "assessable"
        detail = "pass"
    topology = centerline.topology
    return BranchResult(
        case_id = str(case_id),
        branch = str(branch),
        label_value = int(label_value),
        centerline_length_mm = float(length),
        endpoint_distance_mm = float(endpoint),
        tortuosity = (float((length / max(endpoint, 1e-6))) if length else 0.0),
        root_source = (
            centerline.root_assignment.source if centerline.root_assignment else "fallback"
        ),
        centerline_method = str(centerline.method),
        centerline_quality = str(centerline.quality_flag),
        cross_section_count = int(len(profile)),
        valid_cross_section_count = int(np.sum(valid)),
        valid_cross_section_fraction = float(vf),
        lumen_diameter_mean_mm = (float(np.mean(dv)) if dv.size else 0.0),
        lumen_diameter_median_mm = (float(np.median(dv)) if dv.size else 0.0),
        lumen_diameter_min_mm = (float(np.min(dv)) if dv.size else 0.0),
        lumen_area_mean_mm2 = (float(np.mean(av)) if av.size else 0.0),
        lesion_count = int(len(accepted)),
        maximum_stenosis_ratio = (float(maximum.stenosis_ratio) if maximum else 0.0),
        maximum_stenosis_interval = (str(maximum.stenosis_interval) if maximum else "0%"),
        maximum_stenosis_segment = (str(maximum.segment_span) if maximum else "none"),
        maximum_stenosis_position_norm = (float(maximum.position_norm) if maximum else -1.0),
        branch_assessability = str(assess),
        longest_valid_run_mm = float(longest_mm),
        cta_centerline_support_fraction = float(topology.get("cta_support_fraction", vf)),
        cta_recentered_mean_shift_mm = float(topology.get("cta_recentered_mean_shift_mm", 0.0)),
        cta_recentered_max_shift_mm = float(topology.get("cta_recentered_max_shift_mm", 0.0)),
        cta_trimmed_tip_mm = float(topology.get("cta_trimmed_tip_mm", 0.0)),
        branch_assessability_detail = str(detail),
        segment_method_default = str(
            (maximum.segment_method if maximum else "normalized_length_fallback")
        ),
    )
