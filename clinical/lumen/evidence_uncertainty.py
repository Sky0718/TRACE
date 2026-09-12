from __future__ import annotations

import copy
import math
from typing import List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import branch_context as branch_context_ops
from . import evidence_identity as evidence_identity_ops
from . import profiles as profiles_ops
from . import topology as topology_ops

from .models import (
    ALGORITHM_VERSION,
    BranchResult,
    CenterlineResult,
    DEFAULT_MIN_REPORTABLE_RATIO,
    LesionResult,
    RootAssignment,
    base_branch_name,
    severity_interval,
)

def _set_accepted(
    item: LesionResult,
    rule: str,
    *,
    reliability: str = "moderate",
    coverage_scope: str = "whole_branch",
) -> None:
    topology_ops._accept(item, rule, reliability = reliability, coverage_scope = coverage_scope)
    item.v2_0_8_rule = str(rule)
    item.final_decision = (
        "accepted_local_measurement_v2_0_8"
        if (coverage_scope == "local_only")
        else "accepted_v2_0_8"
    )
    item.verifier_version = "uncertainty_diffuse_multilesion_v2_0_8"
    item.final_state_version = "2.0.8"
    item.status_consistency_checked = 1
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_8_{rule}")

def _update_severity(item: LesionResult, estimate: float, rule: str, weight: float) -> None:
    estimate = float(np.clip(estimate, 0.0, 0.99))
    if (not np.isfinite(estimate) or (estimate <= (float(item.stenosis_ratio) + 1e-6))):
        return
    item.stenosis_ratio = estimate
    item.stenosis_interval = severity_interval(estimate)
    item.formal_ratio_source = str(rule)
    item.severity_method = str(rule)
    item.severity_uncertainty_weight = float(np.clip(weight, 0.0, 1.0))
    item.severity_uncertainty_reason = str(rule)
    item.v2_0_8_rule = profiles_ops._join(item.v2_0_8_rule, rule)
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_8_{rule}")

def _apply_evidence_weighted_severity(item: LesionResult, branch: str) -> None:
    if ((int(item.accepted) != 1) or (base_branch_name(branch) not in ("RCA", "LAD", "LCX"))):
        return
    lower = branch_context_ops._finite(item.stenosis_ratio_lower)
    upper = branch_context_ops._finite(item.stenosis_ratio_upper)
    if not (np.isfinite(lower) and np.isfinite(upper) and (upper > (lower + 0.04))):
        return
    gap = float((upper - lower))
    calc = branch_context_ops._finite(item.calcification_fraction, 0.0)
    area = branch_context_ops._finite(item.diameter_area_agreement, 0.0)
    identity = branch_context_ops._finite(item.lumen_identity_score, 0.0)
    tracking = branch_context_ops._finite(item.tracking_score_median, 0.0)
    cta = branch_context_ops._finite(item.cta_mask_concordance, 0.0)
    nav = branch_context_ops._finite(item.navigation_mask_stenosis_ratio, 0.0)

    if (
        (calc >= 0.12)
        and (area >= 0.65)
        and (identity >= 0.80)
        and (tracking >= 0.70)
        and (cta >= 0.20)
        and (gap <= 0.20)
    ):
        weight = float(np.clip((0.35 + (0.70 * calc)), 0.40, 0.75))
        estimate = float((lower + (weight * gap)))
        estimate = max(float(item.stenosis_ratio), estimate)
        _update_severity(
            item,
            estimate,
            "evidence_weighted_calcified_partial_volume_estimate",
            weight,
        )
        return

    if (
        (calc < 0.12)
        and (0.50 <= lower < 0.70)
        and (gap <= 0.16)
        and (area >= 0.80)
        and (identity >= 0.90)
        and (tracking >= 0.90)
        and (cta >= 0.55)
        and (nav >= 0.12)
    ):
        estimate = max(float(item.stenosis_ratio), float(upper))
        _update_severity(
            item,
            estimate,
            "high_concordance_noncalcified_threshold_crossing_estimate",
            1.0,
        )

def _recovery_metrics(
    item: LesionResult,
    accepted: Sequence[LesionResult],
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> Tuple[bool, float, float, float, float]:
    formal = [x for x in accepted if (int(x.accepted) == 1)]
    if not formal:
        return False, float("nan"), 0.0, 0.0, float("inf")
    nearest = min(
        formal, key = lambda x: abs((float(x.peak_distance_mm) - float(item.peak_distance_mm)))
    )
    peak_gap = abs((float(nearest.peak_distance_mm) - float(item.peak_distance_mm)))
    if (peak_gap < 35.0):
        return False, float("nan"), 0.0, 0.0, peak_gap
    if not (
        (float(item.end_distance_mm) < float(nearest.start_distance_mm))
        or (float(nearest.end_distance_mm) < float(item.start_distance_mm))
    ):
        return False, float("nan"), 0.0, 0.0, peak_gap
    left, right = (
        (item, nearest)
        if (float(item.peak_distance_mm) < float(nearest.peak_distance_mm))
        else (nearest, item)
    )
    recovery, plateau_mm, fraction = profiles_ops._recovery_between(left, right, df, centerline)
    passed = bool(
        (
            np.isfinite(recovery)
            and (recovery >= 0.82)
            and (plateau_mm >= 5.0)
            and (fraction >= 0.40)
        )
    )
    return passed, float(recovery), float(plateau_mm), float(fraction), float(peak_gap)

def _secondary_candidate_eligible(item: LesionResult) -> bool:
    raw = evidence_identity_ops._raw_ratio(item)
    if not (np.isfinite(raw) and (0.45 <= raw <= 0.70)):
        return False
    if (branch_context_ops._finite(item.peak_position_norm, 1.0) > 0.88):
        return False
    if ((int(item.identity_conflict_flag) == 1) or (int(item.resolution_limit_flag) == 1)):
        return False
    if (int(item.calcification_navigation_conflict) == 1):
        return False
    reasons = set(profiles_ops._tokens(item.rejection_reasons))
    allowed = {
        "unstable_lumen_center_tracking",
        "insufficient_sustained_flank_recovery",
        "unreliable_one_sided_reference",
        "too_short",
        "insufficient_flank_recovery",
    }
    if (not reasons or not reasons.issubset(allowed)):
        return False
    if not (
        ("unstable_lumen_center_tracking" in reasons)
        or ("insufficient_sustained_flank_recovery" in reasons)
    ):
        return False
    return bool(
        (
            (branch_context_ops._finite(item.diameter_area_agreement, 0.0) >= 0.70)
            and (branch_context_ops._finite(item.cta_mask_concordance, 0.0) >= 0.55)
            and (branch_context_ops._finite(item.tracking_score_median, 0.0) >= 0.72)
            and (branch_context_ops._finite(item.reference_valid_fraction, 0.0) >= 0.50)
            and (branch_context_ops._finite(item.lumen_identity_score, 0.0) >= 0.72)
        )
    )

def _segment_uncertainty(items: Sequence[LesionResult]) -> None:
    order = {"proximal": 0, "mid": 1, "distal": 2}
    inv = {0: "proximal", 1: "mid", 2: "distal"}
    formal = [x for x in items if (int(x.accepted) == 1)]
    nonformal = [x for x in items if (int(x.accepted) == 0)]
    for item in formal:
        competitors: List[LesionResult] = []
        for other in nonformal:
            raw = evidence_identity_ops._raw_ratio(other)
            if not np.isfinite(raw):
                continue
            if (raw < max(0.50, (float(item.stenosis_ratio) - 0.10))):
                continue
            if (branch_context_ops._finite(other.evidence_score, 0.0) < 0.85):
                continue
            if (abs((float(other.peak_distance_mm) - float(item.peak_distance_mm))) < 10.0):
                continue
            if (str(other.peak_segment) == str(item.peak_segment)):
                continue
            competitors.append(other)
        segments = [str(item.start_segment), str(item.peak_segment), str(item.end_segment)]
        segments.extend(str(x.peak_segment) for x in competitors)
        valid = [order[s] for s in segments if (s in order)]
        if valid:
            lo, hi = min(valid), max(valid)
            item.segment_uncertainty_span = (inv[lo] if (lo == hi) else f"{inv[lo]}-{inv[hi]}")
        else:
            item.segment_uncertainty_span = str(item.segment_span)
        if competitors:
            item.competing_segment_candidates = ";".join(
                f"{x.peak_segment}:{evidence_identity_ops._raw_ratio(x):.3f}@{float(x.peak_distance_mm):.1f}mm"
                for x in sorted(competitors, key = lambda x: float(x.peak_distance_mm))
            )
            item.segment_review_required = 1
            item.clinical_review_priority = (
                "high" if (max(evidence_identity_ops._raw_ratio(x) for x in competitors) >= 0.70) else "moderate"
            )
            item.segment_alternative = ";".join(sorted({str(x.peak_segment) for x in competitors}))
            item.warning_flags = profiles_ops._join(
                item.warning_flags, "v2_0_8_competing_segment_candidate"
            )
        elif (int(item.segment_review_required) == 1):
            item.clinical_review_priority = "moderate"
            item.segment_alternative = item.segment_uncertainty_span
        elif item.severity_uncertainty_reason:
            item.clinical_review_priority = "moderate"
        else:
            item.clinical_review_priority = "routine"

def detect_recovery_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = branch_context_ops.detect_short_branch_checked_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    items = [copy.deepcopy(x) for x in base_results]
    base = base_branch_name(branch)

    if (base in ("RCA", "LAD", "LCX")):
        accepted = [x for x in items if (int(x.accepted) == 1)]
        for item in sorted(
            [
                x
                for x in items
                if ((int(x.accepted) == 0) and _secondary_candidate_eligible(x))
            ],
            key = lambda x: float(x.evidence_score),
            reverse = True,
        ):
            passed, recovery, plateau, fraction, peak_gap = _recovery_metrics(
                item, accepted, df, centerline
            )
            if not passed:
                continue
            item.diffuse_multilesion_recovery = 1
            item.diffuse_multilesion_recovery_median = float(recovery)
            item.diffuse_multilesion_recovery_plateau_mm = float(plateau)
            item.diffuse_multilesion_recovery_fraction = float(fraction)
            item.multilesion_independence_score = float(
                (
                    (
                        np.clip(((recovery - 0.78) / 0.18), 0.0, 1.0) * np.clip((plateau / 8.0), 0.0, 1.0)
                    ) * np.clip((fraction / 0.65), 0.0, 1.0)
                )
            )
            item.multilesion_recovery_plateau_mm = float(plateau)
            item.independence_from_all_neighbours = 1
            direct = float(np.clip(evidence_identity_ops._raw_ratio(item), 0.45, 0.69))
            item.stenosis_ratio = direct
            item.stenosis_ratio_lower = direct
            item.stenosis_ratio_upper = max(direct, branch_context_ops._finite(item.stenosis_ratio_upper, direct))
            item.stenosis_interval = severity_interval(direct)
            item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
            item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
            item.formal_ratio_source = "direct_valid_cross_sections_diffuse_secondary_peak"
            item.severity_method = "diffuse_disease_separated_secondary_direct_measurement"
            item.clinical_review_priority = "moderate"
            _set_accepted(
                item,
                "diffuse_disease_recovery_proven_secondary_lesion",
                reliability = "moderate",
                coverage_scope = str((item.coverage_scope or "whole_branch")),
            )
            accepted.append(item)

    branch_context_ops._consolidate_same_process(items, df, centerline)

    for item in items:
        if (int(item.accepted) == 1):
            _apply_evidence_weighted_severity(item, branch)

    _segment_uncertainty(items)

    for item in items:
        if (int(item.accepted) == 1):
            local = (str(item.coverage_scope) == "local_only")
            rule = str(
                (
                    item.v2_0_8_rule
                    or item.v2_0_7_rule
                    or item.v2_0_5_rule
                    or "passed_v2_0_8_final_audit"
                )
            )
            _set_accepted(
                item,
                rule,
                reliability = str((item.measurement_reliability or "moderate")),
                coverage_scope = ("local_only" if local else "whole_branch"),
            )
        else:
            topology_ops._sync_final_state(item)
            item.final_state_version = "2.0.8"
            item.verifier_version = "uncertainty_diffuse_multilesion_v2_0_8"
            item.status_consistency_checked = 1
            if str(item.final_decision).endswith("v2_0_7"):
                item.final_decision = str(item.final_decision).replace("v2_0_7", "v2_0_8")
            if not item.v2_0_8_rule:
                item.v2_0_8_rule = str((item.v2_0_7_rule or "v2_0_8_state_sync"))

    formal = sorted(
        [x for x in items if (int(x.accepted) == 1)],
        key = lambda x: float(x.peak_distance_mm),
    )
    for index, item in enumerate(formal, start = 1):
        item.lesion_index = int(index)

    df["stenosis_v2_0_8_profile"] = 1
    return (formal + [x for x in items if (int(x.accepted) != 1)]), df

def summarize_recovery_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = branch_context_ops.summarize_short_branch_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    result.final_state_version = "2.0.8"
    formal = [x for x in lesions if (int(x.accepted) == 1)]
    result.formal_lesion_count = int(len(formal))
    result.lesion_count = int(len(formal))
    result.review_required_lesion_count = int(
        sum((int(x.segment_review_required) == 1) for x in formal)
    )
    result.severity_uncertainty_lesion_count = int(
        sum(bool(str(x.severity_uncertainty_reason)) for x in formal)
    )
    result.diffuse_secondary_lesion_count = int(
        sum((int(x.diffuse_multilesion_recovery) == 1) for x in formal)
    )
    if formal:
        maximum = max(formal, key = lambda x: float(x.stenosis_ratio))
        result.maximum_stenosis_ratio = float(maximum.stenosis_ratio)
        result.maximum_stenosis_interval = str(maximum.stenosis_interval)
        result.maximum_stenosis_segment = str(maximum.peak_segment)
        result.maximum_stenosis_position_norm = float(maximum.peak_position_norm)
    return result

def _set_rule(item: LesionResult, rule: str) -> None:
    item.v2_0_9_rule = profiles_ops._join(item.v2_0_9_rule, str(rule))
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_9_{rule}")

def _area_equivalent_ratio(item: LesionResult) -> float:
    area = branch_context_ops._finite(item.area_stenosis_ratio)
    if not np.isfinite(area):
        return float("nan")
    return float(
        np.clip((1.0 - math.sqrt(max(0.0, (1.0 - float(np.clip(area, 0.0, 0.999)))))), 0.0, 0.99)
    )

def _resolution_upper(item: LesionResult, spacing: Sequence[float]) -> float:
    dmin = branch_context_ops._finite(item.minimum_lumen_diameter_mm)
    ref_d = branch_context_ops._finite(item.reference_diameter_mm)
    if not (np.isfinite(dmin) and np.isfinite(ref_d) and (ref_d > 1e-6)):
        return float("nan")
    sp = np.asarray(list(spacing), dtype = np.float64)
    inplane = (float(np.min(sp[:2])) if (sp.size >= 2) else float(np.min(sp)))
    inplane = max(inplane, 0.05)
    residual = max((dmin - inplane), (0.15 * inplane))
    return float(np.clip((1.0 - (residual / ref_d)), 0.0, 0.97))

def _quality_support(item: LesionResult) -> float:
    values = np.asarray(
        [
            branch_context_ops._finite(item.diameter_area_agreement, 0.0),
            branch_context_ops._finite(item.lumen_identity_score, 0.0),
            branch_context_ops._finite(item.tracking_score_median, 0.0),
            branch_context_ops._finite(item.reference_valid_fraction, 0.0),
        ],
        dtype = np.float64,
    )
    return float(np.clip(np.mean(np.clip(values, 0.0, 1.0)), 0.0, 1.0))

def _apply_blooming_protection(item: LesionResult, branch: str) -> bool:
    base = base_branch_name(branch)
    if ((int(item.accepted) != 1) or (base not in ("RCA", "LAD", "LCX"))):
        return False
    formal = float(np.clip(item.stenosis_ratio, 0.0, 0.99))
    calc = branch_context_ops._finite(item.calcification_fraction, 0.0)
    identity = branch_context_ops._finite(item.lumen_identity_score, 0.0)
    cta = branch_context_ops._finite(item.cta_mask_concordance, 0.0)
    tracking = branch_context_ops._finite(item.tracking_score_median, 0.0)
    nav = branch_context_ops._finite(item.navigation_mask_stenosis_ratio, 0.0)
    area_eq = _area_equivalent_ratio(item)
    if not (
        (formal >= 0.70)
        and (calc >= 0.45)
        and (identity < 0.80)
        and (cta < 0.45)
        and (tracking < 0.78)
        and (nav < 0.20)
        and np.isfinite(area_eq)
        and ((formal - area_eq) >= 0.12)
    ):
        return False
    adjusted = float(np.clip(((0.75 * area_eq) + (0.25 * formal)), 0.0, formal))
    old_formal = formal
    item.stenosis_ratio_direct_measured = float(
        (evidence_identity_ops._raw_ratio(item) if np.isfinite(evidence_identity_ops._raw_ratio(item)) else old_formal)
    )
    item.blooming_overestimate_flag = 1
    item.blooming_adjusted_ratio = adjusted
    item.blooming_adjustment_reason = "heavy_calcification_poor_identity_area_equivalent_consensus"
    item.stenosis_ratio = adjusted
    item.stenosis_interval = severity_interval(adjusted)
    item.stenosis_ratio_lower = adjusted
    item.stenosis_ratio_upper = max(old_formal, branch_context_ops._finite(item.stenosis_ratio_upper, old_formal))
    item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
    item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
    item.formal_ratio_source = "blooming_protected_area_equivalent_consensus"
    item.severity_method = "blooming_protected_area_equivalent_consensus"
    item.severity_uncertainty_reason = "calcification_blooming_overestimate_interval"
    item.severity_uncertainty_weight = 0.75
    item.measurement_reliability = "moderate"
    item.clinical_review_priority = "high"
    _set_rule(item, "blooming_overestimate_downward_consensus")
    return True

def _apply_resolution_uncertainty(
    item: LesionResult, branch: str, spacing: Sequence[float]
) -> bool:
    if ((int(item.accepted) != 1) or (int(item.blooming_overestimate_flag) == 1)):
        return False
    base = base_branch_name(branch)
    formal = float(np.clip(item.stenosis_ratio, 0.0, 0.99))
    direct = float(np.clip(evidence_identity_ops._raw_ratio(item), 0.0, 0.99))
    item.stenosis_ratio_direct_measured = direct
    full_pixel_upper = _resolution_upper(item, spacing)
    existing_upper = branch_context_ops._finite(item.stenosis_ratio_upper)
    if (np.isfinite(existing_upper) and (existing_upper > (formal + 0.04))):
        upper = (
            min(full_pixel_upper, existing_upper)
            if np.isfinite(full_pixel_upper)
            else existing_upper
        )
    else:
        upper = full_pixel_upper
    item.stenosis_ratio_resolution_upper = upper
    if (not np.isfinite(upper) or (upper <= (formal + 0.04))):
        return False
    dmin = branch_context_ops._finite(item.minimum_lumen_diameter_mm)
    sp = np.asarray(list(spacing), dtype = np.float64)
    inplane = (float(np.min(sp[:2])) if (sp.size >= 2) else float(np.min(sp)))
    near_resolution = bool((np.isfinite(dmin) and (dmin <= (2.6 * max(inplane, 0.05)))))
    if not near_resolution:
        return False
    identity = branch_context_ops._finite(item.lumen_identity_score, 0.0)
    tracking = branch_context_ops._finite(item.tracking_score_median, 0.0)
    ref_valid = branch_context_ops._finite(item.reference_valid_fraction, 0.0)
    area = branch_context_ops._finite(item.diameter_area_agreement, 0.0)
    cta = branch_context_ops._finite(item.cta_mask_concordance, 0.0)
    nav = branch_context_ops._finite(item.navigation_mask_stenosis_ratio, 0.0)
    calc = branch_context_ops._finite(item.calcification_fraction, 0.0)
    support = _quality_support(item)
    source = str((item.formal_ratio_source or item.severity_method)).lower()
    if (("low_hu" in source) or (branch_context_ops._finite(item.peak_relative_lumen_hu, 1.0) < 0.45)):
        return False

    weight = 0.0
    rule = ""
    if (base in ("D1", "D2", "RAMUS", "RAD")):
        short_support = branch_context_ops._finite(item.short_branch_independent_support_score, 0.0)
        if not (
            (formal >= 0.50)
            and (short_support >= 0.72)
            and (ref_valid >= 0.70)
            and (area >= 0.60)
            and (identity >= 0.88)
            and (tracking >= 0.78)
            and ((calc >= 0.20) or (nav >= 0.10) or (cta >= 0.50))
        ):
            return False
        weight = float(np.clip((0.35 + (0.20 * support)), 0.40, 0.55))
        rule = "short_branch_subvoxel_bounded_estimate"
    elif (base in ("RCA", "LAD", "LCX")):
        if not (
            (formal >= 0.55)
            and (ref_valid >= 0.70)
            and (area >= 0.75)
            and (identity >= 0.85)
            and (tracking >= 0.78)
        ):
            return False
        current_lower = branch_context_ops._finite(item.stenosis_ratio_lower, formal)
        if (calc < 0.12):
            if (formal >= 0.70):
                if not (
                    (current_lower < 0.65)
                    and ("partial_volume" in source)
                    and (cta >= 0.55)
                    and (nav >= 0.12)
                    and (identity >= 0.90)
                    and (tracking >= 0.88)
                ):
                    return False
                weight = float(np.clip((0.18 + (0.18 * support)), 0.24, 0.36))
                rule = "high_quality_partial_volume_crossing_bounded_estimate"
            else:
                if ((cta < 0.45) or (nav < 0.10)):
                    return False
                if not any(
                    (token in source)
                    for token in (
                        "partial_volume",
                        "threshold_crossing",
                        "high_concordance",
                        "orphan_main",
                        "strong_threshold",
                    )
                ):
                    return False
                weight = float(np.clip((0.22 + (0.22 * support)), 0.28, 0.44))
                rule = "noncalcified_near_resolution_bounded_estimate"
        elif (calc < 0.45):
            if (
                ((formal >= 0.70) and (current_lower >= 0.65))
                or (formal >= 0.80)
                or (cta < 0.50)
                or (nav < 0.08)
            ):
                return False
            weight = float(np.clip((0.18 + (0.18 * support)), 0.22, 0.36))
            rule = "mixed_plaque_near_resolution_bounded_estimate"
        else:
            return False
    else:
        return False

    estimate = float(np.clip((formal + (weight * (upper - formal))), formal, upper))
    if (estimate <= (formal + 0.015)):
        return False
    item.stenosis_ratio = estimate
    item.stenosis_interval = severity_interval(estimate)
    current_lower = branch_context_ops._finite(item.stenosis_ratio_lower, formal)
    item.stenosis_ratio_lower = min(current_lower, formal)
    item.stenosis_ratio_upper = max(branch_context_ops._finite(item.stenosis_ratio_upper, formal), upper)
    item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
    item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
    item.formal_ratio_source = rule
    item.severity_method = rule
    item.resolution_uncertainty_applied = 1
    item.resolution_uncertainty_mm = max(inplane, 0.05)
    item.severity_uncertainty_weight = weight
    item.severity_uncertainty_reason = rule
    item.clinical_review_priority = "moderate"
    _set_rule(item, rule)
    return True

def _enforce_interval_consistency(item: LesionResult) -> None:
    formal = float(np.clip(item.stenosis_ratio, 0.0, 0.99))
    direct = evidence_identity_ops._raw_ratio(item)
    item.stenosis_ratio_direct_measured = direct
    lower = branch_context_ops._finite(item.stenosis_ratio_lower)
    upper = branch_context_ops._finite(item.stenosis_ratio_upper)
    if not np.isfinite(lower):
        lower = min(formal, (direct if np.isfinite(direct) else formal))
    if not np.isfinite(upper):
        upper = max(formal, (direct if np.isfinite(direct) else formal))
    lower = min(lower, formal)
    upper = max(upper, formal)
    if (upper < lower):
        lower, upper = upper, lower
    item.stenosis_ratio_lower = float(np.clip(lower, 0.0, 0.99))
    item.stenosis_ratio_upper = float(np.clip(upper, item.stenosis_ratio_lower, 0.99))
    item.stenosis_interval = severity_interval(formal)
    item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
    item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
    item.severity_interval_consistency_checked = 1

def _mark_reported_segment(item: LesionResult) -> None:
    span = str(
        (item.segment_uncertainty_span or item.segment_span or item.peak_segment or item.segment)
    )
    if ((int(item.segment_review_required) == 1) or bool(str(item.competing_segment_candidates))):
        item.reported_segment = span
        item.reported_segment_span = span
        item.reported_segment_status = "uncertain_review_required"
    else:
        item.reported_segment = str((item.peak_segment or item.segment))
        item.reported_segment_span = str((item.segment_span or item.reported_segment))
        item.reported_segment_status = "anatomical_or_fallback_best_estimate"

def _mark_borderline_review(item: LesionResult) -> None:
    if (int(item.accepted) == 1):
        return
    if ((int(item.identity_conflict_flag) == 1) or (int(item.resolution_limit_flag) == 1)):
        return
    raw = evidence_identity_ops._raw_ratio(item)
    evidence = branch_context_ops._finite(item.evidence_score, 0.0)
    identity = branch_context_ops._finite(item.lumen_identity_score, 0.0)
    tracking = branch_context_ops._finite(item.tracking_score_median, 0.0)
    area = branch_context_ops._finite(item.diameter_area_agreement, 0.0)
    cta = branch_context_ops._finite(item.cta_mask_concordance, 0.0)
    ref_valid = branch_context_ops._finite(item.reference_valid_fraction, 0.0)
    reasons = set(profiles_ops._tokens(item.rejection_reasons))
    seam_tokens = {
        "root_or_label_seam_measurement_unreliable",
        "branch_origin_or_label_seam_zone",
        "dynamic_branch_seam_exclusion",
        "root_seam_or_ostial_measurement_unreliable",
    }
    if (
        (0.25 <= raw <= 0.55)
        and (evidence >= 0.80)
        and (identity >= 0.75)
        and (tracking >= 0.70)
        and (area >= 0.70)
        and (cta >= 0.50)
        and (ref_valid >= 0.50)
        and bool((reasons & seam_tokens))
    ):
        item.borderline_review_candidate = 1
        item.borderline_review_reason = "proximal_seam_or_true_ostial_review"
        item.clinical_review_priority = "high"
        _set_rule(item, "borderline_proximal_seam_review_only")
        return
    allowed = {
        "below_report_threshold",
        "below_branch_specific_threshold",
        "insufficient_local_prominence",
        "insufficient_flank_recovery",
        "insufficient_sustained_flank_recovery",
        "too_short",
        "too_short_for_stable_lesion",
        "unreliable_one_sided_reference",
    }
    standard_borderline = bool(
        (
            (0.25 <= raw <= 0.50)
            and (evidence >= 0.82)
            and (identity >= 0.82)
            and (tracking >= 0.80)
            and (area >= 0.82)
            and (cta >= 0.60)
            and (ref_valid >= 0.70)
        )
    )
    low_ratio_but_cross_modal = bool(
        (
            (0.20 <= raw < 0.25)
            and (evidence >= 0.70)
            and (identity >= 0.78)
            and (tracking >= 0.72)
            and (area >= 0.90)
            and (cta >= 0.70)
            and (branch_context_ops._finite(item.navigation_mask_stenosis_ratio, 0.0) >= 0.15)
            and (ref_valid >= 0.60)
        )
    )
    if (reasons and reasons.issubset(allowed) and (standard_borderline or low_ratio_but_cross_modal)):
        item.borderline_review_candidate = 1
        item.borderline_review_reason = (
            "cross_modal_low_ratio_borderline_candidate"
            if low_ratio_but_cross_modal
            else "high_quality_borderline_nonformal_candidate"
        )
        item.clinical_review_priority = "moderate"
        _set_rule(item, "borderline_high_quality_review_only")

def detect_uncertainty_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = detect_recovery_checked_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    items = [copy.deepcopy(x) for x in base_results]
    for item in items:
        item.v2_0_9_rule = ""
        if (int(item.accepted) == 1):
            _apply_blooming_protection(item, branch)
            _apply_resolution_uncertainty(item, branch, spacing)
            _enforce_interval_consistency(item)
            _mark_reported_segment(item)
            item.final_state_version = "2.0.9"
            item.verifier_version = "bounded_resolution_review_v2_0_9"
            item.status_consistency_checked = 1
            if str(item.final_decision).endswith("v2_0_8"):
                item.final_decision = str(item.final_decision).replace("v2_0_8", "v2_0_9")
            _set_rule(item, (item.v2_0_9_rule or "passed_v2_0_9_final_audit"))
        else:
            _mark_borderline_review(item)
            _mark_reported_segment(item)
            item.final_state_version = "2.0.9"
            item.verifier_version = "bounded_resolution_review_v2_0_9"
            item.status_consistency_checked = 1
            if str(item.final_decision).endswith("v2_0_8"):
                item.final_decision = str(item.final_decision).replace("v2_0_8", "v2_0_9")
            if not item.v2_0_9_rule:
                item.v2_0_9_rule = "v2_0_9_state_sync"
    formal = sorted(
        [x for x in items if (int(x.accepted) == 1)], key = lambda x: float(x.peak_distance_mm)
    )
    for index, item in enumerate(formal, start = 1):
        item.lesion_index = int(index)
    df["stenosis_v2_0_9_profile"] = 1
    return (formal + [x for x in items if (int(x.accepted) != 1)]), df

def summarize_uncertainty_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = summarize_recovery_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    result.final_state_version = "2.0.9"
    formal = [x for x in lesions if (int(x.accepted) == 1)]
    result.formal_lesion_count = int(len(formal))
    result.lesion_count = int(len(formal))
    result.borderline_review_candidate_count = int(
        sum((int(x.borderline_review_candidate) == 1) for x in lesions if (int(x.accepted) == 0))
    )
    result.formal_segment_review_count = int(
        sum((int(x.segment_review_required) == 1) for x in formal)
    )
    result.formal_severity_uncertainty_count = int(
        sum(
            (
                (int(x.resolution_uncertainty_applied) == 1)
                or (int(x.blooming_overestimate_flag) == 1)
                or bool(str(x.severity_uncertainty_reason))
            )
            for x in formal
        )
    )
    result.review_required_lesion_count = result.formal_segment_review_count
    result.severity_uncertainty_lesion_count = result.formal_severity_uncertainty_count
    if formal:
        maximum = max(formal, key = lambda x: float(x.stenosis_ratio))
        result.maximum_stenosis_ratio = float(maximum.stenosis_ratio)
        result.maximum_stenosis_interval = str(maximum.stenosis_interval)
        result.maximum_stenosis_segment = str((maximum.reported_segment or maximum.peak_segment))
        result.maximum_stenosis_position_norm = float(maximum.peak_position_norm)
    return result
