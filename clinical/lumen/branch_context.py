from __future__ import annotations

import copy
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import evidence_identity as evidence_identity_ops
from . import evidence_resolution as evidence_resolution_ops
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

def _reject_or_limit(
    item: LesionResult,
    reason: str,
    *,
    not_assessable: bool = False,
    conflict_flag: bool = False,
) -> None:
    item.accepted = 0
    item.clinical_review_reason = str(reason)
    item.verifier_version = "calcium_conflict_local_coverage_v2_0_5"
    item.v2_0_5_rule = str(reason)
    if conflict_flag:
        item.calcification_navigation_conflict = 1
        item.identity_conflict_flag = 1
        item.identity_conflict_reason = profiles_ops._join(item.identity_conflict_reason, str(reason))
    if not_assessable:
        item.assessment_status = "not_assessable"
        item.not_assessable_reason = profiles_ops._join(item.not_assessable_reason, str(reason))
        item.rejection_reasons = profiles_ops._join(item.rejection_reasons, str(reason))
        item.final_decision = "not_assessable_v2_0_5"
        item.measurement_reliability = "not_assessable"
        item.clinical_output_status = "not_assessable"
    else:
        item.rejection_reasons = profiles_ops._join(item.rejection_reasons, str(reason))
        item.final_decision = "rejected_v2_0_5"
        item.measurement_reliability = "low"
        item.clinical_output_status = "rejected"
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_5_{reason}")

def _is_locally_measurable_in_incomplete_branch(
    item: LesionResult,
    centerline: CenterlineResult,
) -> bool:
    total = (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0)
    if (total <= 1e-6):
        return False
    peak_fraction = (float(item.peak_distance_mm) / total)
    recovery = max(
        evidence_resolution_ops._float(item.proximal_recovery, 0.0),
        evidence_resolution_ops._float(item.distal_recovery, 0.0),
    )
    return bool(
        (
            str(item.assessment_status).startswith("not_assessable_incomplete_branch_coverage")
            and np.isfinite(item.stenosis_ratio)
            and (float(item.stenosis_ratio) >= 0.35)
            and (str(item.peak_segment) != "distal")
            and (peak_fraction <= 0.65)
            and (float(item.peak_valid_run_mm) >= max(5.0, (float(item.lesion_length_mm) + 2.0)))
            and (float(item.reference_valid_fraction) >= 0.75)
            and (float(item.lumen_identity_score) >= 0.88)
            and (float(item.tracking_score_median) >= 0.75)
            and (float(item.tracking_supported_fraction) >= 0.75)
            and (float(item.center_inside_fraction) >= 0.90)
            and (float(item.profile_signal_z) >= 8.0)
            and (float(item.diameter_area_agreement) >= 0.85)
            and (recovery >= 0.45)
        )
    )

def _apply_remaining_context_filters(
    item: LesionResult,
    branch: str,
) -> None:
    if (int(item.accepted) != 1):
        return
    base = base_branch_name(branch)
    if (base not in ("RCA", "LAD", "LCX")):
        item.clinical_output_status = "formal"
        return

    raw = evidence_identity_ops._raw_ratio(item)
    nav = evidence_resolution_ops._float(item.navigation_mask_stenosis_ratio)
    rel_hu = evidence_resolution_ops._float(item.peak_relative_lumen_hu)
    calc = evidence_resolution_ops._float(item.calcification_fraction, 0.0)
    area_agree = evidence_resolution_ops._float(item.diameter_area_agreement, 0.0)
    recovery = max(
        evidence_resolution_ops._float(item.proximal_recovery, 0.0),
        evidence_resolution_ops._float(item.distal_recovery, 0.0),
    )
    length = float(item.lesion_length_mm)
    rule = str(item.v2_0_5_rule)

    if (
        (rule == "orphan_main_moderate_identity_rescue")
        and (calc < 0.12)
        and (not np.isfinite(rel_hu) or (rel_hu < 0.90))
        and (not np.isfinite(nav) or (nav < 0.12))
    ):
        _reject_or_limit(
            item,
            "orphan_moderate_without_independent_plaque_or_lumen_support",
            not_assessable = False,
        )
        return

    moderate_calcification_conflict = bool(
        (
            (raw < 0.60)
            and (calc >= 0.20)
            and np.isfinite(nav)
            and (nav < 0.07)
            and (
                (area_agree < 0.80)
                or (np.isfinite(rel_hu) and (rel_hu < 0.95))
                or ((length < 6.5) and (recovery < 0.55))
            )
        )
    )
    heavy_calcified_mild_spike = bool(
        (
            (raw < 0.45)
            and (calc >= 0.45)
            and np.isfinite(nav)
            and (nav < 0.115)
            and (length < 10.0)
            and (recovery < 0.55)
        )
    )
    if (moderate_calcification_conflict or heavy_calcified_mild_spike):
        _reject_or_limit(
            item,
            "calcification_blooming_navigation_conflict",
            not_assessable = True,
            conflict_flag = True,
        )
        return

    if ((raw < 0.45) and (calc < 0.15) and np.isfinite(rel_hu) and (rel_hu < 0.55)):
        _reject_or_limit(
            item,
            "low_HU_mild_target_lumen_identity_conflict",
            not_assessable = True,
            conflict_flag = True,
        )
        return

    item.clinical_output_status = "formal"
    item.clinical_review_reason = ""
    item.verifier_version = "calcium_conflict_local_coverage_v2_0_5"
    if not item.v2_0_5_rule:
        item.v2_0_5_rule = "passed_v2_0_5_context_filter"

def detect_context_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = evidence_resolution_ops.detect_resolution_checked_lesions(
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
        if (int(item.accepted) == 1):
            _apply_remaining_context_filters(item, branch)

    accepted_now = [x for x in items if (int(x.accepted) == 1)]
    for item in items:
        if (int(item.accepted) == 1):
            continue
        if not _is_locally_measurable_in_incomplete_branch(item, centerline):
            continue
        if evidence_resolution_ops._near_existing(item, accepted_now):
            continue
        item.accepted = 1
        item.assessment_status = "locally_assessable_in_incomplete_branch"
        item.not_assessable_reason = ""
        item.rejection_reasons = ""
        item.final_decision = "accepted_local_measurement_v2_0_5"
        item.measurement_reliability = "moderate"
        item.local_coverage_override = 1
        item.clinical_output_status = "formal_limited_branch_coverage"
        item.clinical_review_reason = (
            "local lesion and reference sections are valid, but distal branch "
            "coverage remains incomplete"
        )
        item.warning_flags = profiles_ops._join(
            item.warning_flags,
            "v2_0_5_locally_assessable_despite_incomplete_distal_coverage",
        )
        item.verifier_version = "calcium_conflict_local_coverage_v2_0_5"
        item.v2_0_5_rule = "locally_assessable_despite_incomplete_distal_coverage"
        if not np.isfinite(item.stenosis_ratio_lower):
            item.stenosis_ratio_lower = float(item.stenosis_ratio)
        if not np.isfinite(item.stenosis_ratio_upper):
            item.stenosis_ratio_upper = float(item.stenosis_ratio)
        item.stenosis_interval_lower = severity_interval(float(item.stenosis_ratio_lower))
        item.stenosis_interval_upper = severity_interval(float(item.stenosis_ratio_upper))
        item.formal_ratio_source = (
            item.formal_ratio_source or "direct_valid_local_sections_in_incomplete_branch"
        )
        accepted_now.append(item)

    formal = [x for x in items if (int(x.accepted) == 1)]
    if (formal and (max(float(x.stenosis_ratio) for x in formal) >= 0.50)):
        for item in formal:
            if (float(item.stenosis_ratio) >= 0.35):
                continue
            nav = evidence_resolution_ops._float(item.navigation_mask_stenosis_ratio)
            calc = evidence_resolution_ops._float(item.calcification_fraction, 0.0)
            recovery = max(
                evidence_resolution_ops._float(item.proximal_recovery, 0.0),
                evidence_resolution_ops._float(item.distal_recovery, 0.0),
            )
            if ((not np.isfinite(nav) or (nav < 0.18)) and (calc < 0.10) and (recovery < 0.40)):
                item.secondary_low_confidence_flag = 1
                _reject_or_limit(
                    item,
                    "secondary_low_confidence_mild_candidate",
                    not_assessable = False,
                )

    final_accepted = sorted(
        [x for x in items if (int(x.accepted) == 1)],
        key = lambda x: float(x.peak_distance_mm),
    )
    for index, item in enumerate(final_accepted, start = 1):
        item.lesion_index = int(index)
        item.verifier_version = "calcium_conflict_local_coverage_v2_0_5"
        if not item.clinical_output_status:
            item.clinical_output_status = "formal"
        if not item.v2_0_5_rule:
            item.v2_0_5_rule = "passed_v2_0_5_context_filter"

    nonaccepted = [x for x in items if (int(x.accepted) != 1)]
    for item in nonaccepted:
        if not item.clinical_output_status:
            item.clinical_output_status = (
                "not_assessable" if (str(item.assessment_status) != "assessable") else "rejected"
            )
        item.verifier_version = (item.verifier_version or "calcium_conflict_local_coverage_v2_0_5")

    df["stenosis_v2_0_5_profile"] = 1
    return (final_accepted + nonaccepted), df

def summarize_context_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = evidence_resolution_ops.summarize_resolution_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    local_override = any(
        ((int(x.accepted) == 1) and (int(x.local_coverage_override) == 1)) for x in lesions
    )
    if local_override:
        result.branch_assessability = (
            "limited_incomplete_branch_coverage_with_locally_assessable_lesion"
        )
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail,
            "local lesion measurement valid; distal branch coverage incomplete;v2_0_5",
        )
    else:
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail,
            "v2_0_5_calcium_conflict_and_local_coverage_checked",
        )
    return result

def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return (out if np.isfinite(out) else float(default))
    except Exception:
        return float(default)

def _direct_ratio(item: LesionResult) -> float:
    lower = _finite(item.stenosis_ratio_lower)
    if np.isfinite(lower):
        return float(np.clip(lower, 0.0, 1.0))
    return float(np.clip(evidence_identity_ops._raw_ratio(item), 0.0, 1.0))

def _set_rejected(item: LesionResult, reason: str) -> None:
    topology_ops._reject(item, reason, not_assessable = False)
    item.v2_0_7_rule = str(reason)
    item.final_decision = "rejected_v2_0_7"
    item.verifier_version = "evidence_concordant_short_branch_v2_0_7"
    item.final_state_version = "2.0.7"
    item.status_consistency_checked = 1

def _set_accepted(
    item: LesionResult,
    rule: str,
    *,
    reliability: str = "moderate",
    coverage_scope: str = "whole_branch",
) -> None:
    topology_ops._accept(item, rule, reliability = reliability, coverage_scope = coverage_scope)
    item.v2_0_7_rule = str(rule)
    item.final_decision = (
        "accepted_local_measurement_v2_0_7"
        if (coverage_scope == "local_only")
        else "accepted_v2_0_7"
    )
    item.verifier_version = "evidence_concordant_short_branch_v2_0_7"
    item.final_state_version = "2.0.7"
    item.status_consistency_checked = 1
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_7_{rule}")

def _short_branch_audit(
    item: LesionResult,
    centerline: CenterlineResult,
) -> Tuple[bool, str, float]:
    direct = _direct_ratio(item)
    nav = _finite(item.navigation_mask_stenosis_ratio)
    calc = _finite(item.calcification_fraction, 0.0)
    cta = _finite(item.cta_mask_concordance)
    rel_hu = _finite(item.peak_relative_lumen_hu)
    prominence = _finite(item.local_prominence, 0.0)
    area_agreement = _finite(item.diameter_area_agreement, 0.0)
    identity = _finite(item.lumen_identity_score, 0.0)
    tracking = _finite(item.tracking_score_median, 0.0)
    tracked_fraction = _finite(item.tracking_supported_fraction, 0.0)
    center_inside = _finite(item.center_inside_fraction, 0.0)
    reference_fraction = _finite(item.reference_valid_fraction, 0.0)
    position = _finite(item.peak_position_norm, 0.0)
    lesion_length = max(0.0, _finite(item.lesion_length_mm, 0.0))
    total_length = (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0)
    lesion_fraction = (lesion_length / max(total_length, 1e-6))
    parent_used = (int(item.parent_reference_used) == 1)

    plaque_support = (calc >= 0.15)
    enhanced_lumen_support = (np.isfinite(rel_hu) and (rel_hu >= 0.75))
    navigation_support = (np.isfinite(nav) and (nav >= 0.18) and np.isfinite(cta) and (cta >= 0.55))
    independent_support = (plaque_support or enhanced_lumen_support or navigation_support)

    score = 0.0
    score += (0.18 * float(np.clip(((direct - 0.35) / 0.35), 0.0, 1.0)))
    score += (0.16 * float(np.clip((prominence / 0.35), 0.0, 1.0)))
    score += (0.16 * float(np.clip((area_agreement / 0.85), 0.0, 1.0)))
    if np.isfinite(cta):
        score += (0.12 * float(np.clip(((cta - 0.35) / 0.45), 0.0, 1.0)))
    score += (0.10 * float(np.clip(((identity - 0.80) / 0.20), 0.0, 1.0)))
    score += (0.10 * float(np.clip(((tracking - 0.70) / 0.25), 0.0, 1.0)))
    score += (0.10 * float(independent_support))
    score += (0.08 * float(np.clip(reference_fraction, 0.0, 1.0)))

    if (direct < 0.40):
        return False, "short_branch_direct_narrowing_below_reliable_resolution", score
    if (
        (lesion_length > 14.0)
        or (lesion_fraction > 0.55)
        or (str(item.segment_span) == "proximal-distal")
    ):
        return False, "short_branch_diffuse_reference_artifact", score
    if ((position > 0.78) and (not np.isfinite(nav) or (nav < 0.18)) and (calc < 0.10)):
        return False, "short_branch_distal_taper_without_independent_support", score
    if (
        (identity < 0.82)
        or (tracking < 0.75)
        or (tracked_fraction < 0.70)
        or (center_inside < 0.85)
        or (reference_fraction < 0.58)
    ):
        return False, "short_branch_lumen_tracking_or_reference_unreliable", score
    if parent_used:
        if (
            (prominence < 0.18)
            or (area_agreement < 0.58)
            or not np.isfinite(cta)
            or (cta < 0.42)
            or not independent_support
        ):
            return False, "parent_reference_without_independent_daughter_lumen_support", score
    else:
        if (
            (prominence < 0.18)
            or (area_agreement < 0.65)
            or not np.isfinite(cta)
            or (cta < 0.40)
            or not independent_support
        ):
            return False, "short_branch_candidate_without_independent_image_support", score
    return True, "short_branch_evidence_concordant", score

def _independent_from_neighbours(
    item: LesionResult,
    accepted: Sequence[LesionResult],
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> Tuple[bool, float, float, float]:
    others = [x for x in accepted if ((x is not item) and (int(x.accepted) == 1))]
    if not others:
        return True, 1.0, float("inf"), 1.0
    left_candidates = [
        x for x in others if (float(x.peak_distance_mm) < float(item.peak_distance_mm))
    ]
    right_candidates = [
        x for x in others if (float(x.peak_distance_mm) > float(item.peak_distance_mm))
    ]
    neighbours: List[LesionResult] = []
    if left_candidates:
        neighbours.append(max(left_candidates, key = lambda x: float(x.peak_distance_mm)))
    if right_candidates:
        neighbours.append(min(right_candidates, key = lambda x: float(x.peak_distance_mm)))
    if not neighbours:
        return True, 1.0, float("inf"), 1.0
    scores: List[float] = []
    plateaus: List[float] = []
    fractions: List[float] = []
    for neighbour in neighbours:
        if not (
            (float(item.end_distance_mm) < float(neighbour.start_distance_mm))
            or (float(neighbour.end_distance_mm) < float(item.start_distance_mm))
        ):
            return False, 0.0, 0.0, 0.0
        peak_gap = abs((float(item.peak_distance_mm) - float(neighbour.peak_distance_mm)))
        if (peak_gap < 8.0):
            return False, 0.0, 0.0, 0.0
        left, right = (
            (item, neighbour)
            if (float(item.peak_distance_mm) < float(neighbour.peak_distance_mm))
            else (neighbour, item)
        )
        recovery, plateau_mm, fraction = profiles_ops._recovery_between(left, right, df, centerline)
        if not np.isfinite(recovery):
            return False, 0.0, 0.0, 0.0
        score = float(np.clip(((recovery - 0.78) / 0.22), 0.0, 1.0))
        score *= float(np.clip((plateau_mm / 3.0), 0.0, 1.0))
        score *= float(np.clip((fraction / 0.75), 0.0, 1.0))
        if not ((recovery >= 0.88) and (plateau_mm >= 2.0) and (fraction >= 0.65)):
            return False, score, float(plateau_mm), float(fraction)
        scores.append(score)
        plateaus.append(float(plateau_mm))
        fractions.append(float(fraction))
    return (
        True,
        (float(min(scores)) if scores else 1.0),
        (float(min(plateaus)) if plateaus else float("inf")),
        (float(min(fractions)) if fractions else 1.0),
    )

def _duplicate_rank(item: LesionResult) -> Tuple[float, float, float, float]:
    primary = (
        0.0 if (str(item.v2_0_5_rule) == "recovery_proven_independent_secondary_lesion") else 1.0
    )
    direct = (1.0 if (int(item.parent_reference_used) == 0) else 0.0)
    focality = -max(0.0, _finite(item.lesion_length_mm, 0.0))
    evidence = _finite(item.evidence_score, 0.0)
    return primary, direct, evidence, focality

def _consolidate_same_process(
    items: Sequence[LesionResult],
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> None:
    group_counter = 0
    changed = True
    while changed:
        changed = False
        accepted = sorted(
            [x for x in items if (int(x.accepted) == 1)],
            key = lambda x: float(x.peak_distance_mm),
        )
        for i in range((len(accepted) - 1)):
            left, right = accepted[i], accepted[(i + 1)]
            overlap = not (
                (float(left.end_distance_mm) < float(right.start_distance_mm))
                or (float(right.end_distance_mm) < float(left.start_distance_mm))
            )
            peak_gap = abs((float(right.peak_distance_mm) - float(left.peak_distance_mm)))
            ratio_close = (abs((_direct_ratio(left) - _direct_ratio(right))) <= 0.03)
            prominence_close = (
                abs(
                    (
                        _finite(left.local_prominence, 0.0) - _finite(right.local_prominence, 0.0)
                    )
                )
                <= 0.03
            )
            area_close = (
                abs(
                    (
                        _finite(left.diameter_area_agreement, 0.0) - _finite(right.diameter_area_agreement, 0.0)
                    )
                )
                <= 0.05
            )
            same_fingerprint = (
                (peak_gap <= 12.0) and ratio_close and prominence_close and area_close
            )
            independent, _, _, _ = _independent_from_neighbours(right, [left], df, centerline)
            if (not overlap and not same_fingerprint and independent):
                continue
            if (not overlap and not same_fingerprint):
                continue
            group_counter += 1
            group_id = f"{left.branch}_duplicate_group_{group_counter:02d}"
            winner, loser = (
                (left, right)
                if (_duplicate_rank(left) >= _duplicate_rank(right))
                else (right, left)
            )
            winner.duplicate_group_id = group_id
            loser.duplicate_group_id = group_id
            winner.deduplication_action_v2_0_7 = "kept_best_measurement"
            loser.deduplication_action_v2_0_7 = "rejected_duplicate_measurement"
            _set_rejected(loser, "duplicate_or_overlapping_measurement_v2_0_7")
            changed = True
            break

def _lm_limited_reference_candidate(item: LesionResult) -> Tuple[bool, float]:
    reasons = set(profiles_ops._tokens(item.rejection_reasons))
    if ("centerline_CTA_continuity_failed" in reasons):
        return False, float("nan")
    allowed = {
        "terminal_endpoint_unreliable",
        "terminal_taper_zone",
        "insufficient_local_prominence",
        "insufficient_flank_recovery",
    }
    if (reasons and not reasons.issubset(allowed)):
        return False, float("nan")
    raw = evidence_identity_ops._raw_ratio(item)
    nav = _finite(item.navigation_mask_stenosis_ratio)
    cta = _finite(item.cta_mask_concordance)
    identity = _finite(item.lumen_identity_score, 0.0)
    area_agreement = _finite(item.diameter_area_agreement, 0.0)
    area_ratio = _finite(item.area_stenosis_ratio, 0.0)
    calc = _finite(item.calcification_fraction, 0.0)
    ref_fraction = _finite(item.reference_valid_fraction, 0.0)
    prominence = _finite(item.local_prominence, 0.0)
    peak_position = _finite(item.peak_position_norm, 0.0)
    if not (
        np.isfinite(raw)
        and (0.35 <= raw <= 0.80)
        and np.isfinite(nav)
        and (nav >= 0.15)
        and np.isfinite(cta)
        and (cta >= 0.50)
        and (identity >= 0.82)
        and (area_agreement >= 0.70)
        and (area_ratio >= 0.45)
        and (calc >= 0.15)
        and (ref_fraction >= 0.75)
        and (prominence >= 0.15)
        and (0.15 <= peak_position <= 0.80)
    ):
        return False, float("nan")
    estimate = float(np.sqrt((max(raw, 0.0) * max(nav, 0.0))))
    if (estimate < 0.25):
        return False, estimate
    return True, float(np.clip(estimate, 0.25, raw))

def detect_short_branch_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = topology_ops.detect_topology_checked_lesions(
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

    if (base in ("D1", "D2", "RAMUS", "RAD")):
        for item in items:
            if (int(item.accepted) != 1):
                continue
            passed, reason, score = _short_branch_audit(item, centerline)
            item.short_branch_independent_support_score = float(score)
            item.short_branch_audit_reason = str(reason)
            if not passed:
                _set_rejected(item, reason)
                continue
            direct = _direct_ratio(item)
            if (int(item.parent_reference_used) == 1):
                parent_upper = _finite(item.parent_reference_stenosis_ratio, direct)
                item.parent_reference_advisory_only = 1
                item.stenosis_ratio = float(direct)
                item.stenosis_ratio_lower = float(direct)
                item.stenosis_ratio_upper = float(max(direct, parent_upper))
                item.stenosis_interval = severity_interval(item.stenosis_ratio)
                item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
                item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
                item.formal_ratio_source = (
                    "direct_valid_cross_sections_parent_reference_upper_bound"
                )
                item.severity_method = (
                    "direct_CTA_measurement_with_parent_contact_uncertainty_upper_bound"
                )
            item.v2_0_7_rule = "short_branch_evidence_concordant"
            item.warning_flags = profiles_ops._join(
                item.warning_flags, "v2_0_7_short_branch_evidence_concordant"
            )

    if (base in ("RCA", "LAD", "LCX")):
        accepted_now = [x for x in items if (int(x.accepted) == 1)]
        for item in list(accepted_now):
            if (str(item.v2_0_5_rule) != "recovery_proven_independent_secondary_lesion"):
                continue
            independent, score, plateau, fraction = _independent_from_neighbours(
                item, accepted_now, df, centerline
            )
            item.multilesion_independence_score = float(score)
            item.multilesion_recovery_plateau_mm = float(plateau)
            item.recovery_plateau_fraction = float(fraction)
            item.independence_from_all_neighbours = int(independent)
            if not independent:
                _set_rejected(item, "secondary_lesion_not_independent_from_all_neighbours")

    _consolidate_same_process(items, df, centerline)

    if ((base == "LM") and not any((int(x.accepted) == 1) for x in items)):
        eligible: List[Tuple[float, LesionResult]] = []
        for item in items:
            ok, estimate = _lm_limited_reference_candidate(item)
            if ok:
                eligible.append((float(estimate), item))
        if eligible:
            estimate, item = max(eligible, key = lambda pair: pair[0])
            raw = evidence_identity_ops._raw_ratio(item)
            item.stenosis_ratio = float(estimate)
            item.stenosis_ratio_lower = float(estimate)
            item.stenosis_ratio_upper = float(max(estimate, raw))
            item.stenosis_interval = severity_interval(item.stenosis_ratio)
            item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
            item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
            item.formal_ratio_source = "LM_geometric_mean_CTA_navigation_limited_reference"
            item.severity_method = "short_LM_one_sided_reference_conservative_estimate"
            _set_accepted(
                item,
                "LM_short_branch_limited_reference_rescue",
                reliability = "moderate",
            )

    for item in items:
        item.segment_review_required = int(
            (
                (float(item.segment_confidence) < 0.60)
                or ("rooted_length_only" in str(item.anatomical_landmark_evidence))
            )
        )
        if (int(item.accepted) == 1):
            local = (str(item.coverage_scope) == "local_only")
            _set_accepted(
                item,
                str((item.v2_0_7_rule or item.v2_0_5_rule or "passed_v2_0_7_final_audit")),
                reliability = str((item.measurement_reliability or "moderate")),
                coverage_scope = ("local_only" if local else "whole_branch"),
            )
        else:
            topology_ops._sync_final_state(item)
            item.final_state_version = "2.0.7"
            item.verifier_version = "evidence_concordant_short_branch_v2_0_7"
            item.status_consistency_checked = 1
            if str(item.final_decision).endswith("v2_0_6"):
                item.final_decision = str(item.final_decision).replace("v2_0_6", "v2_0_7")

    formal = sorted(
        [x for x in items if (int(x.accepted) == 1)],
        key = lambda x: float(x.peak_distance_mm),
    )
    for index, item in enumerate(formal, start = 1):
        item.lesion_index = int(index)

    df["stenosis_v2_0_7_profile"] = 1
    return (formal + [x for x in items if (int(x.accepted) != 1)]), df

def summarize_short_branch_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = topology_ops.summarize_topology_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    result.final_state_version = "2.0.7"
    formal = [x for x in lesions if (int(x.accepted) == 1)]
    local = [x for x in formal if (str(x.coverage_scope) == "local_only")]
    result.formal_lesion_count = int(len(formal))
    result.local_only_lesion_count = int(len(local))
    if local:
        result.coverage_scope = "local_only"
        result.whole_branch_negative_interpretation_valid = 0
        result.branch_assessability = (
            "limited_incomplete_branch_coverage_with_locally_assessable_lesion"
        )
    elif (str(result.branch_assessability) == "assessable"):
        result.coverage_scope = "whole_branch"
        result.whole_branch_negative_interpretation_valid = 1
    else:
        result.coverage_scope = "insufficient"
        result.whole_branch_negative_interpretation_valid = 0

    unresolved = [
        x
        for x in lesions
        if (
            (int(x.accepted) == 0)
            and str(x.assessment_status).startswith("not_assessable")
            and (evidence_identity_ops._raw_ratio(x) >= 0.45)
            and (_finite(x.evidence_score, 0.0) >= 0.85)
        )
    ]
    if (not formal and unresolved):
        result.branch_assessability = "not_assessable_high_signal_unresolved"
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail,
            "v2_0_7_high_signal_candidate_unresolved",
        )
        result.coverage_scope = "insufficient"
        result.whole_branch_negative_interpretation_valid = 0
    return result
