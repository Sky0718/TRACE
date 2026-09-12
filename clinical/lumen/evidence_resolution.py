from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import evidence_identity as evidence_identity_ops
from . import profiles as profiles_ops

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

def _float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return (result if np.isfinite(result) else float(default))

def _candidate_key(lesion: LesionResult) -> Tuple[int, float]:
    return int(lesion.peak_index), round(float(lesion.peak_distance_mm), 2)

def _resolution_floor_mm(spacing: Sequence[float]) -> float:
    sp = np.asarray(spacing, dtype = float).reshape(-1)
    in_plane = (float(np.max(sp[:2])) if (sp.size >= 2) else 0.35)
    return float(max(1.15, (3.0 * in_plane)))

def _metrics(
    lesion: LesionResult,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
) -> Dict[str, float]:
    metrics = dict(profiles_ops._candidate_metrics(lesion, profile, centerline))
    lesion.peak_relative_lumen_hu = float(metrics.get("peak_relative_lumen_hu", np.nan))
    lesion.peak_radial_cv = float(metrics.get("peak_radial_cv", np.nan))
    lesion.peak_eccentricity = float(metrics.get("peak_eccentricity", np.nan))
    return metrics

def _nav(lesion: LesionResult) -> float:
    return _float(lesion.navigation_mask_stenosis_ratio)

def _concord(lesion: LesionResult) -> float:
    return _float(lesion.cta_mask_concordance)

def _recovery(lesion: LesionResult) -> float:
    return float(
        max(
            _float(lesion.proximal_recovery, 0.0),
            _float(lesion.distal_recovery, 0.0),
        )
    )

def _short_focal_severe_signature(
    lesion: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
) -> bool:
    base = base_branch_name(branch)
    return bool(
        (
            (base in ("RCA", "LAD", "LCX"))
            and (evidence_identity_ops._raw_ratio(lesion) >= 0.80)
            and (1.8 <= float(lesion.lesion_length_mm) <= 8.5)
            and (float(metrics.get("identity", 0.0)) >= 0.74)
            and (float(metrics.get("edge", 0.0)) >= 0.52)
            and (float(metrics.get("center_inside_fraction", 0.0)) >= 0.90)
            and (float(lesion.profile_signal_z) >= 15.0)
            and (float(lesion.diameter_area_agreement) >= 0.88)
            and (_recovery(lesion) >= 0.72)
        )
    )

def _hard_decision(
    lesion: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
    spacing: Sequence[float],
) -> Tuple[str, str, str]:
    base = base_branch_name(branch)
    small = (base in ("D1", "D2", "RAMUS", "RAD"))
    raw = evidence_identity_ops._raw_ratio(lesion)
    nav = _nav(lesion)
    concord = _concord(lesion)
    rel_hu = _float(metrics.get("peak_relative_lumen_hu", np.nan))
    identity = float(metrics.get("identity", 0.0))
    tracking = float(metrics.get("tracking_score", 0.0))
    overlap = float(metrics.get("tracking_overlap", 0.0))
    edge = float(metrics.get("edge", 0.0))
    calc = float(lesion.calcification_fraction)
    reference = _float(lesion.reference_diameter_mm, 0.0)
    pos = _float(lesion.position_norm, -1.0)
    severe_short_signature = _short_focal_severe_signature(lesion, metrics, branch)

    resolution_floor = _resolution_floor_mm(spacing)
    if (
        (raw >= 0.48)
        and (pos >= 0.60)
        and (reference > 0.0)
        and (reference < resolution_floor)
        and (not np.isfinite(nav) or (nav < 0.16))
    ):
        lesion.resolution_limit_flag = 1
        return (
            "not_assessable",
            f"subresolution_distal_lumen_reference={reference:.3f}mm;floor={resolution_floor:.3f}mm",
            "limited_by_spatial_resolution",
        )

    if (
        (int(lesion.repair_succeeded) == 1)
        and (raw >= 0.55)
        and np.isfinite(rel_hu)
        and (rel_hu < 0.70)
        and (not np.isfinite(concord) or (concord < 0.55))
        and (not np.isfinite(nav) or (nav < 0.16))
        and not severe_short_signature
    ):
        return (
            "not_assessable",
            "repaired_candidate_low_relative_HU_and_low_navigation_support",
            "identity_conflict",
        )

    calcified_rescue_support = (
        (calc >= 0.35)
        and np.isfinite(rel_hu)
        and (rel_hu >= 1.15)
        and (identity >= 0.78)
        and (edge >= 0.70)
    )
    strong_direct_lumen_support = (
        np.isfinite(rel_hu)
        and (rel_hu >= 0.90)
        and (identity >= 0.85)
        and (tracking >= 0.78)
        and (edge >= 0.72)
        and (float(lesion.profile_signal_z) >= 15.0)
        and (float(lesion.diameter_area_agreement) >= 0.90)
        and (_recovery(lesion) >= 0.55)
    )
    if (
        (raw >= 0.70)
        and np.isfinite(nav)
        and (nav < 0.06)
        and np.isfinite(concord)
        and (concord < 0.30)
        and not severe_short_signature
        and not calcified_rescue_support
        and not strong_direct_lumen_support
    ):
        return (
            "not_assessable",
            "severe_CTA_navigation_mask_identity_conflict",
            "identity_conflict",
        )

    if (
        (base in ("RCA", "LAD", "LCX"))
        and (raw >= 0.50)
        and np.isfinite(rel_hu)
        and (rel_hu < 0.58)
        and (not np.isfinite(nav) or (nav < 0.12))
        and (not np.isfinite(concord) or (concord < 0.55))
        and (calc < 0.10)
        and not severe_short_signature
    ):
        return (
            "not_assessable",
            "low_relative_lumen_HU_with_weak_target_branch_support",
            "identity_conflict",
        )

    if (
        (base in ("RCA", "LAD", "LCX"))
        and (raw < 0.32)
        and np.isfinite(nav)
        and (nav < 0.11)
        and (calc < 0.08)
    ):
        return "reject", "weak_navigation_support_for_low_mild_candidate", "low"
    if (
        (base in ("RCA", "LAD", "LCX"))
        and (raw < 0.45)
        and np.isfinite(nav)
        and (nav < 0.08)
        and (calc < 0.08)
        and np.isfinite(rel_hu)
        and (rel_hu < 0.95)
    ):
        return "reject", "low_navigation_and_low_relative_HU_mild_candidate", "low"

    if (
        (base in ("RCA", "LAD", "LCX"))
        and (raw < 0.45)
        and np.isfinite(nav)
        and (nav < 0.05)
        and (calc < 0.20)
        and (float(lesion.local_prominence) < 0.30)
    ):
        return "reject", "very_low_navigation_support_for_mild_calcified_candidate", "low"

    if (small and (raw < 0.50) and (not np.isfinite(nav) or (nav < 0.10)) and (calc < 0.08)):
        strong_small = (
            (float(lesion.profile_signal_z) >= 10.0)
            and (float(lesion.local_prominence) >= 0.28)
            and (_recovery(lesion) >= 0.45)
            and np.isfinite(rel_hu)
            and (rel_hu >= 0.95)
            and (identity >= 0.88)
            and (tracking >= 0.82)
        )
        if not strong_small:
            return "reject", "small_branch_low_support_mild_candidate", "low"

    branch_length = max(float(lesion.end_distance_mm), float(lesion.peak_distance_mm), 1.0)
    if ((float(lesion.lesion_length_mm) > max(38.0, (0.42 * branch_length))) and (raw < 0.75)):
        return "reject", "diffuse_change_not_single_focal_stenosis", "low"

    reliability = "high"
    if (
        (identity < 0.82)
        or (tracking < 0.75)
        or (overlap < 0.58)
        or (edge < 0.62)
        or (np.isfinite(concord) and (concord < 0.36))
    ):
        reliability = "moderate"
    return "keep", "", reliability

def _apply_severity(
    lesion: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
    rescue_rule: str = "",
) -> None:
    base = base_branch_name(branch)
    raw = evidence_identity_ops._raw_ratio(lesion)
    previous_upper = max(
        raw,
        _float(lesion.stenosis_ratio_upper, raw),
        _float(lesion.stenosis_ratio, raw),
    )
    upper = float(np.clip(previous_upper, raw, 0.99))
    lesion.stenosis_ratio_lower = raw
    lesion.stenosis_ratio_upper = upper
    lesion.stenosis_interval_lower = severity_interval(raw)
    lesion.stenosis_interval_upper = severity_interval(upper)

    nav = _nav(lesion)
    concord = _concord(lesion)
    rel_hu = _float(metrics.get("peak_relative_lumen_hu", np.nan))
    identity = float(metrics.get("identity", 0.0))
    tracking = float(metrics.get("tracking_score", 0.0))
    edge = float(metrics.get("edge", 0.0))
    center_inside = float(metrics.get("center_inside_fraction", 0.0))
    recovery = _recovery(lesion)

    final = raw
    source = "direct_valid_cross_sections"

    if (rescue_rule == "short_focal_severe_main_low_HU"):
        final = float(min(raw, 0.69))
        source = "short_focal_severe_conservative_low_HU_estimate"
    elif (rescue_rule == "orphan_main_low_navigation"):
        final = float(min(raw, 0.65))
        source = "orphan_main_direct_measurement_conservative_cap"
    elif (
        (base in ("RCA", "LAD", "LCX"))
        and (raw >= 0.67)
        and (upper >= 0.70)
        and (identity >= 0.82)
        and (tracking >= 0.72)
        and (edge >= 0.65)
        and (center_inside >= 0.85)
        and (recovery >= 0.30)
        and (
            (np.isfinite(nav) and (nav >= 0.08))
            or (np.isfinite(concord) and (concord >= 0.34))
            or (np.isfinite(rel_hu) and (rel_hu >= 0.75))
            or (float(lesion.calcification_fraction) >= 0.20)
        )
    ):
        final = float(min(upper, (raw + 0.10)))
        source = "direct_with_strong_threshold_crossing_upper_bound"
    elif (
        (base in ("RCA", "LAD", "LCX"))
        and (raw >= 0.50)
        and (upper > (raw + 0.03))
        and (identity >= 0.90)
        and (tracking >= 0.85)
        and np.isfinite(concord)
        and (concord >= 0.55)
        and np.isfinite(nav)
        and (nav >= 0.10)
        and (float(lesion.profile_signal_z) >= 15.0)
        and (float(lesion.diameter_area_agreement) >= 0.85)
        and (recovery >= 0.45)
    ):
        final = float(min(upper, (raw + 0.17)))
        source = "direct_with_multievidence_partial_volume_upper_bound"

    lesion.stenosis_ratio = float(np.clip(final, 0.0, 0.99))
    lesion.stenosis_interval = severity_interval(lesion.stenosis_ratio)
    lesion.formal_ratio_source = source
    lesion.severity_method = source

def _rescue_signature(
    lesion: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
    branch_has_accepted: bool,
    spacing: Sequence[float],
) -> Tuple[str, float]:
    if (str(lesion.assessment_status) != "assessable"):
        return "", 0.0
    raw = evidence_identity_ops._raw_ratio(lesion)
    if (raw < 0.25):
        return "", 0.0
    base = base_branch_name(branch)
    small = (base in ("D1", "D2", "RAMUS", "RAD"))
    length = float(lesion.lesion_length_mm)
    identity = float(metrics.get("identity", 0.0))
    tracking = float(metrics.get("tracking_score", 0.0))
    edge = float(metrics.get("edge", 0.0))
    center_inside = float(metrics.get("center_inside_fraction", 0.0))
    rel_hu = _float(metrics.get("peak_relative_lumen_hu", np.nan))
    concord = _concord(lesion)
    nav = _nav(lesion)
    z = float(lesion.profile_signal_z)
    area = float(lesion.diameter_area_agreement)
    recovery = _recovery(lesion)
    calc = float(lesion.calcification_fraction)
    reference = _float(lesion.reference_diameter_mm, 0.0)
    pos = _float(lesion.position_norm, -1.0)

    if (
        (raw >= 0.55)
        and (pos >= 0.60)
        and (reference > 0)
        and (reference < _resolution_floor_mm(spacing))
        and (not np.isfinite(nav) or (nav < 0.16))
    ):
        return "", 0.0

    if (
        _short_focal_severe_signature(lesion, metrics, branch)
        and (calc < 0.10)
        and np.isfinite(rel_hu)
        and (rel_hu < 0.70)
    ):
        return "short_focal_severe_main_low_HU", ((raw + (0.20 * identity)) + (0.01 * z))

    if (
        (base in ("RCA", "LAD", "LCX"))
        and (raw >= 0.68)
        and (4.0 <= length <= 18.0)
        and (calc >= 0.15)
        and (identity >= 0.70)
        and (edge >= 0.50)
        and (z >= 18.0)
        and (area >= 0.84)
        and (recovery >= 0.45)
        and (
            (
                (not np.isfinite(rel_hu) or (rel_hu >= 0.90))
                and (
                    (np.isfinite(nav) and (nav >= 0.14))
                    or (np.isfinite(concord) and (concord >= 0.35))
                )
            )
            or (
                (calc >= 0.70)
                and (identity >= 0.72)
                and (tracking >= 0.70)
                and (z >= 20.0)
                and (area >= 0.92)
                and (recovery >= 0.55)
            )
        )
    ):
        return "calcified_main_multievidence_rescue", ((raw + (0.18 * identity)) + (0.01 * z))

    if (
        not branch_has_accepted
        and (base in ("RCA", "LAD", "LCX"))
        and (raw >= 0.55)
        and (3.0 <= length <= 18.0)
        and (float(lesion.peak_distance_mm) >= 3.0)
        and (float(lesion.position_norm) >= 0.04)
        and (identity >= 0.80)
        and (tracking >= 0.75)
        and (z >= 10.0)
        and (area >= 0.70)
        and (recovery >= 0.35)
        and np.isfinite(rel_hu)
        and (rel_hu >= 0.75)
    ):
        rule = (
            "orphan_main_low_navigation"
            if (not np.isfinite(nav) or (nav < 0.10))
            else "orphan_main_multievidence_rescue"
        )
        return rule, ((raw + (0.20 * identity)) + (0.01 * z))

    if (
        not branch_has_accepted
        and (base in ("RCA", "LAD", "LCX"))
        and (0.30 <= raw < 0.55)
        and (length <= 18.0)
        and (str(lesion.segment) != "distal")
        and (float(lesion.position_norm) < 0.68)
        and (identity >= 0.90)
        and (tracking >= 0.85)
        and np.isfinite(concord)
        and (concord >= 0.60)
        and (z >= 8.0)
        and (area >= 0.80)
        and (recovery >= 0.30)
        and ((np.isfinite(nav) and (nav >= 0.08)) or (np.isfinite(rel_hu) and (rel_hu >= 0.95)))
    ):
        return "orphan_main_moderate_identity_rescue", ((raw + (0.18 * identity)) + (0.01 * z))

    if (
        (base == "LM")
        and (raw >= 0.25)
        and (length <= 16.0)
        and (identity >= 0.82)
        and (tracking >= 0.75)
        and np.isfinite(concord)
        and (concord >= 0.72)
        and (z >= 8.0)
        and (area >= 0.75)
        and (center_inside >= 0.70)
    ):
        return "LM_short_branch_one_sided_reference_rescue", (
            (raw + (0.20 * identity)) + (0.01 * z)
        )

    if (small and (
        (raw >= 0.55)
        and (2.0 <= length <= 10.0)
        and (identity >= 0.78)
        and (tracking >= 0.75)
        and np.isfinite(rel_hu)
        and (rel_hu >= 1.20)
        and (z >= 12.0)
        and (area >= 0.80)
        and (recovery >= 0.40)
    )):
        return "small_branch_strong_focal_rescue", ((raw + (0.20 * identity)) + (0.01 * z))

    if (small and (
        (raw >= 0.30)
        and (length <= 15.0)
        and (calc >= 0.10)
        and (identity >= 0.90)
        and (tracking >= 0.82)
        and np.isfinite(concord)
        and (concord >= 0.60)
        and (z >= 7.0)
        and (area >= 0.70)
        and (recovery >= 0.25)
    )):
        return "small_branch_calcified_multievidence_rescue", (
            (raw + (0.18 * identity)) + (0.01 * z)
        )

    return "", 0.0

def _strength(lesion: LesionResult) -> float:
    nav = _nav(lesion)
    nav_term = (nav if np.isfinite(nav) else 0.0)
    return float(
        (
            (
                (
                    ((2.0 * float(lesion.stenosis_ratio)) + (0.45 * float(lesion.evidence_score))) + (0.35 * float(lesion.lumen_identity_score))
                ) + (0.18 * nav_term)
            ) + (0.12 * _recovery(lesion))
        )
    )

def _near_existing(candidate: LesionResult, accepted: Sequence[LesionResult]) -> bool:
    for prior in accepted:
        same_segment = (str(prior.segment) == str(candidate.segment))
        close = (abs((float(prior.peak_distance_mm) - float(candidate.peak_distance_mm))) < 10.0)
        if (same_segment and close):
            return True
    return False

def detect_resolution_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    import copy

    base_results, df = evidence_identity_ops.detect_identity_checked_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    if df.empty:
        return base_results, df

    accepted: List[LesionResult] = []
    audit: List[LesionResult] = []
    handled_keys: set = set()

    for original in base_results:
        if (int(original.accepted) != 1):
            continue
        item = copy.deepcopy(original)
        item.verifier_version = "resolution_context_verifier_v2_0_5"
        evidence_identity_ops._relocalize_if_needed(
            item, df, centerline, branch, branch_masks, spacing, branch_roots
        )
        metrics = _metrics(item, df, centerline)
        action, reason, reliability = _hard_decision(item, metrics, branch, spacing)
        item.measurement_reliability = reliability
        item.v2_0_5_rule = (reason or "passed_resolution_context_verifier")
        if (action == "not_assessable"):
            item.accepted = 0
            item.assessment_status = "not_assessable"
            item.not_assessable_reason = reason
            item.identity_conflict_flag = int(("identity" in reason))
            item.identity_conflict_reason = (reason if item.identity_conflict_flag else "")
            item.rejection_reasons = reason
            item.final_decision = "not_assessable_v2_0_5"
        elif (action == "reject"):
            item.accepted = 0
            item.assessment_status = "assessable"
            item.rejection_reasons = reason
            item.final_decision = "rejected_v2_0_5_context_filter"
        else:
            item.accepted = 1
            item.assessment_status = "assessable"
            item.rejection_reasons = ""
            item.not_assessable_reason = ""
            item.identity_conflict_flag = 0
            item.identity_conflict_reason = ""
            _apply_severity(item, metrics, branch)
            item.final_decision = "accepted_v2_0_5"
            accepted.append(item)
        audit.append(item)
        handled_keys.add(_candidate_key(original))

    rescue_candidates: List[Tuple[float, LesionResult]] = []
    branch_has_accepted = bool(accepted)
    for original in base_results:
        if (int(original.accepted) == 1):
            continue
        key = _candidate_key(original)
        if (key in handled_keys):
            continue
        item = copy.deepcopy(original)
        if (str(item.assessment_status) != "assessable"):
            continue
        raw = evidence_identity_ops._raw_ratio(item)
        if (not np.isfinite(raw) or (raw < 0.25)):
            continue
        evidence_identity_ops._relocalize_if_needed(
            item, df, centerline, branch, branch_masks, spacing, branch_roots
        )
        metrics = _metrics(item, df, centerline)
        rule, score = _rescue_signature(item, metrics, branch, branch_has_accepted, spacing)
        if not rule:
            continue
        action, reason, reliability = _hard_decision(item, metrics, branch, spacing)
        if ((action == "not_assessable") and (
            rule
            not in (
                "short_focal_severe_main_low_HU",
                "calcified_main_multievidence_rescue",
            )
        )):
            continue
        if _near_existing(item, accepted):
            continue
        item.accepted = 1
        item.assessment_status = "assessable"
        item.rejection_reasons = ""
        item.not_assessable_reason = ""
        item.repair_attempted = 1
        item.repair_succeeded = 1
        item.repair_method = rule
        item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_5_{rule}")
        item.final_decision = "accepted_after_v2_0_5_context_rescue"
        item.measurement_reliability = (
            "moderate" if (("low_HU" in rule) or ("calcified" in rule)) else "high"
        )
        item.v2_0_5_rule = rule
        _apply_severity(item, metrics, branch, rescue_rule = rule)
        rescue_candidates.append((float(score), item))
        handled_keys.add(key)

    for _, item in sorted(rescue_candidates, key = lambda pair: pair[0], reverse = True):
        if _near_existing(item, accepted):
            item.accepted = 0
            item.rejection_reasons = "alternate_rescue_measurement_same_local_process"
            item.final_decision = "rejected_alternate_v2_0_5_rescue"
            audit.append(item)
        else:
            accepted.append(item)
            audit.append(item)

    accepted.sort(key = lambda x: float(x.peak_distance_mm))
    consolidated: List[LesionResult] = []
    for item in accepted:
        if not consolidated:
            item.consolidation_action = (item.consolidation_action or "independent_v2_0_5")
            consolidated.append(item)
            continue
        prev = consolidated[-1]
        physical_gap = float((item.start_distance_mm - prev.end_distance_mm))
        recovery, plateau_mm, fraction = profiles_ops._recovery_between(prev, item, df, centerline)
        item.recovery_between_adjacent_lesions = recovery
        item.recovery_plateau_mm = plateau_mm
        item.recovery_plateau_fraction = fraction
        sustained_recovery = (
            (physical_gap >= 1.5)
            and np.isfinite(recovery)
            and (recovery >= 0.90)
            and (plateau_mm >= 2.5)
            and (fraction >= 0.70)
        )
        same_local_process = (
            (physical_gap <= 0.8)
            or (
                (abs((float(item.peak_distance_mm) - float(prev.peak_distance_mm))) <= 12.0)
                and not sustained_recovery
            )
            or ((physical_gap <= 7.0) and not sustained_recovery)
        )
        if same_local_process:
            stronger = (item if (_strength(item) > _strength(prev)) else prev)
            weaker = (prev if (stronger is item) else item)
            stronger.warning_flags = profiles_ops._join(
                stronger.warning_flags,
                weaker.warning_flags,
                "v2_0_5_same_unrecovered_process_consolidated",
            )
            stronger.consolidation_action = (
                "kept_stronger_local_peak_same_unrecovered_process_v2_0_5"
            )
            weaker.accepted = 0
            weaker.rejection_reasons = "merged_same_unrecovered_process_v2_0_5"
            weaker.final_decision = "merged_into_stronger_local_peak_v2_0_5"
            audit.append(weaker)
            consolidated[-1] = stronger
        else:
            item.consolidation_action = "separate_after_sustained_recovery_v2_0_5"
            consolidated.append(item)

    topology = dict((centerline.topology or {}))
    branch_length = max(
        (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0),
        1e-6,
    )
    support = float(topology.get("cta_support_fraction", 1.0))
    trim_fraction = (float(topology.get("cta_trimmed_tip_mm", 0.0)) / branch_length)
    incomplete = ((trim_fraction > 0.45) or (
        (support < 0.34) and ("cta_lumen_continuity_low" in str(centerline.quality_flag))
    ))
    if incomplete:
        for item in consolidated:
            item.accepted = 0
            item.assessment_status = "not_assessable_incomplete_branch_coverage"
            item.not_assessable_reason = (
                f"cta_support_fraction={support:.3f};trimmed_tip_fraction={trim_fraction:.3f}"
            )
            item.rejection_reasons = "incomplete_branch_CTA_coverage"
            item.final_decision = "not_assessable_incomplete_branch_coverage_v2_0_5"
            item.measurement_reliability = "not_assessable"
            item.v2_0_5_rule = "incomplete_branch_CTA_coverage"

    final_accepted = [x for x in consolidated if (int(x.accepted) == 1)]
    for i, item in enumerate(final_accepted, start = 1):
        item.lesion_index = int(i)
        item.verifier_version = "resolution_context_verifier_v2_0_5"
        item.final_decision = "accepted"
        if not item.formal_ratio_source:
            item.formal_ratio_source = "direct_valid_cross_sections"
        if not item.v2_0_5_rule:
            item.v2_0_5_rule = "passed_resolution_context_verifier"

    passthrough: List[LesionResult] = []
    existing_audit_keys = {
        (_candidate_key(x), int(x.accepted), str(x.final_decision)) for x in audit
    }
    for original in base_results:
        key = _candidate_key(original)
        if (key in handled_keys):
            continue
        item = copy.deepcopy(original)
        item.verifier_version = "resolution_context_verifier_v2_0_5"
        item.measurement_reliability = (item.measurement_reliability or (
            "not_assessable" if (str(item.assessment_status) != "assessable") else "unverified"
        ))
        item.v2_0_5_rule = (item.v2_0_5_rule or "base_candidate_passthrough")
        item.final_decision = (item.final_decision or (
            "base_not_assessable_passthrough"
            if (str(item.assessment_status) != "assessable")
            else "base_rejected_passthrough"
        ))
        audit_key = (key, int(item.accepted), str(item.final_decision))
        if (audit_key in existing_audit_keys):
            continue
        existing_audit_keys.add(audit_key)
        passthrough.append(item)

    df["stenosis_v2_0_5_profile"] = 1
    rejected_audit = [x for x in audit if (int(x.accepted) != 1)]
    return ((final_accepted + rejected_audit) + passthrough), df

def summarize_resolution_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = evidence_identity_ops.summarize_identity_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    length = max(float(result.centerline_length_mm), 1e-6)
    support = float(result.cta_centerline_support_fraction)
    trim_fraction = (float(result.cta_trimmed_tip_mm) / length)
    quality = str(result.centerline_quality)
    base = base_branch_name(branch)

    adaptive_support_floor = (0.30 if (base in ("LM", "D1", "D2", "RAMUS", "RAD")) else 0.42)
    if ((trim_fraction > 0.45) or ((support < 0.34) and ("cta_lumen_continuity_low" in quality))):
        result.branch_assessability = "not_assessable_incomplete_branch_coverage"
        result.branch_assessability_detail = (
            f"cta_support_fraction={support:.3f};trimmed_tip_fraction={trim_fraction:.3f};"
            f"centerline_quality={quality};v2_0_5"
        )
    elif ((support < adaptive_support_floor) or ("cta_lumen_continuity_low" in quality)):
        result.branch_assessability = "limited_incomplete_CTA_support"
        result.branch_assessability_detail = (
            f"cta_support_fraction={support:.3f};trimmed_tip_fraction={trim_fraction:.3f};"
            f"centerline_quality={quality};v2_0_5"
        )
    else:
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail,
            "v2_0_5_resolution_context_checked",
        )
    return result
