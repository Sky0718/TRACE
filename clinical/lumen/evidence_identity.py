from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import detection_stability as detection_stability_ops
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

def _raw_ratio(lesion: LesionResult) -> float:
    value = (
        float(lesion.stenosis_ratio_uncorrected)
        if np.isfinite(lesion.stenosis_ratio_uncorrected)
        else float(lesion.stenosis_ratio)
    )
    return float(np.clip(value, 0.0, 0.99))

def _apply_severity(lesion: LesionResult, metrics: Mapping[str, float], branch: str) -> None:
    base = base_branch_name(branch)
    raw = _raw_ratio(lesion)
    prior = float(np.clip(lesion.stenosis_ratio, 0.0, 0.99))
    upper = float(max(raw, prior))
    lesion.stenosis_ratio_lower = raw
    lesion.stenosis_ratio_upper = upper
    lesion.stenosis_interval_lower = severity_interval(raw)
    lesion.stenosis_interval_upper = severity_interval(upper)

    if (base in ("D1", "D2", "RAMUS", "RAD")):
        final = raw
        lesion.severity_method = "direct_valid_cross_sections_small_branch"
    elif (upper > (raw + 0.015)):
        final = upper
        lesion.severity_method = "direct_measurement_with_partial_volume_uncertainty"
    else:
        final = raw
        lesion.severity_method = "direct_valid_cross_sections"
    lesion.stenosis_ratio = float(np.clip(final, 0.0, 0.99))
    lesion.stenosis_interval = severity_interval(lesion.stenosis_ratio)

def _identity_conflict(
    lesion: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
) -> Tuple[bool, str]:
    base = base_branch_name(branch)
    small = (base in ("D1", "D2", "RAMUS", "RAD"))
    raw = _raw_ratio(lesion)
    nav = (
        float(lesion.navigation_mask_stenosis_ratio)
        if np.isfinite(lesion.navigation_mask_stenosis_ratio)
        else float("nan")
    )
    concord = (
        float(lesion.cta_mask_concordance)
        if np.isfinite(lesion.cta_mask_concordance)
        else float("nan")
    )
    rel_hu = float(metrics.get("peak_relative_lumen_hu", np.nan))
    radial = float(metrics.get("peak_radial_cv", np.nan))
    eccentricity = float(metrics.get("peak_eccentricity", np.nan))
    shape_bad = ((np.isfinite(radial) and (radial > 0.62)) or (
        np.isfinite(eccentricity) and (eccentricity > 4.8)
    ))
    low_hu = (np.isfinite(rel_hu) and (rel_hu < 0.58))

    if (
        (raw >= 0.70)
        and np.isfinite(nav)
        and (nav < 0.075)
        and np.isfinite(concord)
        and (concord < 0.31)
        and (
            low_hu
            or shape_bad
            or (float(metrics.get("tracking_overlap", 0.0)) < 0.60)
            or (float(metrics.get("edge", 0.0)) < 0.65)
        )
    ):
        return True, "severe_CTA_navigation_identity_conflict"

    if (
        small
        and (raw >= 0.65)
        and np.isfinite(concord)
        and (concord < 0.46)
        and (
            (np.isfinite(nav) and (nav < 0.16))
            or shape_bad
            or low_hu
            or (float(metrics.get("edge", 0.0)) < 0.62)
        )
    ):
        return True, "small_branch_target_lumen_identity_conflict"

    if (
        (raw >= 0.55)
        and np.isfinite(nav)
        and (nav < 0.07)
        and np.isfinite(concord)
        and (concord < 0.36)
        and (
            low_hu
            or (float(metrics.get("tracking_overlap", 0.0)) < 0.58)
            or (float(metrics.get("center_shift", 99.0)) > 1.10)
            or (float(metrics.get("edge", 0.0)) < 0.58)
        )
    ):
        return True, "moderate_CTA_navigation_identity_conflict"

    if (low_hu and (raw >= 0.68) and (float(metrics.get("edge", 0.0)) < 0.70)):
        return True, "low_relative_lumen_HU_identity_conflict"
    return False, ""

def _mild_false_positive(
    lesion: LesionResult, metrics: Mapping[str, float], branch: str
) -> str:
    raw = _raw_ratio(lesion)
    nav = (
        float(lesion.navigation_mask_stenosis_ratio)
        if np.isfinite(lesion.navigation_mask_stenosis_ratio)
        else float("nan")
    )
    concord = (
        float(lesion.cta_mask_concordance)
        if np.isfinite(lesion.cta_mask_concordance)
        else float("nan")
    )
    if (
        (raw < 0.50)
        and np.isfinite(nav)
        and (nav < 0.075)
        and np.isfinite(concord)
        and (concord < 0.48)
        and (float(lesion.local_prominence) < 0.16)
    ):
        return "weak_navigation_and_CTA_mask_support_for_mild_candidate"
    if (
        (raw < 0.36)
        and (float(lesion.local_prominence) < 0.10)
        and (max(float(lesion.proximal_recovery), float(lesion.distal_recovery)) < 0.22)
    ):
        return "low_prominence_low_recovery_mild_candidate"
    return ""

def _can_rescue(
    original: LesionResult,
    metrics: Mapping[str, float],
    branch: str,
) -> bool:
    base = base_branch_name(branch)
    if (base not in ("RCA", "LAD", "LCX")):
        return False
    if ((int(original.pre_verifier_accepted) != 1) or (int(original.accepted) == 1)):
        return False
    raw = _raw_ratio(original)
    if ((raw < 0.65) or (float(original.lesion_length_mm) > 25.0)):
        return False
    reasons = set(profiles_ops._tokens(original.rejection_reasons))
    allowed = {
        "unstable_lumen_center_tracking",
        "insufficient_sustained_valid_lumen_tracking",
        "root_or_label_seam_measurement_unreliable",
    }
    if (not reasons or not reasons.issubset(allowed)):
        return False
    if (
        (float(metrics.get("identity", 0.0)) < 0.82)
        or (float(metrics.get("edge", 0.0)) < 0.58)
        or (float(original.profile_signal_z) < 6.0)
        or (float(original.diameter_area_agreement) < 0.72)
        or (float(metrics.get("center_inside_fraction", 0.0)) < 0.88)
        or (float(metrics.get("tracking_score", 0.0)) < 0.72)
        or (float(metrics.get("tracking_overlap", 0.0)) < 0.58)
        or (float(metrics.get("center_shift", 99.0)) > 1.40)
        or (max(float(original.proximal_recovery), float(original.distal_recovery)) < 0.25)
    ):
        return False
    nav = (
        float(original.navigation_mask_stenosis_ratio)
        if np.isfinite(original.navigation_mask_stenosis_ratio)
        else float("nan")
    )
    concord = (
        float(original.cta_mask_concordance)
        if np.isfinite(original.cta_mask_concordance)
        else float("nan")
    )
    rel_hu = float(metrics.get("peak_relative_lumen_hu", np.nan))
    has_independent_support = (
        (np.isfinite(nav) and (nav >= 0.05))
        or (np.isfinite(concord) and (concord >= 0.30))
        or (np.isfinite(rel_hu) and (rel_hu >= 0.75))
    )
    return bool(has_independent_support)

def _relocalize_if_needed(
    lesion: LesionResult,
    df: pd.DataFrame,
    centerline: CenterlineResult,
    branch: str,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]],
) -> None:
    branch_length = (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0)
    warning = str(lesion.warning_flags)
    if (("v2_0_2_consolidated" in warning) or (
        float(lesion.lesion_length_mm) > max(30.0, (0.30 * branch_length))
    )):
        profiles_ops._peak_centered_boundaries(
            lesion, df, centerline, branch, branch_masks, spacing, branch_roots
        )
        lesion.consolidation_action = "relocalized_from_overbroad_v2_0_2_proposal"

def _same_segment_competitor(
    candidate: LesionResult,
    accepted: Sequence[LesionResult],
) -> bool:
    for prior in accepted:
        same_segment = (str(prior.segment) == str(candidate.segment))
        close = (abs((float(prior.peak_distance_mm) - float(candidate.peak_distance_mm))) < 25.0)
        if (same_segment and close):
            return True
    return False

def detect_identity_checked_lesions(
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

    base_results, df = detection_stability_ops.detect_stable_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    if (df.empty or not base_results):
        return base_results, df

    accepted: List[LesionResult] = []
    audit: List[LesionResult] = []

    for original in base_results:
        item = copy.deepcopy(original)
        item.verifier_version = "identity_conflict_repair_recovery_v2_0_5"
        item.final_decision = ""
        if ((int(original.accepted) != 1) or (str(original.assessment_status) != "assessable")):
            continue
        _relocalize_if_needed(
            item, df, centerline, branch, branch_masks, spacing, branch_roots
        )
        metrics = profiles_ops._candidate_metrics(item, df, centerline)
        item.peak_relative_lumen_hu = float(metrics["peak_relative_lumen_hu"])
        item.peak_radial_cv = float(metrics["peak_radial_cv"])
        item.peak_eccentricity = float(metrics["peak_eccentricity"])
        conflict, reason = _identity_conflict(item, metrics, branch)
        mild_reason = _mild_false_positive(item, metrics, branch)
        branch_length = (
            float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0
        )
        diffuse = ((float(item.lesion_length_mm) > max(36.0, (0.34 * branch_length))) and (
            _raw_ratio(item) < 0.75
        ))
        if conflict:
            item.accepted = 0
            item.assessment_status = "not_assessable_identity_conflict"
            item.not_assessable_reason = reason
            item.identity_conflict_flag = 1
            item.identity_conflict_reason = reason
            item.rejection_reasons = reason
            item.final_decision = "not_assessable_identity_conflict"
        elif mild_reason:
            item.accepted = 0
            item.rejection_reasons = mild_reason
            item.final_decision = "rejected_low_support_mild_candidate"
        elif diffuse:
            item.accepted = 0
            item.rejection_reasons = "diffuse_change_not_focal_stenosis"
            item.final_decision = "rejected_diffuse_change"
        else:
            item.accepted = 1
            item.assessment_status = "assessable"
            item.rejection_reasons = ""
            item.identity_conflict_flag = 0
            item.final_decision = "accepted_v2_0_2_verified"
            _apply_severity(item, metrics, branch)
            accepted.append(item)
        audit.append(item)

    rescue_pool: List[LesionResult] = []
    for original in base_results:
        if (int(original.accepted) == 1):
            continue
        item = copy.deepcopy(original)
        item.verifier_version = "identity_conflict_repair_recovery_v2_0_5"
        if (str(item.assessment_status) != "assessable"):
            continue
        metrics = profiles_ops._candidate_metrics(item, df, centerline)
        if not _can_rescue(item, metrics, branch):
            continue
        conflict, _ = _identity_conflict(item, metrics, branch)
        if conflict:
            continue
        profiles_ops._peak_centered_boundaries(
            item, df, centerline, branch, branch_masks, spacing, branch_roots
        )
        metrics = profiles_ops._candidate_metrics(item, df, centerline)
        if _same_segment_competitor(item, accepted):
            continue
        item.accepted = 1
        item.assessment_status = "assessable"
        item.rejection_reasons = ""
        item.repair_attempted = 1
        item.repair_succeeded = 1
        item.repair_method = "narrow_strong_main_candidate_repair_v2_0_5"
        item.warning_flags = profiles_ops._join(item.warning_flags, "v2_0_5_repaired_candidate")
        item.final_decision = "accepted_after_narrow_repair"
        item.peak_relative_lumen_hu = float(metrics["peak_relative_lumen_hu"])
        item.peak_radial_cv = float(metrics["peak_radial_cv"])
        item.peak_eccentricity = float(metrics["peak_eccentricity"])
        _apply_severity(item, metrics, branch)
        rescue_pool.append(item)

    for segment in sorted({str(x.segment) for x in rescue_pool}):
        group = [x for x in rescue_pool if (str(x.segment) == segment)]
        group.sort(key = lambda x: (float(x.stenosis_ratio), float(x.evidence_score)), reverse = True)
        if group:
            accepted.append(group[0])
            audit.extend(group)
            for weaker in group[1:]:
                weaker.accepted = 0
                weaker.rejection_reasons = "alternate_repair_measurement_same_segment"
                weaker.final_decision = "rejected_alternate_repair_measurement"

    accepted.sort(key = lambda x: float(x.peak_distance_mm))
    consolidated: List[LesionResult] = []
    for item in accepted:
        if not consolidated:
            item.consolidation_action = (item.consolidation_action or "independent_proposal")
            consolidated.append(item)
            continue
        prev = consolidated[-1]
        physical_gap = float((item.start_distance_mm - prev.end_distance_mm))
        recovery, plateau_mm, fraction = profiles_ops._recovery_between(prev, item, df, centerline)
        item.recovery_between_adjacent_lesions = recovery
        item.recovery_plateau_mm = plateau_mm
        item.recovery_plateau_fraction = fraction
        sustained_recovery = (
            (physical_gap >= 1.2)
            and np.isfinite(recovery)
            and (recovery >= 0.88)
            and (plateau_mm >= 1.8)
            and (fraction >= 0.60)
        )
        same_process = ((physical_gap <= 0.6) or ((physical_gap <= 5.0) and not sustained_recovery))
        if same_process:
            stronger = (
                item
                if (
                    (float(item.stenosis_ratio), float(item.evidence_score))
                    > (float(prev.stenosis_ratio), float(prev.evidence_score))
                )
                else prev
            )
            weaker = (prev if (stronger is item) else item)
            stronger.warning_flags = profiles_ops._join(
                stronger.warning_flags,
                weaker.warning_flags,
                "v2_0_5_same_unrecovered_process_consolidated",
            )
            stronger.consolidation_action = "kept_stronger_peak_same_unrecovered_process"
            weaker.accepted = 0
            weaker.rejection_reasons = "merged_into_same_unrecovered_process"
            weaker.final_decision = "merged_into_stronger_peak"
            consolidated[-1] = stronger
        else:
            item.consolidation_action = "separate_after_sustained_recovery"
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
            item.final_decision = "not_assessable_incomplete_branch_coverage"

    final_accepted = [x for x in consolidated if (int(x.accepted) == 1)]
    for i, item in enumerate(final_accepted, start = 1):
        item.lesion_index = int(i)
        item.final_decision = "accepted"
        item.verifier_version = "identity_conflict_repair_recovery_v2_0_5"

    known = {(int(x.peak_index), round(float(x.peak_distance_mm), 2)) for x in audit}
    passthrough: List[LesionResult] = []
    for original in base_results:
        key = (int(original.peak_index), round(float(original.peak_distance_mm), 2))
        if (key in known):
            continue
        item = copy.deepcopy(original)
        item.verifier_version = "identity_conflict_repair_recovery_v2_0_5"
        item.final_decision = (
            "base_not_assessable_passthrough"
            if (str(item.assessment_status) != "assessable")
            else "base_rejected_passthrough"
        )
        passthrough.append(item)

    df["stenosis_v2_0_5_profile"] = 1
    rejected_audit = [x for x in audit if (int(x.accepted) != 1)]
    return ((final_accepted + rejected_audit) + passthrough), df

def summarize_identity_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = detection_stability_ops.summarize_stable_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    length = max(float(result.centerline_length_mm), 1e-6)
    support = float(result.cta_centerline_support_fraction)
    trim_fraction = (float(result.cta_trimmed_tip_mm) / length)
    quality = str(result.centerline_quality)
    base = base_branch_name(branch)

    if ((trim_fraction > 0.45) or ((support < 0.34) and ("cta_lumen_continuity_low" in quality))):
        result.branch_assessability = "not_assessable_incomplete_branch_coverage"
        result.branch_assessability_detail = (
            f"cta_support_fraction={support:.3f};trimmed_tip_fraction={trim_fraction:.3f};"
            f"centerline_quality={quality}"
        )
    elif ((support < (0.32 if (base in ("LM", "D1", "D2", "RAMUS", "RAD")) else 0.45)) or (
        "cta_lumen_continuity_low" in quality
    )):
        result.branch_assessability = "limited_incomplete_CTA_support"
        result.branch_assessability_detail = (
            f"cta_support_fraction={support:.3f};trimmed_tip_fraction={trim_fraction:.3f};"
            f"centerline_quality={quality}"
        )
    else:
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail, "v2_0_5_coverage_checked"
        )
    return result
