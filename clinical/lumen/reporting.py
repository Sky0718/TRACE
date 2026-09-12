from __future__ import annotations

import copy
import math
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import evidence_identity as evidence_identity_ops
from . import evidence_uncertainty as evidence_uncertainty_ops
from . import profiles as profiles_ops

from .models import (
    ALGORITHM_VERSION,
    BranchResult,
    CenterlineResult,
    DEFAULT_MIN_REPORTABLE_RATIO,
    LesionResult,
    RootAssignment,
    base_branch_name,
)

def _set_rule(item: LesionResult, rule: str) -> None:
    item.v2_0_10_rule = profiles_ops._join(item.v2_0_10_rule, str(rule))
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_10_{rule}")

def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return (out if np.isfinite(out) else float(default))

def _parse_competing(text: str) -> List[Tuple[str, float, float]]:
    parsed: List[Tuple[str, float, float]] = []
    for part in str((text or "")).split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.match(
            r"(?P<segment>[A-Za-z_-]+)\s*:\s*(?P<ratio>[0-9.]+)\s*@\s*(?P<distance>[0-9.]+)\s*mm",
            part,
        )
        if not match:
            continue
        try:
            parsed.append(
                (
                    str(match.group("segment")).lower(),
                    float(match.group("ratio")),
                    float(match.group("distance")),
                )
            )
        except Exception:
            continue
    return parsed

def _segment_span(values: Sequence[str], fallback: str = "unknown") -> str:
    order = ["proximal", "mid", "distal"]
    found: List[str] = []
    for value in values:
        v = str((value or "")).strip().lower()
        if (v == "lm"):
            return "LM"
        for token in re.split(r"[-;/, ]+", v):
            if ((token in order) and (token not in found)):
                found.append(token)
    if not found:
        return str((fallback or "unknown"))
    found.sort(key = order.index)
    if (len(found) == 1):
        return found[0]
    if (found == ["proximal", "mid"]):
        return "proximal-mid"
    if (found == ["mid", "distal"]):
        return "mid-distal"
    return "proximal-distal"

def _area_equivalent_ratio(item: LesionResult) -> float:
    direct = _finite(item.area_stenosis_ratio, float("nan"))
    if np.isfinite(direct):
        return float(np.clip((1.0 - math.sqrt(max((1.0 - direct), 0.0))), 0.0, 0.99))
    minimum = _finite(item.minimum_lumen_area_mm2, float("nan"))
    reference = _finite(item.reference_area_mm2, float("nan"))
    if (np.isfinite(minimum) and np.isfinite(reference) and (reference > 1e-6)):
        return float(np.clip((1.0 - math.sqrt(max((minimum / reference), 0.0))), 0.0, 0.99))
    return float("nan")

def _candidate_by_distance(
    items: Sequence[LesionResult],
    distance_mm: float,
    segment: str,
) -> Optional[LesionResult]:
    candidates = [
        item
        for item in items
        if (
            (int(item.accepted) != 1)
            and (abs((_finite(item.peak_distance_mm, -1e9) - float(distance_mm))) <= 3.5)
        )
    ]
    if not candidates:
        return None
    same_segment = [x for x in candidates if (str(x.peak_segment).lower() == str(segment).lower())]
    pool = (same_segment or candidates)
    return min(
        pool, key = lambda x: abs((_finite(x.peak_distance_mm, -1e9) - float(distance_mm)))
    )

def _hard_review_exclusion(item: LesionResult) -> bool:
    if ((int(item.identity_conflict_flag) == 1) or (int(item.resolution_limit_flag) == 1)):
        return True
    reason = ";".join(
        [
            str((item.not_assessable_reason or "")),
            str((item.rejection_reasons or "")),
        ]
    ).lower()
    hard_tokens = (
        "identity_conflict",
        "unresolved_lumen",
        "subresolution",
        "resolution_limit",
        "centerline_qc_hard_fail",
        "outside_target_lumen",
        "terminal_endpoint_unreliable",
        "root_or_label_seam_measurement_unreliable",
        "topologically_unreliable_main_path",
        "centerline_cta_continuity_failed",
        "weak_reference_section_coverage",
        "unreliable_one_sided_reference",
        "too_short",
    )
    return any((token in reason) for token in hard_tokens)

def _candidate_is_local_to_other_formal(
    candidate: LesionResult,
    current: LesionResult,
    all_items: Sequence[LesionResult],
) -> bool:
    candidate_peak = _finite(candidate.peak_distance_mm, 0.0)
    candidate_start = _finite(candidate.start_distance_mm, candidate_peak)
    candidate_end = _finite(candidate.end_distance_mm, candidate_peak)
    if (candidate_end < candidate_start):
        candidate_start, candidate_end = candidate_end, candidate_start
    for other in all_items:
        if ((other is current) or (int(other.accepted) != 1)):
            continue
        other_peak = _finite(other.peak_distance_mm, 0.0)
        other_start = _finite(other.start_distance_mm, other_peak)
        other_end = _finite(other.end_distance_mm, other_peak)
        if (other_end < other_start):
            other_start, other_end = other_end, other_start
        interval_gap = max((other_start - candidate_end), (candidate_start - other_end), 0.0)
        peak_gap = abs((other_peak - candidate_peak))
        local_limit = float(
            np.clip(((1.20 * max(_finite(other.lesion_length_mm, 6.0), 6.0)) + 3.0), 8.0, 16.0)
        )
        if ((interval_gap <= 3.0) or (peak_gap <= local_limit)):
            return True
    return False

def _report_local_segment_and_separate_distant_candidates(
    item: LesionResult,
    all_items: Sequence[LesionResult],
) -> None:
    if (int(item.accepted) != 1):
        return
    peak_distance = _finite(item.peak_distance_mm, 0.0)
    lesion_length = max(_finite(item.lesion_length_mm, 0.0), 1.0)
    near_limit = float(np.clip(((1.35 * lesion_length) + 3.0), 8.0, 16.0))
    local: List[Tuple[str, float, float]] = []
    distant: List[Tuple[str, float, float]] = []
    for segment, ratio, distance in _parse_competing(item.competing_segment_candidates):
        if (abs((float(distance) - peak_distance)) <= near_limit):
            local.append((segment, ratio, distance))
        else:
            distant.append((segment, ratio, distance))

    own_peak_segment = str((item.peak_segment or item.segment or "unknown"))
    own_span = str((item.segment_span or own_peak_segment))
    local_span = _segment_span(
        ([own_span] + [segment for segment, _, _ in local]),
        fallback = own_span,
    )
    item.local_segment_competitor_count = int(len(local))
    item.local_segment_competitors = ";".join(
        f"{segment}:{ratio:.3f}@{distance:.1f}mm" for segment, ratio, distance in local
    )
    item.distant_competing_candidate_count = int(len(distant))
    item.distant_competing_segments = ";".join(
        f"{segment}:{ratio:.3f}@{distance:.1f}mm" for segment, ratio, distance in distant
    )

    item.reported_segment = own_peak_segment
    item.reported_segment_span = own_span
    item.segment_uncertainty_span = local_span
    item.segment_alternative = ";".join(sorted({segment for segment, _, _ in local}))

    confidence = _finite(item.segment_confidence, 0.0)
    crosses_boundary = (("-" in own_span) and (own_span not in ("", "unknown")))
    local_ambiguity = bool(local)
    low_confidence = (confidence < 0.60)
    item.segment_review_required = int((crosses_boundary or local_ambiguity or low_confidence))
    if local_ambiguity:
        item.reported_segment_status = "local_competitor_review_required"
        item.localization_audit_status = "local_competitor_uncertainty"
    elif low_confidence:
        item.reported_segment_status = "fallback_review_required"
        item.localization_audit_status = "low_confidence_segment_fallback"
    elif crosses_boundary:
        item.reported_segment_status = "lesion_crosses_anatomical_boundary"
        item.localization_audit_status = "lesion_crosses_segment_boundary"
    elif (confidence >= 0.75):
        item.reported_segment_status = "anatomical_landmark_supported"
        item.localization_audit_status = "anatomical_landmark_supported"
    else:
        item.reported_segment_status = "best_estimate"
        item.localization_audit_status = "best_estimate_without_high_confidence_landmark"
    adjusted_confidence = ((confidence - (0.18 if local_ambiguity else 0.0)) - (
        0.10 if crosses_boundary else 0.0
    ))
    item.localization_confidence = float(np.clip(adjusted_confidence, 0.0, 1.0))

    eligible: List[Tuple[str, float, float, LesionResult, float]] = []
    for segment, ratio, distance in distant:
        candidate = _candidate_by_distance(all_items, distance, segment)
        if ((candidate is None) or _hard_review_exclusion(candidate)):
            continue
        if _candidate_is_local_to_other_formal(candidate, item, all_items):
            continue
        formal_items = [other for other in all_items if (int(other.accepted) == 1)]
        if formal_items:
            nearest = min(
                formal_items,
                key = lambda other: abs(
                    (
                        _finite(other.peak_distance_mm, 0.0) - _finite(candidate.peak_distance_mm, 0.0)
                    )
                ),
            )
            if (nearest is not item):
                continue
        raw_ratio = evidence_identity_ops._raw_ratio(candidate)
        evidence = _finite(candidate.evidence_score, 0.0)
        identity = _finite(candidate.lumen_identity_score, 0.0)
        tracking = _finite(candidate.tracking_score_median, 0.0)
        independent = bool(
            (
                (_finite(candidate.cta_mask_concordance, 0.0) >= 0.42)
                or (_finite(candidate.navigation_mask_stenosis_ratio, 0.0) >= 0.10)
                or (_finite(candidate.diameter_area_agreement, 0.0) >= 0.78)
            )
        )
        if (
            (raw_ratio < 0.25)
            or (evidence < 0.75)
            or (identity < 0.72)
            or (tracking < 0.65)
            or not independent
        ):
            continue
        score = ((((0.45 * raw_ratio) + (0.30 * evidence)) + (0.15 * identity)) + (0.10 * tracking))
        eligible.append((segment, raw_ratio, distance, candidate, score))

    selected: List[Tuple[str, float, float, LesionResult, float]] = []
    for entry in sorted(eligible, key = lambda x: (x[0], x[2], -x[4])):
        segment, raw_ratio, distance, candidate, score = entry
        matched_index = None
        for index, previous in enumerate(selected):
            if ((previous[0] == segment) and (abs((previous[2] - distance)) <= 20.0)):
                matched_index = index
                break
        if (matched_index is None):
            selected.append(entry)
        elif (score > selected[matched_index][4]):
            selected[matched_index] = entry

    reviewable_distant: List[Tuple[str, float, float]] = []
    for segment, raw_ratio, distance, candidate, _ in selected:
        candidate.additional_lesion_review_candidate = 1
        candidate.additional_lesion_review_reason = "distant_high_signal_separate_lesion_review"
        candidate.borderline_review_candidate = 1
        candidate.borderline_review_reason = "additional_high_signal_separate_lesion_review"
        candidate.clinical_review_priority = ("high" if (raw_ratio >= 0.70) else "moderate")
        _set_rule(candidate, "additional_high_signal_separate_lesion_review")
        reviewable_distant.append((segment, raw_ratio, distance))

    item.additional_lesion_review_required = int(bool(reviewable_distant))
    if reviewable_distant:
        item.clinical_review_priority = (
            "high" if any((ratio >= 0.70) for _, ratio, _ in reviewable_distant) else "moderate"
        )
        _set_rule(item, "additional_separate_lesion_review_available")
    if local_ambiguity:
        _set_rule(item, "local_segment_competitor_review")
    elif crosses_boundary:
        _set_rule(item, "accepted_lesion_crosses_segment_boundary")
    elif low_confidence:
        _set_rule(item, "segment_fallback_review")
    else:
        _set_rule(item, "localization_audit_pass")

def _suppress_adjacent_mild_shoulder(items: Sequence[LesionResult], branch: str) -> None:
    if (base_branch_name(branch) not in ("RCA", "LAD", "LCX")):
        return
    formal = [item for item in items if (int(item.accepted) == 1)]
    for item in list(formal):
        ratio = float(np.clip(item.stenosis_ratio, 0.0, 0.99))
        if not (0.25 <= ratio < 0.45):
            continue
        severe = [
            other
            for other in formal
            if ((other is not item) and (float(other.stenosis_ratio) >= 0.70))
        ]
        if not severe:
            continue
        nearest = min(
            severe, key = lambda x: abs((float(x.peak_distance_mm) - float(item.peak_distance_mm)))
        )
        peak_gap = abs((float(nearest.peak_distance_mm) - float(item.peak_distance_mm)))
        interval_gap = max(
            (float(nearest.start_distance_mm) - float(item.end_distance_mm)),
            (float(item.start_distance_mm) - float(nearest.end_distance_mm)),
            0.0,
        )
        if ((peak_gap > 18.0) or (interval_gap > 8.0)):
            continue
        area_equivalent = _area_equivalent_ratio(item)
        toward_severe_recovery = (
            _finite(item.distal_recovery, 1.0)
            if (float(nearest.peak_distance_mm) > float(item.peak_distance_mm))
            else _finite(item.proximal_recovery, 1.0)
        )
        if not (
            np.isfinite(area_equivalent)
            and (area_equivalent < 0.25)
            and (_finite(item.calcification_fraction, 0.0) < 0.10)
            and (_finite(item.navigation_mask_stenosis_ratio, 0.0) < 0.15)
            and (toward_severe_recovery < 0.35)
        ):
            continue
        reason = "adjacent_mild_diameter_shoulder_to_severe_lesion_review_only"
        item.accepted = 0
        item.clinical_output_status = "rejected"
        item.assessment_status = "assessable"
        item.rejection_reasons = profiles_ops._join(item.rejection_reasons, reason)
        item.final_decision = "rejected_v2_0_10"
        item.final_state_version = "2.0.10"
        item.verifier_version = "local_segment_additional_review_v2_0_10"
        item.adjacent_severe_shoulder_flag = 1
        item.adjacent_severe_shoulder_reason = reason
        item.area_equivalent_diameter_stenosis_ratio = area_equivalent
        item.borderline_review_candidate = 1
        item.borderline_review_reason = reason
        item.clinical_review_priority = "moderate"
        _set_rule(item, reason)

def detect_and_measure_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = evidence_uncertainty_ops.detect_uncertainty_checked_lesions(
        case_id,
        branch,
        profile,
        centerline,
        branch_masks,
        spacing,
        minimum_reportable_ratio = minimum_reportable_ratio,
        branch_roots = branch_roots,
    )
    items = [copy.deepcopy(item) for item in base_results]
    for item in items:
        item.v2_0_10_rule = ""
    _suppress_adjacent_mild_shoulder(items, branch)
    formal_snapshot = [item for item in items if (int(item.accepted) == 1)]
    for item in formal_snapshot:
        _report_local_segment_and_separate_distant_candidates(item, items)
    for item in items:
        item.final_state_version = "2.0.10"
        item.verifier_version = "local_segment_additional_review_v2_0_10"
        item.status_consistency_checked = 1
        if (int(item.accepted) == 1):
            if str(item.final_decision).endswith("v2_0_9"):
                item.final_decision = str(item.final_decision).replace("v2_0_9", "v2_0_10")
            _set_rule(item, (item.v2_0_10_rule or "passed_v2_0_10_localization_audit"))
        elif not item.v2_0_10_rule:
            item.v2_0_10_rule = "v2_0_10_state_sync"
    formal = sorted(
        [item for item in items if (int(item.accepted) == 1)],
        key = lambda x: _finite(x.peak_distance_mm, 0.0),
    )
    for index, item in enumerate(formal, start = 1):
        item.lesion_index = int(index)
    df["stenosis_v2_0_10_profile"] = 1
    return (formal + [item for item in items if (int(item.accepted) != 1)]), df

def summarize_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = evidence_uncertainty_ops.summarize_uncertainty_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    result.final_state_version = "2.0.10"
    formal = [item for item in lesions if (int(item.accepted) == 1)]
    nonformal = [item for item in lesions if (int(item.accepted) == 0)]
    result.formal_lesion_count = int(len(formal))
    result.lesion_count = int(len(formal))
    result.formal_segment_review_count = int(
        sum((int(item.segment_review_required) == 1) for item in formal)
    )
    result.localization_review_lesion_count = int(
        sum(
            (
                (int(item.segment_review_required) == 1)
                or (int(item.additional_lesion_review_required) == 1)
            )
            for item in formal
        )
    )
    result.additional_high_signal_candidate_count = int(
        sum((int(item.additional_lesion_review_candidate) == 1) for item in nonformal)
    )
    result.additional_lesion_review_count = int(
        sum((int(item.additional_lesion_review_required) == 1) for item in formal)
    )
    result.adjacent_shoulder_review_count = int(
        sum((int(item.adjacent_severe_shoulder_flag) == 1) for item in nonformal)
    )
    result.review_required_lesion_count = int(result.localization_review_lesion_count)
    if formal:
        maximum = max(formal, key = lambda x: float(x.stenosis_ratio))
        result.maximum_stenosis_ratio = float(maximum.stenosis_ratio)
        result.maximum_stenosis_interval = str(maximum.stenosis_interval)
        result.maximum_stenosis_segment = str((maximum.reported_segment or maximum.peak_segment))
        result.maximum_stenosis_segment_span = str(
            (maximum.reported_segment_span or maximum.segment_span)
        )
        result.maximum_stenosis_segment_status = str(maximum.reported_segment_status)
        result.maximum_stenosis_position_norm = float(maximum.peak_position_norm)
    else:
        result.maximum_stenosis_segment_span = "none"
        result.maximum_stenosis_segment_status = ""
    return result
