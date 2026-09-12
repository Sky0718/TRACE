from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from . import branch_context as branch_context_ops
from . import evidence_identity as evidence_identity_ops
from . import evidence_resolution as evidence_resolution_ops
from . import profiles as profiles_ops

from .models import (
    ALGORITHM_VERSION,
    BranchResult,
    CenterlineResult,
    DEFAULT_MIN_REPORTABLE_RATIO,
    LesionResult,
    PARENT_BRANCHES,
    RootAssignment,
    base_branch_name,
    severity_interval,
)

_V206_SEGMENT_CACHE: Dict[Tuple[str, int, Tuple[int, ...], int], Dict[str, Any]] = {}

def _mask_pointer(mask: np.ndarray) -> int:
    try:
        return int(np.asarray(mask).__array_interface__["data"][0])
    except Exception:
        return id(mask)

def _internal_branchpoint_indices(
    branch: str,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
) -> List[int]:
    if ((branch not in branch_masks) or (centerline.path_voxel.shape[0] < 8)):
        return []
    mask = np.asarray(branch_masks[branch]).astype(bool)
    key = (
        str(branch),
        _mask_pointer(mask),
        tuple(int(v) for v in mask.shape),
        int(np.sum(mask)),
    )
    cached = _V206_SEGMENT_CACHE.get(key)
    if (cached is not None):
        return list(cached.get("landmarks", []))
    landmarks: List[int] = []
    try:
        from scipy.ndimage import convolve
        from skimage.morphology import skeletonize

        try:
            skel = skeletonize(mask, method = "lee").astype(bool)
        except Exception:
            skel = skeletonize(mask).astype(bool)
        if not np.any(skel):
            _V206_SEGMENT_CACHE[key] = {"landmarks": []}
            return []
        neighbour_count = (convolve(
            skel.astype(np.uint8),
            np.ones((3, 3, 3), dtype = np.uint8),
            mode = "constant",
            cval = 0,
        ).astype(np.int16) - skel.astype(np.int16))
        coords = np.argwhere((skel & (neighbour_count >= 4)))
        if (coords.shape[0] == 0):
            _V206_SEGMENT_CACHE[key] = {"landmarks": []}
            return []
        sp = np.asarray(spacing, dtype = np.float64)
        tree = cKDTree((centerline.path_voxel.astype(np.float64) * sp[None, :]))
        dist, idx = tree.query((coords.astype(np.float64) * sp[None, :]), k = 1)
        keep = (dist <= max(2.0, (4.0 * float(np.min(sp)))))
        mapped = np.sort(np.unique(idx[keep].astype(int)))
        if mapped.size:
            cum = np.asarray(centerline.cumulative_mm, dtype = float)
            total = (float(cum[-1]) if cum.size else 0.0)
            groups: List[List[int]] = [[int(mapped[0])]]
            for value in mapped[1:]:
                value = int(value)
                prev = groups[-1][-1]
                gap_mm = (
                    float((cum[value] - cum[prev])) if (cum.size > value) else float((value - prev))
                )
                if (gap_mm <= 3.5):
                    groups[-1].append(value)
                else:
                    groups.append([value])
            for group in groups:
                candidate = int(round(float(np.median(group))))
                frac = (
                    float((cum[candidate] / max(total, 1e-6)))
                    if ((total > 0) and (cum.size > candidate))
                    else 0.0
                )
                if (0.10 <= frac <= 0.92):
                    landmarks.append(candidate)
    except Exception:
        landmarks = []
    _V206_SEGMENT_CACHE[key] = {"landmarks": list(landmarks)}
    return landmarks

def _pick_landmark(
    indices: Sequence[int],
    n: int,
    lo: float,
    hi: float,
    target: float,
) -> Optional[int]:
    valid = [int(i) for i in indices if (lo <= (float(i) / max((n - 1), 1)) <= hi)]
    if not valid:
        return None
    return min(valid, key = lambda i: abs(((float(i) / max((n - 1), 1)) - target)))

def _segment_boundaries(
    branch: str,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Dict[str, Any]:
    base = base_branch_name(branch)
    n = int(centerline.path_voxel.shape[0])
    if (n == 0):
        return {"method": "unavailable", "confidence": 0.0, "boundaries": [], "evidence": "none"}
    if (base in ("LM",)):
        return {
            "method": "LM_whole_branch",
            "confidence": 1.0,
            "boundaries": [],
            "evidence": "LM_label",
        }
    if (base == "LAD"):
        info = dict(
            profiles_ops._length_segment_boundaries(branch, centerline, branch_masks, spacing, branch_roots)
        )
        info["evidence"] = (
            "D1_D2_parent_contact"
            if ("D1_D2" in str(info.get("method", "")))
            else (
                "D1_parent_contact_plus_physical_fallback"
                if ("D1" in str(info.get("method", "")))
                else "rooted_length_fallback"
            )
        )
        return info

    landmarks = _internal_branchpoint_indices(branch, centerline, branch_masks, spacing)
    if (base == "RCA"):
        b1 = _pick_landmark(landmarks, n, 0.16, 0.53, 0.34)
        b2 = _pick_landmark(landmarks, n, 0.52, 0.91, 0.73)
        if ((b1 is not None) and (b2 is not None) and (b2 > (b1 + max(3, int((0.08 * n)))))):
            return {
                "method": "RCA_internal_branchpoint_landmarks",
                "confidence": 0.76,
                "boundaries": [int(b1), int(b2)],
                "raw_boundaries": list(landmarks),
                "evidence": "rooted_RCA_skeleton_sidebranches",
            }
        fallback1 = int(round((0.32 * (n - 1))))
        fallback2 = int(round((0.70 * (n - 1))))
        if (b1 is not None):
            fallback1 = int(b1)
        if (b2 is not None):
            fallback2 = int(b2)
        if (fallback2 <= fallback1):
            fallback2 = min((n - 1), (fallback1 + max(3, int((0.28 * n)))))
        return {
            "method": "RCA_branchpoint_hybrid_rooted_length",
            "confidence": (0.62 if landmarks else 0.50),
            "boundaries": [fallback1, fallback2],
            "raw_boundaries": list(landmarks),
            "evidence": (
                "partial_branchpoints_plus_rooted_length" if landmarks else "rooted_length_only"
            ),
        }

    if (base == "LCX"):
        b1 = _pick_landmark(landmarks, n, 0.14, 0.58, 0.36)
        b2 = _pick_landmark(landmarks, n, 0.50, 0.91, 0.74)
        if ((b1 is not None) and (b2 is not None) and (b2 > (b1 + max(3, int((0.08 * n)))))):
            return {
                "method": "LCX_internal_OM_like_branchpoint_landmarks",
                "confidence": 0.72,
                "boundaries": [int(b1), int(b2)],
                "raw_boundaries": list(landmarks),
                "evidence": "rooted_LCX_skeleton_sidebranches",
            }
        fallback1 = int(round((0.35 * (n - 1))))
        fallback2 = int(round((0.72 * (n - 1))))
        if (b1 is not None):
            fallback1 = int(b1)
        if (b2 is not None):
            fallback2 = int(b2)
        if (fallback2 <= fallback1):
            fallback2 = min((n - 1), (fallback1 + max(3, int((0.28 * n)))))
        return {
            "method": "LCX_branchpoint_hybrid_rooted_length",
            "confidence": (0.60 if landmarks else 0.48),
            "boundaries": [fallback1, fallback2],
            "raw_boundaries": list(landmarks),
            "evidence": (
                "partial_branchpoints_plus_rooted_length" if landmarks else "rooted_length_only"
            ),
        }

    if (base in ("D1", "D2", "RAMUS", "RAD")):
        return {
            "method": "short_branch_rooted_length_adaptive",
            "confidence": 0.60,
            "boundaries": [int(round((0.42 * (n - 1)))), int(round((0.76 * (n - 1))))],
            "evidence": "short_branch_root_and_physical_length",
        }

    info = dict(profiles_ops._length_segment_boundaries(branch, centerline, branch_masks, spacing, branch_roots))
    info["evidence"] = "legacy_fallback"
    return info

def _parent_reference(
    item: LesionResult,
    branch: str,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]],
) -> Tuple[float, float, str]:
    base = base_branch_name(branch)
    if (base not in ("D1", "D2", "RAMUS", "RAD")):
        return float("nan"), float("nan"), ""
    if (float(item.peak_position_norm) > 0.48):
        return float("nan"), float("nan"), ""
    root = (branch_roots.get(branch) if (branch_roots is not None) else None)
    parent_key: Optional[str] = None
    if ((root is not None) and str(root.parent_branch)):
        parent_key = str(root.parent_branch)
    if (parent_key not in branch_masks):
        parents = PARENT_BRANCHES.get(base, ())
        parent_key = next((k for k in branch_masks if (base_branch_name(k) in parents)), None)
    if ((parent_key is None) or (parent_key not in branch_masks)):
        return float("nan"), float("nan"), ""
    parent = np.asarray(branch_masks[parent_key]).astype(bool)
    if not np.any(parent):
        return float("nan"), float("nan"), ""
    if (root is not None):
        root_vox = np.rint(np.asarray(root.voxel, dtype = float)).astype(int)
    else:
        root_vox = np.rint(
            np.array([item.start_voxel_x, item.start_voxel_y, item.start_voxel_z], dtype = float)
        ).astype(int)
    root_vox = np.clip(root_vox, 0, (np.asarray(parent.shape) - 1))
    sp = np.asarray(spacing, dtype = float)
    pad = np.maximum(np.ceil((6.0 / np.maximum(sp, 1e-6))).astype(int), 4)
    lo = np.maximum((root_vox - pad), 0)
    hi = np.minimum(((root_vox + pad) + 1), np.asarray(parent.shape))
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    crop = parent[sl]
    if not np.any(crop):
        return float("nan"), float("nan"), ""
    edt = distance_transform_edt(crop, sampling = tuple(float(v) for v in spacing))
    local = (root_vox - lo)
    radius_voxels = np.maximum(np.ceil((1.5 / np.maximum(sp, 1e-6))).astype(int), 1)
    llo = np.maximum((local - radius_voxels), 0)
    lhi = np.minimum(((local + radius_voxels) + 1), np.asarray(crop.shape))
    neighbourhood = edt[tuple(slice(int(llo[d]), int(lhi[d])) for d in range(3))]
    values = neighbourhood[(neighbourhood > 0)]
    if (values.size == 0):
        values = edt[(edt > 0)]
    if (values.size == 0):
        return float("nan"), float("nan"), ""
    parent_diameter = float((2.0 * np.percentile(values, 75.0)))
    daughter_factor = (0.64 if (base in ("D1", "D2")) else 0.70)
    expected_child = float((parent_diameter * daughter_factor))
    current_ref = evidence_resolution_ops._float(item.reference_diameter_mm, 0.0)
    if (current_ref > 0):
        expected_child = float(np.clip(expected_child, current_ref, (2.25 * current_ref)))
    min_lumen = evidence_resolution_ops._float(item.minimum_lumen_diameter_mm, float("nan"))
    if (not np.isfinite(min_lumen) or (expected_child <= 1e-6)):
        return expected_child, float("nan"), str(parent_key)
    ratio = float(np.clip((1.0 - (min_lumen / expected_child)), 0.0, 0.97))
    return expected_child, ratio, str(parent_key)

def _strong_short_branch_parent_reference(
    item: LesionResult,
    parent_ratio: float,
) -> bool:
    nav = evidence_resolution_ops._float(item.navigation_mask_stenosis_ratio)
    calc = evidence_resolution_ops._float(item.calcification_fraction, 0.0)
    area_ratio = evidence_resolution_ops._float(item.area_stenosis_ratio, 0.0)
    return bool(
        (
            np.isfinite(parent_ratio)
            and (parent_ratio >= 0.40)
            and (float(item.peak_position_norm) <= 0.48)
            and (float(item.lumen_identity_score) >= 0.90)
            and (float(item.tracking_score_median) >= 0.72)
            and (float(item.tracking_supported_fraction) >= 0.70)
            and (float(item.center_inside_fraction) >= 0.85)
            and (float(item.reference_valid_fraction) >= 0.58)
            and (float(item.profile_signal_z) >= 2.5)
            and ((np.isfinite(nav) and (nav >= 0.15)) or (area_ratio >= 0.20) or (calc >= 0.10))
        )
    )

def _accept(
    item: LesionResult,
    rule: str,
    *,
    reliability: str = "moderate",
    coverage_scope: str = "whole_branch",
) -> None:
    item.accepted = 1
    item.rejection_reasons = ""
    item.not_assessable_reason = ""
    item.measurement_reliability = str(reliability)
    item.coverage_scope = str(coverage_scope)
    item.assessment_status = (
        "locally_assessable_in_incomplete_branch"
        if (coverage_scope == "local_only")
        else "assessable"
    )
    item.clinical_output_status = (
        "formal_limited_branch_coverage" if (coverage_scope == "local_only") else "formal"
    )
    item.final_decision = (
        "accepted_local_measurement_v2_0_6"
        if (coverage_scope == "local_only")
        else "accepted_v2_0_6"
    )
    item.clinical_review_reason = (
        "local lesion and reference sections are valid; whole-branch coverage is incomplete"
        if (coverage_scope == "local_only")
        else ""
    )
    item.verifier_version = "state_synced_anatomy_multilesion_v2_0_6"
    item.v2_0_5_rule = str(rule)
    item.final_state_version = "2.0.6"
    item.status_consistency_checked = 1
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_6_{rule}")

def _reject(item: LesionResult, reason: str, not_assessable: bool = False) -> None:
    item.accepted = 0
    item.coverage_scope = ("insufficient" if not_assessable else "whole_branch")
    item.clinical_output_status = ("not_assessable" if not_assessable else "rejected")
    item.assessment_status = ("not_assessable" if not_assessable else "assessable")
    item.final_decision = ("not_assessable_v2_0_6" if not_assessable else "rejected_v2_0_6")
    item.clinical_review_reason = str(reason)
    item.rejection_reasons = profiles_ops._join(item.rejection_reasons, str(reason))
    if not_assessable:
        item.not_assessable_reason = profiles_ops._join(item.not_assessable_reason, str(reason))
    item.measurement_reliability = ("not_assessable" if not_assessable else "low")
    item.verifier_version = "state_synced_anatomy_multilesion_v2_0_6"
    item.final_state_version = "2.0.6"
    item.status_consistency_checked = 1
    item.warning_flags = profiles_ops._join(item.warning_flags, f"v2_0_6_{reason}")

def _sync_final_state(item: LesionResult) -> None:
    local = bool(
        (
            (int(item.local_coverage_override) == 1)
            or (str(item.assessment_status) == "locally_assessable_in_incomplete_branch")
            or (str(item.clinical_output_status) == "formal_limited_branch_coverage")
        )
    )
    if (int(item.accepted) == 1):
        _accept(
            item,
            str((item.v2_0_5_rule or "passed_v2_0_6_final_state")),
            reliability = str((item.measurement_reliability or "moderate")),
            coverage_scope = ("local_only" if local else "whole_branch"),
        )
        if local:
            item.local_coverage_override = 1
    else:
        not_assessable = bool(
            (
                str(item.assessment_status).startswith("not_assessable")
                or (str(item.clinical_output_status) == "not_assessable")
                or bool(str(item.not_assessable_reason).strip())
            )
        )
        reason = (
            str(item.clinical_review_reason).strip()
            or str(item.not_assessable_reason).strip()
            or str(item.rejection_reasons).strip()
            or ("not_assessable_candidate" if not_assessable else "rejected_candidate")
        )
        _reject(item, reason, not_assessable = not_assessable)

def _independent_secondary_candidate(
    item: LesionResult,
    accepted: Sequence[LesionResult],
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> Tuple[bool, float, float, float]:
    if not accepted:
        return False, 0.0, 0.0, 0.0
    best_score = 0.0
    best_plateau = 0.0
    best_fraction = 0.0
    independent = False
    for existing in accepted:
        distance = abs((float(item.peak_distance_mm) - float(existing.peak_distance_mm)))
        if (distance < 8.0):
            continue
        left, right = (
            (item, existing)
            if (item.peak_distance_mm < existing.peak_distance_mm)
            else (existing, item)
        )
        recovery, plateau_mm, fraction = profiles_ops._recovery_between(left, right, df, centerline)
        if not np.isfinite(recovery):
            continue
        score = float(np.clip(((recovery - 0.78) / 0.22), 0.0, 1.0))
        score *= float(np.clip((plateau_mm / 3.0), 0.0, 1.0))
        score *= float(np.clip((fraction / 0.75), 0.0, 1.0))
        if (score > best_score):
            best_score, best_plateau, best_fraction = score, plateau_mm, fraction
        if ((recovery >= 0.88) and (plateau_mm >= 2.0) and (fraction >= 0.65)):
            independent = True
    return independent, best_score, best_plateau, best_fraction

def _can_rescue_main_secondary(item: LesionResult) -> bool:
    raw = evidence_identity_ops._raw_ratio(item)
    nav = evidence_resolution_ops._float(item.navigation_mask_stenosis_ratio)
    calc = evidence_resolution_ops._float(item.calcification_fraction, 0.0)
    rel_hu = evidence_resolution_ops._float(item.peak_relative_lumen_hu)
    area_ratio = evidence_resolution_ops._float(item.area_stenosis_ratio, 0.0)
    reasons = set(profiles_ops._tokens(item.rejection_reasons))
    recoverable_reasons = {
        "unstable_lumen_center_tracking",
        "insufficient_sustained_valid_lumen_tracking",
        "insufficient_continuous_valid_peak_sections",
        "terminal_taper_zone",
    }
    prohibited_reasons = {
        "proximal_echo_after_verified_root_severe_lesion",
        "proximal_echo_after_verified_root_lesion",
        "diameter_area_disagreement",
        "weakly_supported_distal_mild_candidate",
        "too_few_identity_supported_narrowed_sections",
    }
    if ((reasons & prohibited_reasons) or not (reasons & recoverable_reasons)):
        return False
    if (("terminal_taper_zone" in reasons) and (raw < 0.40)):
        return False
    if (("terminal_taper_zone" not in reasons) and (raw < 0.40)):
        return False
    independent_signal = bool(
        (
            (np.isfinite(nav) and (nav >= 0.15))
            or (calc >= 0.12)
            or (np.isfinite(rel_hu) and (rel_hu >= 0.92))
            or (area_ratio >= 0.40)
        )
    )
    return bool(
        (
            np.isfinite(raw)
            and (int(item.resolution_limit_flag) == 0)
            and (int(item.identity_conflict_flag) == 0)
            and (float(item.lumen_identity_score) >= 0.89)
            and (float(item.tracking_score_median) >= 0.77)
            and (float(item.tracking_supported_fraction) >= 0.72)
            and (float(item.center_inside_fraction) >= 0.90)
            and (float(item.reference_valid_fraction) >= 0.68)
            and (float(item.peak_valid_run_mm) >= 3.0)
            and (float(item.profile_signal_z) >= 4.5)
            and (float(item.diameter_area_agreement) >= 0.72)
            and (float(item.local_prominence) >= 0.10)
            and independent_signal
        )
    )

def detect_topology_checked_lesions(
    case_id: str,
    branch: str,
    profile: pd.DataFrame,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    minimum_reportable_ratio: float = DEFAULT_MIN_REPORTABLE_RATIO,
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Tuple[List[LesionResult], pd.DataFrame]:
    base_results, df = branch_context_ops.detect_context_checked_lesions(
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

    segment_info = _segment_boundaries(branch, centerline, branch_masks, spacing, branch_roots)
    for item in items:
        profiles_ops._segment_info(item, branch, centerline, branch_masks, spacing, branch_roots)
        item.anatomical_landmark_evidence = str(segment_info.get("evidence", "unknown"))

    accepted = [x for x in items if (int(x.accepted) == 1)]

    if (base in ("D1", "D2", "RAMUS", "RAD")):
        for item in sorted(items, key = lambda x: float(x.peak_distance_mm)):
            parent_ref, parent_ratio, parent_key = _parent_reference(
                item, branch, branch_masks, spacing, branch_roots
            )
            item.parent_reference_diameter_mm = float(parent_ref)
            item.parent_reference_stenosis_ratio = float(parent_ratio)
            if not _strong_short_branch_parent_reference(item, parent_ratio):
                continue
            item.short_branch_measurement_rule = f"parent_contact_reference_from_{parent_key}"
            item.parent_reference_used = 1
            direct = evidence_identity_ops._raw_ratio(item)
            item.stenosis_ratio_lower = float(direct)
            item.stenosis_ratio_upper = float(max(direct, parent_ratio))
            item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
            item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
            item.stenosis_ratio = float(parent_ratio)
            item.stenosis_interval = severity_interval(item.stenosis_ratio)
            item.reference_diameter_mm = float(parent_ref)
            item.formal_ratio_source = "parent_contact_expected_daughter_reference"
            item.severity_method = "short_branch_parent_contact_reference"
            if ((int(item.accepted) != 1) and (parent_ratio >= 0.40)):
                if not evidence_resolution_ops._near_existing(item, accepted):
                    _accept(
                        item,
                        "short_branch_parent_contact_reference_rescue",
                        reliability = "moderate",
                    )
                    accepted.append(item)
            elif (int(item.accepted) == 1):
                item.warning_flags = profiles_ops._join(
                    item.warning_flags,
                    "v2_0_6_short_branch_parent_reference_used",
                )

    if (base in ("RCA", "LAD", "LCX")):
        for item in sorted(items, key = lambda x: float(x.peak_distance_mm)):
            if (int(item.accepted) == 1):
                continue
            if (str(item.assessment_status) != "assessable"):
                continue
            if not _can_rescue_main_secondary(item):
                continue
            independent, score, plateau_mm, fraction = _independent_secondary_candidate(
                item, accepted, df, centerline
            )
            item.multilesion_independence_score = float(score)
            item.multilesion_recovery_plateau_mm = float(plateau_mm)
            if not independent:
                continue
            item.recovery_plateau_fraction = max(
                evidence_resolution_ops._float(item.recovery_plateau_fraction, 0.0), float(fraction)
            )
            direct = evidence_identity_ops._raw_ratio(item)
            item.stenosis_ratio = float(direct)
            item.stenosis_ratio_lower = float(direct)
            item.stenosis_ratio_upper = float(
                max(direct, evidence_resolution_ops._float(item.stenosis_ratio_upper, direct))
            )
            item.stenosis_interval = severity_interval(item.stenosis_ratio)
            item.stenosis_interval_lower = severity_interval(item.stenosis_ratio_lower)
            item.stenosis_interval_upper = severity_interval(item.stenosis_ratio_upper)
            item.formal_ratio_source = "direct_valid_cross_sections_secondary_lesion"
            _accept(
                item,
                "recovery_proven_independent_secondary_lesion",
                reliability = "moderate",
            )
            accepted.append(item)

    for item in items:
        _sync_final_state(item)
        item.final_state_version = "2.0.6"
        item.status_consistency_checked = 1

    final_accepted = sorted(
        [x for x in items if (int(x.accepted) == 1)],
        key = lambda x: float(x.peak_distance_mm),
    )
    for i, item in enumerate(final_accepted, start = 1):
        item.lesion_index = int(i)

    df["stenosis_v2_0_6_profile"] = 1
    df["segment_method_v2_0_6"] = str(segment_info.get("method", "unknown"))
    df["segment_confidence_v2_0_6"] = float(segment_info.get("confidence", 0.0))
    return (final_accepted + [x for x in items if (int(x.accepted) != 1)]), df

def summarize_topology_checked_branch(
    case_id: str,
    branch: str,
    label_value: int,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
) -> BranchResult:
    result = branch_context_ops.summarize_context_checked_branch(case_id, branch, label_value, centerline, profile, lesions)
    result.algorithm_version = ALGORITHM_VERSION
    formal = [x for x in lesions if (int(x.accepted) == 1)]
    local = [x for x in formal if (str(x.coverage_scope) == "local_only")]
    result.formal_lesion_count = int(len(formal))
    result.local_only_lesion_count = int(len(local))
    result.final_state_version = "2.0.6"
    if local:
        result.coverage_scope = "local_only"
        result.whole_branch_negative_interpretation_valid = 0
        result.branch_assessability = (
            "limited_incomplete_branch_coverage_with_locally_assessable_lesion"
        )
        result.branch_assessability_detail = profiles_ops._join(
            result.branch_assessability_detail,
            "v2_0_6_local_measurement_formally_exported",
        )
    elif (str(result.branch_assessability) == "assessable"):
        result.coverage_scope = "whole_branch"
        result.whole_branch_negative_interpretation_valid = 1
    else:
        result.coverage_scope = "insufficient"
        result.whole_branch_negative_interpretation_valid = 0
    return result
