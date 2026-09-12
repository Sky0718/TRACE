from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.signal import find_peaks

from . import centerline as centerline_ops
from . import topology as topology_ops

from .models import (
    CenterlineResult,
    LesionResult,
    RootAssignment,
    base_branch_name,
)

def _interpolate_small_gaps_physical(
    values: np.ndarray,
    valid: np.ndarray,
    cum: np.ndarray,
    maximum_gap_mm: float = 2.2,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype = float).copy()
    valid = ((np.asarray(valid).astype(bool) & np.isfinite(arr)) & (arr > 0))
    interpolated = np.zeros(arr.size, dtype = bool)
    if (valid.sum() < 2):
        return arr.astype(np.float32), interpolated
    invalid = ~valid
    labeled, count = ndi_label(invalid.astype(np.uint8))
    x = np.arange(arr.size)
    full_interp = np.interp(x, x[valid], arr[valid])
    for cid in range(1, (int(count) + 1)):
        idx = np.where((labeled == cid))[0]
        if ((idx.size == 0) or (idx[0] == 0) or (idx[-1] == (arr.size - 1))):
            continue
        gap_mm = (
            float((cum[(idx[-1] + 1)] - cum[(idx[0] - 1)]))
            if (cum.size > (idx[-1] + 1))
            else float("inf")
        )
        if (gap_mm <= float(maximum_gap_mm)):
            arr[idx] = full_interp[idx]
            interpolated[idx] = True
    return arr.astype(np.float32), interpolated

def _window_indices(cum: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(((cum >= float(lo)) & (cum <= float(hi))))[0]

def _robust_reference_with_validity(
    values: np.ndarray,
    quality: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    percentile: float = 75.0,
    minimum_count: int = 4,
    minimum_fraction: float = 0.45,
) -> Tuple[float, int, float]:
    idx = np.asarray(indices, dtype = int)
    if (idx.size == 0):
        return float("nan"), 0, 0.0
    vals = np.asarray(values, dtype = float)
    q = np.asarray(quality, dtype = float)
    v = np.asarray(valid).astype(bool)
    keep = idx[
        (
            (((v[idx] & np.isfinite(vals[idx])) & (vals[idx] > 0)) & np.isfinite(q[idx])) & (q[idx] >= 0.52)
        )
    ]
    fraction = float((keep.size / max(idx.size, 1)))
    if ((keep.size < int(minimum_count)) or (fraction < float(minimum_fraction))):
        return float("nan"), int(keep.size), fraction
    return float(np.percentile(vals[keep], percentile)), int(keep.size), fraction

def _length_segment_boundaries(
    branch: str,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]] = None,
) -> Dict[str, Any]:
    base = base_branch_name(branch)
    n = centerline.path_voxel.shape[0]
    if (n == 0):
        return {"method": "unavailable", "confidence": 0.0, "boundaries": []}
    if (base == "LM"):
        return {"method": "LM_whole_branch", "confidence": 1.0, "boundaries": []}
    if (base == "LAD"):
        landmarks: Dict[str, int] = {}
        for child in ("D1", "D2"):
            key = next((k for k in branch_masks if (base_branch_name(k) == child)), None)
            if (key is None):
                continue
            root = (branch_roots.get(key) if (branch_roots is not None) else None)
            if (root is None):
                root = centerline_ops.parent_contact_root(key, branch_masks[key], branch_masks, spacing)
            if (root is None):
                continue
            root_phys = (root.voxel * np.asarray(spacing, dtype = np.float32))
            landmarks[child] = int(
                np.argmin(
                    np.linalg.norm((centerline.path_physical_mm - root_phys[None, :]), axis = 1)
                )
            )
        d1, d2 = landmarks.get("D1"), landmarks.get("D2")
        if ((d1 is not None) and (0 < d1 < (n - 1))):
            raw_d1 = int(d1)
            d1_lo = int(round((0.18 * (n - 1))))
            d1_hi = int(round((0.45 * (n - 1))))
            d1_use = int(np.clip(raw_d1, d1_lo, d1_hi))
            if ((d2 is not None) and ((raw_d1 + max(2, int((0.03 * n)))) < int(d2) < (n - 1))):
                raw_d2 = int(d2)
                d2_lo = max((d1_use + max(3, int((0.16 * n)))), int(round((0.55 * (n - 1)))))
                d2_hi = int(round((0.84 * (n - 1))))
                d2_use = int(np.clip(raw_d2, d2_lo, max(d2_lo, d2_hi)))
                adjusted = ((d1_use != raw_d1) or (d2_use != raw_d2))
                return {
                    "method": ("LAD_D1_D2_hybrid_landmarks" if adjusted else "LAD_D1_D2_landmarks"),
                    "confidence": (0.82 if adjusted else 0.95),
                    "boundaries": [int(d1_use), int(d2_use)],
                    "raw_boundaries": [int(raw_d1), int(raw_d2)],
                }
            fallback_d2 = max(
                (d1_use + max(3, int((0.16 * n)))),
                int(round((0.62 * (n - 1)))),
            )
            fallback_d2 = int(min(fallback_d2, int(round((0.84 * (n - 1))))))
            return {
                "method": "LAD_D1_hybrid_plus_length_fallback",
                "confidence": 0.68,
                "boundaries": [int(d1_use), int(fallback_d2)],
                "raw_boundaries": [int(raw_d1)],
            }
    return {
        "method": "normalized_length_fallback",
        "confidence": 0.45,
        "boundaries": [int(round(((n - 1) / 3.0))), int(round(((2.0 * (n - 1)) / 3.0)))],
    }

def _segment_for_index(index: int, branch: str, info: Mapping[str, Any]) -> str:
    base = base_branch_name(branch)
    if (base == "LM"):
        return "LM"
    boundaries = list(info.get("boundaries", []))
    if (len(boundaries) >= 2):
        return (
            "proximal"
            if (index < boundaries[0])
            else ("mid" if (index < boundaries[1]) else "distal")
        )
    return "unknown"

def _segment_span(start_segment: str, peak_segment: str, end_segment: str) -> str:
    order = {"proximal": 0, "mid": 1, "distal": 2}
    segments = [s for s in (start_segment, peak_segment, end_segment) if (s in order)]
    if not segments:
        return (peak_segment if peak_segment else "unknown")
    lo = min(order[s] for s in segments)
    hi = max(order[s] for s in segments)
    inv = {0: "proximal", 1: "mid", 2: "distal"}
    if (lo == hi):
        return inv[lo]
    return f"{inv[lo]}-{inv[hi]}"

def _runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask).astype(bool)
    runs: List[Tuple[int, int]] = []
    i = 0
    while (i < m.size):
        if not m[i]:
            i += 1
            continue
        j = i
        while (((j + 1) < m.size) and m[(j + 1)]):
            j += 1
        runs.append((i, j))
        i = (j + 1)
    return runs

def _close_small_1d_gaps(mask: np.ndarray, cum: np.ndarray, maximum_gap_mm: float) -> np.ndarray:
    m = np.asarray(mask).astype(bool).copy()
    invalid_runs = _runs_from_mask(~m)
    for start, end in invalid_runs:
        if ((start == 0) or (end == (m.size - 1))):
            continue
        gap = (float((cum[(end + 1)] - cum[(start - 1)])) if (cum.size > (end + 1)) else float("inf"))
        if (gap <= float(maximum_gap_mm)):
            m[start : (end + 1)] = True
    return m

def _split_recovery_intervals(
    start: int,
    end: int,
    signal: np.ndarray,
    d: np.ndarray,
    dref: np.ndarray,
    cum: np.ndarray,
    step_mm: float,
) -> List[Tuple[int, int]]:
    if ((end - start) < 4):
        return [(start, end)]
    sub = np.asarray(signal[start : (end + 1)], dtype = float)
    peaks, props = find_peaks(
        sub,
        prominence = max(0.055, (float(np.nanmax(sub)) * 0.12)),
        distance = max(1, int(round((3.0 / max(step_mm, 0.2))))),
    )
    if (peaks.size <= 1):
        return [(start, end)]
    peaks = (peaks + start)
    split_points: List[int] = []
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        valley_rel = int(np.argmin(signal[left_peak : (right_peak + 1)]))
        valley = (left_peak + valley_rel)
        valley_signal = float(signal[valley])
        peak_floor = min(float(signal[left_peak]), float(signal[right_peak]))
        recovery = (
            float((d[valley] / max(dref[valley], 1e-6)))
            if (np.isfinite(d[valley]) and np.isfinite(dref[valley]))
            else 0.0
        )
        around = _window_indices(cum, float((cum[valley] - 0.8)), float((cum[valley] + 0.8)))
        sustained = (
            (
                np.sum(((d[around] / np.maximum(dref[around], 1e-6)) >= 0.82))
                >= max(2, int(round((1.0 / max(step_mm, 0.2)))))
            )
            if around.size
            else False
        )
        if (((valley_signal <= (0.42 * max(peak_floor, 1e-6))) or (recovery >= 0.86)) and sustained):
            split_points.append(valley)
    if not split_points:
        return [(start, end)]
    intervals: List[Tuple[int, int]] = []
    s = start
    for point in split_points:
        intervals.append((s, point))
        s = (point + 1)
    intervals.append((s, end))
    return [(s, e) for s, e in intervals if (e >= s)]

def _valid_run_around_peak(
    valid: np.ndarray, peak: int, cum: np.ndarray
) -> Tuple[int, float, np.ndarray]:
    if ((peak < 0) or (peak >= valid.size) or not valid[peak]):
        return 0, 0.0, np.array([], dtype = int)
    a = b = int(peak)
    while (((a - 1) >= 0) and valid[(a - 1)]):
        a -= 1
    while (((b + 1) < valid.size) and valid[(b + 1)]):
        b += 1
    idx = np.arange(a, (b + 1), dtype = int)
    span = (float((cum[b] - cum[a])) if (idx.size >= 2) else 0.0)
    return int(idx.size), span, idx

def _lesion_coordinates(centerline: CenterlineResult, index: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = int(np.clip(index, 0, (centerline.path_voxel.shape[0] - 1)))
    return centerline.path_voxel[idx].astype(float), centerline.path_physical_mm[idx].astype(float)

def _tokens(value: Any) -> List[str]:
    return [x for x in str((value or "")).split(";") if (x and (x.lower() != "nan"))]

def _join(*values: Any) -> str:
    out: List[str] = []
    for value in values:
        for token in _tokens(value):
            if (token not in out):
                out.append(token)
    return ";".join(out)

def _finite_median(
    values: np.ndarray, indices: np.ndarray, default: float = float("nan")
) -> float:
    if (indices.size == 0):
        return float(default)
    arr = np.asarray(values, dtype = float)[indices]
    arr = arr[np.isfinite(arr)]
    return (float(np.median(arr)) if arr.size else float(default))

def _longest_plateau(mask: np.ndarray, cum: np.ndarray) -> float:
    idx = np.where(np.asarray(mask).astype(bool))[0]
    if (idx.size == 0):
        return 0.0
    best = 0.0
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if (cur != (prev + 1)):
            best = max(best, (float((cum[prev] - cum[start])) if (prev > start) else 0.0))
            start = cur
        prev = cur
    best = max(best, (float((cum[prev] - cum[start])) if (prev > start) else 0.0))
    return float(best)

def _segment_info(
    lesion: LesionResult,
    branch: str,
    centerline: CenterlineResult,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]],
) -> None:
    total = max(
        (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0), 1e-6
    )
    lesion.start_position_norm = float((lesion.start_distance_mm / total))
    lesion.peak_position_norm = float((lesion.peak_distance_mm / total))
    lesion.end_position_norm = float((lesion.end_distance_mm / total))
    lesion.position_norm = lesion.peak_position_norm
    info = topology_ops._segment_boundaries(branch, centerline, branch_masks, spacing, branch_roots)
    lesion.start_segment = _segment_for_index(int(lesion.start_index), branch, info)
    lesion.peak_segment = _segment_for_index(int(lesion.peak_index), branch, info)
    lesion.end_segment = _segment_for_index(int(lesion.end_index), branch, info)
    lesion.segment = lesion.peak_segment
    lesion.segment_span = _segment_span(
        lesion.start_segment, lesion.peak_segment, lesion.end_segment
    )
    lesion.segment_method = str(info.get("method", lesion.segment_method))
    lesion.segment_confidence = float(info.get("confidence", lesion.segment_confidence))
    for prefix, index in (
        ("start", lesion.start_index),
        ("peak", lesion.peak_index),
        ("end", lesion.end_index),
    ):
        vox, phys = _lesion_coordinates(centerline, int(index))
        setattr(lesion, f"{prefix}_voxel_x", float(vox[0]))
        setattr(lesion, f"{prefix}_voxel_y", float(vox[1]))
        setattr(lesion, f"{prefix}_voxel_z", float(vox[2]))
        setattr(lesion, f"{prefix}_x_mm", float(phys[0]))
        setattr(lesion, f"{prefix}_y_mm", float(phys[1]))
        setattr(lesion, f"{prefix}_z_mm", float(phys[2]))

def _peak_centered_boundaries(
    lesion: LesionResult,
    df: pd.DataFrame,
    centerline: CenterlineResult,
    branch: str,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    branch_roots: Optional[Mapping[str, RootAssignment]],
) -> None:
    n = len(df)
    if (n == 0):
        return
    valid = (pd.to_numeric(df.get("valid", 0), errors = "coerce").fillna(0).to_numpy(int) > 0)
    ratio = pd.to_numeric(df.get("diameter_stenosis_ratio", np.nan), errors = "coerce").to_numpy(
        float
    )
    cum = np.asarray(centerline.cumulative_mm, dtype = float)
    peak0 = int(np.clip(int(lesion.peak_index), 0, (n - 1)))
    search_mm = max(5.0, min(18.0, (0.65 * max(float(lesion.lesion_length_mm), 4.0))))
    search = np.where((np.abs((cum - cum[peak0])) <= search_mm))[0]
    trusted_search = (
        search[(valid[search] & np.isfinite(ratio[search]))]
        if search.size
        else np.array([], dtype = int)
    )
    if trusted_search.size:
        peak = int(trusted_search[np.argmax(ratio[trusted_search])])
    else:
        peak = peak0
    peak_ratio = (float(ratio[peak]) if np.isfinite(ratio[peak]) else float(lesion.stenosis_ratio))
    threshold = max(0.14, (0.44 * max(peak_ratio, 0.0)))
    allowed_gap_mm = 0.9
    left = right = peak
    gap = 0.0
    i = (peak - 1)
    while (i >= 0):
        step = float((cum[(i + 1)] - cum[i]))
        if (valid[i] and np.isfinite(ratio[i]) and (ratio[i] >= threshold)):
            left = i
            gap = 0.0
        else:
            gap += max(step, 0.0)
            if (gap > allowed_gap_mm):
                break
        i -= 1
    gap = 0.0
    i = (peak + 1)
    while (i < n):
        step = float((cum[i] - cum[(i - 1)]))
        if (valid[i] and np.isfinite(ratio[i]) and (ratio[i] >= threshold)):
            right = i
            gap = 0.0
        else:
            gap += max(step, 0.0)
            if (gap > allowed_gap_mm):
                break
        i += 1
    lesion.start_index = int(left)
    lesion.peak_index = int(peak)
    lesion.end_index = int(right)
    lesion.start_distance_mm = float(cum[left])
    lesion.peak_distance_mm = float(cum[peak])
    lesion.end_distance_mm = float(cum[right])
    lesion.lesion_length_mm = float(max((cum[right] - cum[left]), 0.0))
    _segment_info(lesion, branch, centerline, branch_masks, spacing, branch_roots)

def _candidate_metrics(
    lesion: LesionResult,
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> Dict[str, float]:
    n = len(df)
    start = max(0, int(lesion.start_index))
    end = min((n - 1), int(lesion.end_index))
    peak = int(np.clip(int(lesion.peak_index), 0, (n - 1)))
    idx = (np.arange(start, (end + 1), dtype = int) if (end >= start) else np.array([], dtype = int))
    valid = (pd.to_numeric(df.get("valid", 0), errors = "coerce").fillna(0).to_numpy(int) > 0)
    identity = (
        pd.to_numeric(df.get("lumen_identity_score", 0.0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    edge = (
        pd.to_numeric(df.get("gradient_boundary_support", 0.0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    overlap = pd.to_numeric(df.get("tracking_overlap", np.nan), errors = "coerce").to_numpy(float)
    score = pd.to_numeric(df.get("tracking_score", 0.0), errors = "coerce").fillna(0).to_numpy(float)
    inside = (
        pd.to_numeric(df.get("center_inside_component", 0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    shift = pd.to_numeric(df.get("center_shift_mm", np.nan), errors = "coerce").to_numpy(float)
    radial = pd.to_numeric(df.get("radial_cv", np.nan), errors = "coerce").to_numpy(float)
    ecc = pd.to_numeric(df.get("eccentricity", np.nan), errors = "coerce").to_numpy(float)
    lumen_hu = pd.to_numeric(df.get("lumen_mean_hu", np.nan), errors = "coerce").to_numpy(float)
    good = (idx[valid[idx]] if idx.size else np.array([], dtype = int))
    critical_ratio = (
        pd.to_numeric(df.get("diameter_stenosis_ratio", 0.0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    threshold = max(
        0.22,
        (
            0.62 * max(
                float(
                    (
                        lesion.stenosis_ratio_uncorrected
                        if np.isfinite(lesion.stenosis_ratio_uncorrected)
                        else lesion.stenosis_ratio
                    )
                ),
                0.0,
            )
        ),
    )
    critical = (good[(critical_ratio[good] >= threshold)] if good.size else np.array([], dtype = int))
    if (critical.size < 2):
        critical = good
    outside = (valid & (
        (np.asarray(centerline.cumulative_mm) < (lesion.start_distance_mm - 2.0)) | (np.asarray(centerline.cumulative_mm) > (lesion.end_distance_mm + 2.0))
    ))
    normal_hu = lumen_hu[(outside & np.isfinite(lumen_hu))]
    if (normal_hu.size < 5):
        normal_hu = lumen_hu[(valid & np.isfinite(lumen_hu))]
    normal_hu_med = (float(np.median(normal_hu)) if normal_hu.size else float("nan"))
    peak_hu = (float(lumen_hu[peak]) if np.isfinite(lumen_hu[peak]) else float("nan"))
    relative_hu = (
        float((peak_hu / normal_hu_med))
        if (np.isfinite(peak_hu) and np.isfinite(normal_hu_med) and (abs(normal_hu_med) > 1e-6))
        else float("nan")
    )
    return {
        "identity": _finite_median(identity, critical, 0.0),
        "edge": _finite_median(edge, critical, 0.0),
        "tracking_overlap": _finite_median(overlap, critical, 0.0),
        "tracking_score": _finite_median(score, critical, 0.0),
        "center_inside_fraction": (
            float(np.mean((inside[critical] >= 0.5))) if critical.size else 0.0
        ),
        "center_shift": _finite_median(shift, critical, 99.0),
        "radial_cv": _finite_median(radial, critical, 1.0),
        "peak_radial_cv": (float(radial[peak]) if np.isfinite(radial[peak]) else float("nan")),
        "peak_eccentricity": (float(ecc[peak]) if np.isfinite(ecc[peak]) else float("nan")),
        "peak_relative_lumen_hu": relative_hu,
        "critical_count": float(critical.size),
        "valid_count": float(good.size),
    }

def _recovery_between(
    left: LesionResult,
    right: LesionResult,
    df: pd.DataFrame,
    centerline: CenterlineResult,
) -> Tuple[float, float, float]:
    start = (int(left.end_index) + 1)
    end = (int(right.start_index) - 1)
    if ((start > end) or (start < 0) or (end >= len(df))):
        return float("nan"), 0.0, 0.0
    idx = np.arange(start, (end + 1), dtype = int)
    valid = (pd.to_numeric(df.get("valid", 0), errors = "coerce").fillna(0).to_numpy(int) > 0)
    ds = pd.to_numeric(
        df.get("smoothed_min_diameter_mm", df.get("min_diameter_mm")), errors = "coerce"
    ).to_numpy(float)
    ref_d = pd.to_numeric(df.get("reference_diameter_mm", np.nan), errors = "coerce").to_numpy(float)
    identity = (
        pd.to_numeric(df.get("lumen_identity_score", 0), errors = "coerce").fillna(0).to_numpy(float)
    )
    edge = (
        pd.to_numeric(df.get("gradient_boundary_support", 0), errors = "coerce")
        .fillna(0)
        .to_numpy(float)
    )
    caliber = (ds / np.maximum(ref_d, 1e-6))
    trusted = (((valid & (identity >= 0.50)) & (edge >= 0.25)) & np.isfinite(caliber))
    gap_trusted = idx[trusted[idx]]
    if (gap_trusted.size == 0):
        return float("nan"), 0.0, 0.0
    recovery = float(np.median(caliber[gap_trusted]))
    recovered = np.zeros(len(df), dtype = bool)
    recovered[gap_trusted] = (caliber[gap_trusted] >= 0.86)
    plateau_mm = _longest_plateau(recovered[idx], np.asarray(centerline.cumulative_mm)[idx])
    fraction = float(np.mean((caliber[gap_trusted] >= 0.86)))
    return recovery, plateau_mm, fraction
