from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter1d
import math
from scipy.ndimage import maximum_filter1d
import numpy as np
from scipy.ndimage import percentile_filter
from skimage.measure import label as skimage_label
from scipy.ndimage import uniform_filter1d
from . import runtime, models, graph

def connected_components_3d(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    (labeled, num) = skimage_label(
        mask.astype(bool), background = 0, return_num = True, connectivity = 3
    )
    return (labeled.astype(np.int32, copy = False), int(num))

def remove_small_components_by_volume(
    mask: np.ndarray, voxel_vol: float, min_vol: float
) -> np.ndarray:
    (labeled, num) = connected_components_3d(mask)
    if (num == 0):
        return np.zeros_like(mask, dtype = bool)
    sizes = np.bincount(labeled.ravel())[1:]
    keep = (np.where(((sizes * float(voxel_vol)) >= float(min_vol)))[0] + 1)
    return np.isin(labeled, keep)

def count_lesions_3d(mask: np.ndarray) -> int:
    (_, n) = connected_components_3d(mask.astype(bool))
    return int(n)

def compute_agatston_score(
    hu: np.ndarray, calc_mask: np.ndarray, spacing: Tuple[float, float, float]
) -> float:
    pixel_area_cm2 = ((float(spacing[0]) * float(spacing[1])) / 100.0)
    score = 0.0
    for z in range(hu.shape[2]):
        slc = calc_mask[:, :, z]
        if not np.any(slc):
            continue
        (labeled, num) = skimage_label(
            slc.astype(bool), background = 0, return_num = True, connectivity = 2
        )
        for cid in range(1, (num + 1)):
            lesion = (labeled == cid)
            if not np.any(lesion):
                continue
            max_hu = float(np.max(hu[:, :, z][lesion]))
            if (max_hu >= 400):
                w = 4
            elif (max_hu >= 300):
                w = 3
            elif (max_hu >= 200):
                w = 2
            else:
                w = 1
            score += ((float(np.sum(lesion)) * pixel_area_cm2) * float(w))
    return float(score)

def agatston_grade(score: float) -> Tuple[str, str]:
    s = float(score)
    if (s == 0):
        return ("none", "very_low_risk")
    if (s < 100):
        return ("mild", "low_risk")
    if (s < 400):
        return ("moderate", "intermediate_risk")
    return ("severe", "high_risk")

def path_arclength_mm(path: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    if (path.shape[0] == 0):
        return np.array([], dtype = np.float32)
    phys = (path.astype(np.float32) * np.asarray(spacing, dtype = np.float32)[None, :])
    if (phys.shape[0] == 1):
        return np.array([0.0], dtype = np.float32)
    step = np.sqrt(np.sum((np.diff(phys, axis = 0) ** 2), axis = 1))
    return np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)

def segment_name_from_norm(pos: float) -> str:
    if (pos < 0.0):
        return "none"
    if (pos < 0.33):
        return "proximal"
    if (pos < 0.66):
        return "mid"
    return "distal"

def odd_window_from_mm(cum: np.ndarray, target_mm: float, minimum: int = 3) -> int:
    if (cum.size < 2):
        w = max(int(minimum), 3)
    else:
        step = float(np.median(np.diff(cum)))
        step = max(step, 0.25)
        w = max(int(minimum), int(round((float(target_mm) / step))))
    if ((w % 2) == 0):
        w += 1
    return max(3, int(w))

def combined_stenosis_profile(
    diameter: np.ndarray,
    reference: np.ndarray,
    cum: np.ndarray,
    area: Optional[np.ndarray] = None,
    area_reference: Optional[np.ndarray] = None,
    calcium_penalty: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    d = np.asarray(diameter, dtype = np.float32)
    ref = np.asarray(reference, dtype = np.float32)
    raw = np.clip((1.0 - (d / np.maximum(ref, 1e-06))), 0.0, 1.0).astype(np.float32)
    n = int(raw.size)
    if (n == 0):
        return {
            "raw": raw,
            "smooth": raw,
            "support": raw,
            "contrast": raw,
            "area_ratio": raw,
            "calcium_penalty": raw,
            "combined": raw,
        }
    smooth = gaussian_filter1d(raw, sigma = max(0.75, (n / 190.0)), mode = "nearest").astype(
        np.float32
    )
    short_win = odd_window_from_mm(cum, target_mm = 2.5, minimum = 3)
    mid_win = odd_window_from_mm(cum, target_mm = 5.0, minimum = max(5, (short_win + 2)))
    long_win = odd_window_from_mm(cum, target_mm = 10.0, minimum = max(7, (mid_win + 2)))
    short_support = uniform_filter1d(smooth, size = max(3, short_win), mode = "nearest").astype(
        np.float32
    )
    mid_support = uniform_filter1d(smooth, size = max(3, mid_win), mode = "nearest").astype(
        np.float32
    )
    long_support = uniform_filter1d(smooth, size = max(3, long_win), mode = "nearest").astype(
        np.float32
    )
    flank_win = odd_window_from_mm(cum, target_mm = 10.0, minimum = max(5, (mid_win + 2)))
    local_high = maximum_filter1d(d, size = max(3, flank_win), mode = "nearest").astype(
        np.float32
    )
    contrast = np.clip((1.0 - (d / np.maximum(local_high, 1e-06))), 0.0, 1.0).astype(np.float32)
    contrast = gaussian_filter1d(
        contrast, sigma = max(0.6, (n / 230.0)), mode = "nearest"
    ).astype(np.float32)
    persistent = np.minimum.reduce([smooth, short_support, mid_support])
    broad_persistent = np.minimum(smooth, long_support)
    diameter_evidence = (
        (((0.38 * smooth)
        + (0.28 * persistent))
        + (0.2 * broad_persistent))
        + (0.14 * np.minimum(smooth, contrast))
    ).astype(np.float32)
    area_ratio = np.zeros_like(diameter_evidence, dtype = np.float32)
    if ((area is not None) and (area_reference is not None)):
        a = np.asarray(area, dtype = np.float32)
        ar = np.asarray(area_reference, dtype = np.float32)
        if ((((a.size == n) and (ar.size == n)) and np.any((a > 0))) and np.any((ar > 0))):
            area_raw = np.clip((1.0 - (a / np.maximum(ar, 1e-06))), 0.0, 1.0).astype(np.float32)
            area_smooth = gaussian_filter1d(
                area_raw, sigma = max(0.8, (n / 190.0)), mode = "nearest"
            ).astype(np.float32)
            area_mid = uniform_filter1d(
                area_smooth, size = max(3, mid_win), mode = "nearest"
            ).astype(np.float32)
            area_long = uniform_filter1d(
                area_smooth, size = max(3, long_win), mode = "nearest"
            ).astype(np.float32)
            area_ratio = (
                ((0.5 * area_smooth)
                + (0.32 * np.minimum(area_smooth, area_mid)))
                + (0.18 * np.minimum(area_smooth, area_long))
            ).astype(np.float32)
            combined = ((0.68 * diameter_evidence) + (0.32 * np.minimum(
                np.maximum(diameter_evidence, area_ratio),
                ((0.5 * diameter_evidence) + (0.5 * area_ratio)),
            )))
        else:
            combined = diameter_evidence
    else:
        combined = diameter_evidence
    calc_pen = np.zeros_like(combined, dtype = np.float32)
    if (calcium_penalty is not None):
        cp = np.asarray(calcium_penalty, dtype = np.float32)
        if (cp.size == n):
            calc_pen = np.clip(cp, 0.0, 1.0)
            suppress = ((0.34 * calc_pen) * np.clip((1.0 - (combined / 0.72)), 0.0, 1.0))
            combined = (combined * (1.0 - suppress))
    combined = gaussian_filter1d(
        np.clip(combined, 0.0, 1.0), sigma = max(0.45, (n / 260.0)), mode = "nearest"
    )
    combined = np.clip(combined, 0.0, 1.0).astype(np.float32)
    return {
        "raw": raw,
        "smooth": smooth,
        "support": mid_support,
        "contrast": contrast,
        "area_ratio": area_ratio.astype(np.float32),
        "calcium_penalty": calc_pen.astype(np.float32),
        "combined": combined,
    }

def cross_section_area_profile(
    branch_mask: np.ndarray,
    path: np.ndarray,
    spacing: Tuple[float, float, float],
    diameter_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    if (path.shape[0] < 3):
        return np.zeros(path.shape[0], dtype = np.float32)
    mask = np.asarray(branch_mask).astype(bool)
    sp = np.asarray(spacing, dtype = np.float32)
    phys = (path.astype(np.float32) * sp[None, :])
    n = int(path.shape[0])
    if (
        ((diameter_hint is not None)
        and (np.asarray(diameter_hint).size == n))
        and np.any((np.asarray(diameter_hint) > 0))
    ):
        d_hint = np.asarray(diameter_hint, dtype = np.float32)
        radius_mm = float(np.clip((np.percentile(d_hint[(d_hint > 0)], 80.0) * 0.85), 1.2, 6.0))
    else:
        radius_mm = 3.0
    step_mm = float(np.clip((np.min(sp) * 0.75), 0.3, 0.6))
    grid = np.arange(-radius_mm, (radius_mm + (0.5 * step_mm)), step_mm, dtype = np.float32)
    (uu, vv) = np.meshgrid(grid, grid, indexing = "xy")
    disk = (((uu * uu) + (vv * vv)) <= (radius_mm * radius_mm))
    offsets_2d = np.stack([uu[disk], vv[disk]], axis = 1).astype(np.float32)
    area_unit = (step_mm * step_mm)
    out = np.zeros(n, dtype = np.float32)
    ref_axis_options = [
        np.array([0.0, 0.0, 1.0], dtype = np.float32),
        np.array([0.0, 1.0, 0.0], dtype = np.float32),
        np.array([1.0, 0.0, 0.0], dtype = np.float32),
    ]
    shape = np.asarray(mask.shape, dtype = np.int32)
    for i in range(n):
        a = max(0, (i - 3))
        b = min((n - 1), (i + 3))
        tangent = (phys[b] - phys[a])
        norm = float(np.linalg.norm(tangent))
        if (norm <= 1e-06):
            out[i] = 0.0
            continue
        t = (tangent / norm)
        ref_axis = ref_axis_options[0]
        if (abs(float(np.dot(t, ref_axis))) > 0.88):
            ref_axis = ref_axis_options[1]
        if (abs(float(np.dot(t, ref_axis))) > 0.88):
            ref_axis = ref_axis_options[2]
        u = np.cross(t, ref_axis)
        u_norm = float(np.linalg.norm(u))
        if (u_norm <= 1e-06):
            out[i] = 0.0
            continue
        u = (u / u_norm)
        v = np.cross(t, u)
        v_norm = float(np.linalg.norm(v))
        if (v_norm <= 1e-06):
            out[i] = 0.0
            continue
        v = (v / v_norm)
        samples_phys = (
            (phys[i][None, :]
            + (offsets_2d[:, 0:1] * u[None, :]))
            + (offsets_2d[:, 1:2] * v[None, :])
        )
        vox = np.rint((samples_phys / sp[None, :])).astype(np.int32)
        good = np.all(((vox >= 0) & (vox < shape[None, :])), axis = 1)
        if not np.any(good):
            out[i] = 0.0
            continue
        vv_idx = vox[good]
        hits = mask[vv_idx[:, 0], vv_idx[:, 1], vv_idx[:, 2]]
        out[i] = float((np.sum(hits) * area_unit))
    if (out.size >= 7):
        out = gaussian_filter1d(out, sigma = max(0.7, (out.size / 200.0)), mode = "nearest")
    return np.maximum(out.astype(np.float32), 0.0)

def calcium_penalty_profile(
    calcium_mask: Optional[np.ndarray],
    path: np.ndarray,
    spacing: Tuple[float, float, float],
) -> np.ndarray:
    if ((calcium_mask is None) or (path.shape[0] == 0)):
        return np.zeros(path.shape[0], dtype = np.float32)
    cm = np.asarray(calcium_mask).astype(bool)
    if not np.any(cm):
        return np.zeros(path.shape[0], dtype = np.float32)
    dist_to_calc = distance_transform_edt(~cm, sampling = spacing).astype(np.float32)
    rr = np.clip(path[:, 0], 0, (cm.shape[0] - 1))
    cc = np.clip(path[:, 1], 0, (cm.shape[1] - 1))
    zz = np.clip(path[:, 2], 0, (cm.shape[2] - 1))
    d = dist_to_calc[rr, cc, zz]
    penalty = np.exp((-0.5 * ((d / 1.35) ** 2))).astype(np.float32)
    if (penalty.size >= 7):
        penalty = gaussian_filter1d(
            penalty, sigma = max(0.65, (penalty.size / 220.0)), mode = "nearest"
        )
    return np.clip(penalty, 0.0, 1.0).astype(np.float32)

def reference_diameter_profile(
    diameter: np.ndarray, cum: Optional[np.ndarray] = None
) -> np.ndarray:
    d = np.asarray(diameter, dtype = np.float32)
    n = int(d.size)
    if (n == 0):
        return d
    d = np.nan_to_num(d, nan = 0.0, posinf = 0.0, neginf = 0.0)
    positive = d[(d > 0)]
    floor = max(0.2, float(np.percentile(positive, 5.0)) if positive.size else 0.2)
    d = np.maximum(d, floor)
    if (n < 7):
        return np.full_like(d, float(np.max(d)))
    smooth = gaussian_filter1d(d, sigma = max(1.0, (n / 125.0)), mode = "nearest")
    win = max(9, int(round((n * 0.24))))
    if ((win % 2) == 0):
        win += 1
    envelope = percentile_filter(smooth, percentile = 80.0, size = win, mode = "nearest")
    base_ref = np.maximum(envelope, smooth)
    if (cum is None):
        ref = gaussian_filter1d(base_ref, sigma = max(1.0, (n / 145.0)), mode = "nearest")
        return np.maximum(ref, (smooth + 0.0001)).astype(np.float32)
    s = np.asarray(cum, dtype = np.float32)
    if ((s.size != n) or (float(s[-1]) <= 1e-06)):
        ref = gaussian_filter1d(base_ref, sigma = max(1.0, (n / 145.0)), mode = "nearest")
        return np.maximum(ref, (smooth + 0.0001)).astype(np.float32)
    total = float(s[-1])
    buffer_mm = max(2.0, (0.018 * total))
    window_mm = max(10.0, min(28.0, (0.22 * total)))
    bidir = base_ref.copy()
    for i in range(n):
        center = float(s[i])
        prox = np.where(((s >= (center - window_mm)) & (s <= (center - buffer_mm))))[0]
        dist = np.where(((s >= (center + buffer_mm)) & (s <= (center + window_mm))))[0]
        refs = []
        if (prox.size >= 3):
            refs.append(float(np.percentile(smooth[prox], 72.0)))
        if (dist.size >= 3):
            refs.append(float(np.percentile(smooth[dist], 72.0)))
        if (len(refs) == 2):
            bidir[i] = max(base_ref[i], ((0.5 * refs[0]) + (0.5 * refs[1])))
        elif (len(refs) == 1):
            bidir[i] = max(base_ref[i], ((0.72 * base_ref[i]) + (0.28 * refs[0])))
    ref = np.maximum(base_ref, bidir)
    ref = gaussian_filter1d(ref, sigma = max(1.0, (n / 145.0)), mode = "nearest")
    ref = np.maximum(ref, (smooth + 0.0001))
    return ref.astype(np.float32)

def compute_centerline_concepts(
    branch_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    stenosis_threshold: float = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
    calcium_mask: Optional[np.ndarray] = None,
) -> models.CenterlineConcept:
    (path, method, quality) = graph.extract_centerline_path(branch_mask, spacing)
    if (path.shape[0] < 2):
        return models.CenterlineConcept(
            method = method,
            quality_flag = quality,
            point_count = int(path.shape[0]),
            length_mm = 0.0,
            endpoint_distance_mm = 0.0,
            tortuosity = 0.0,
            mean_diameter_mm = 0.0,
            median_diameter_mm = 0.0,
            min_diameter_mm = 0.0,
            max_diameter_mm = 0.0,
            stenosis_presence = 0,
            stenosis_ratio = 0.0,
            stenosis_position_norm = -1.0,
            stenosis_segment = "none",
            stenosis_length_mm = 0.0,
            reference_diameter_mm = 0.0,
            min_lumen_diameter_mm = 0.0,
        )
    radius_map = distance_transform_edt(branch_mask.astype(bool), sampling = spacing).astype(
        np.float32
    )
    rr = np.clip(path[:, 0], 0, (branch_mask.shape[0] - 1))
    cc = np.clip(path[:, 1], 0, (branch_mask.shape[1] - 1))
    zz = np.clip(path[:, 2], 0, (branch_mask.shape[2] - 1))
    radius = radius_map[rr, cc, zz]
    diameter = np.maximum((2.0 * radius), 0.0).astype(np.float32)
    if (diameter.size >= 7):
        diameter_smooth = gaussian_filter1d(
            diameter, sigma = max(0.65, (diameter.size / 180.0)), mode = "nearest"
        )
    else:
        diameter_smooth = diameter
    cum = path_arclength_mm(path, spacing)
    length = float(cum[-1]) if (cum.size > 0) else 0.0
    p0 = (path[0].astype(np.float32) * np.asarray(spacing, dtype = np.float32))
    p1 = (path[-1].astype(np.float32) * np.asarray(spacing, dtype = np.float32))
    endpoint_dist = float(np.linalg.norm((p1 - p0)))
    tort = float((length / max(endpoint_dist, 1e-06))) if (length > 0) else 0.0
    ref = reference_diameter_profile(diameter_smooth, cum)
    area_profile = cross_section_area_profile(
        branch_mask, path, spacing, diameter_hint = diameter_smooth
    )
    area_ref = reference_diameter_profile(
        (2.0 * np.sqrt((np.maximum(area_profile, 0.0) / math.pi))), cum
    )
    area_ref = (math.pi * (np.maximum((area_ref / 2.0), 1e-06) ** 2))
    calc_penalty = calcium_penalty_profile(calcium_mask, path, spacing)
    sten = combined_stenosis_profile(
        diameter_smooth,
        ref,
        cum,
        area = area_profile,
        area_reference = area_ref,
        calcium_penalty = calc_penalty,
    )
    ratio = np.asarray(sten.get("combined", []), dtype = np.float32)
    n = int(ratio.size)
    valid = np.ones(n, dtype = bool)
    if (n >= 20):
        valid[: max(1, int((0.06 * n)))] = False
        valid[-max(1, int((0.06 * n))) :] = False
    if np.any(valid):
        valid_ratio = ratio.copy()
        valid_ratio[~valid] = -1.0
        idx = int(np.argmax(valid_ratio))
    else:
        idx = int(np.argmax(ratio)) if ratio.size else 0
    stenosis_ratio = float(max(0.0, ratio[idx])) if ratio.size else 0.0
    position_norm = (
        float((cum[idx] / max(length, 1e-06))) if ((cum.size > idx) and (length > 0)) else -1.0
    )
    segment = segment_name_from_norm(position_norm)
    lesion_len = 0.0
    stenosis_presence = 0
    if (ratio.size > 0):
        floor = max(float(stenosis_threshold), (0.6 * stenosis_ratio))
        flags = (ratio >= floor)
        a = idx
        b = idx
        while (((a - 1) >= 0) and flags[(a - 1)]):
            a -= 1
        while (((b + 1) < n) and flags[(b + 1)]):
            b += 1
        lesion_len = float((cum[b] - cum[a])) if (cum.size > b) else 0.0
        area_peak = (
            float(
                np.asarray(sten.get("area_ratio", np.zeros_like(ratio)), dtype = np.float32)[
                    idx
                ]
            )
            if ratio.size
            else 0.0
        )
        calcium_peak = (
            float(
                np.asarray(
                    sten.get("calcium_penalty", np.zeros_like(ratio)), dtype = np.float32
                )[idx]
            )
            if ratio.size
            else 0.0
        )
        area_agreement_ok = bool((area_peak >= max(0.18, (0.55 * stenosis_ratio))))
        min_len_regular = max(2.0, (0.02 * max(length, 1.0)))
        min_len_area_supported = max(1.0, (0.01 * max(length, 1.0)))
        persistence_ok = bool(
            (lesion_len >= (min_len_area_supported if area_agreement_ok else min_len_regular))
        )
        strong_narrowing = bool((stenosis_ratio >= max(float(stenosis_threshold), 0.25)))
        borderline_ok = bool(
            ((((stenosis_ratio >= 0.25)
            and area_agreement_ok)
            or ((stenosis_ratio >= 0.32) and (lesion_len >= min_len_regular)))
            or (stenosis_ratio >= 0.5))
        )
        calcium_ok = bool(
            (((calcium_peak < 0.7)
            or ((stenosis_ratio >= 0.25) and area_agreement_ok))
            or (stenosis_ratio >= 0.5))
        )
        stenosis_presence = int(
            (((strong_narrowing and persistence_ok) and borderline_ok) and calcium_ok)
        )
    if not stenosis_presence:
        position_norm = -1.0
        segment = "none"
        lesion_len = 0.0
    return models.CenterlineConcept(
        method = method,
        quality_flag = quality,
        point_count = int(path.shape[0]),
        length_mm = float(length),
        endpoint_distance_mm = float(endpoint_dist),
        tortuosity = float(tort),
        mean_diameter_mm = float(np.mean(diameter_smooth)) if diameter_smooth.size else 0.0,
        median_diameter_mm = (
            float(np.median(diameter_smooth)) if diameter_smooth.size else 0.0
        ),
        min_diameter_mm = float(np.min(diameter_smooth)) if diameter_smooth.size else 0.0,
        max_diameter_mm = float(np.max(diameter_smooth)) if diameter_smooth.size else 0.0,
        stenosis_presence = int(stenosis_presence),
        stenosis_ratio = float(stenosis_ratio if stenosis_presence else 0.0),
        stenosis_position_norm = float(position_norm if stenosis_presence else -1.0),
        stenosis_segment = str(segment),
        stenosis_length_mm = float(lesion_len if stenosis_presence else 0.0),
        reference_diameter_mm = float(ref[idx]) if ref.size else 0.0,
        min_lumen_diameter_mm = float(diameter_smooth[idx]) if diameter_smooth.size else 0.0,
    )

def branch_long_name(short: str) -> str:
    return runtime.BRANCH_LONG_NAMES.get(short, short.lower())

def centerline_profile_arrays(
    branch_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    calcium_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    (path, method, quality) = graph.extract_centerline_path(branch_mask, spacing)
    if (path.shape[0] < 2):
        return {"ok": False, "path": path, "method": method, "quality": quality}
    radius_map = distance_transform_edt(branch_mask.astype(bool), sampling = spacing).astype(
        np.float32
    )
    rr = np.clip(path[:, 0], 0, (branch_mask.shape[0] - 1))
    cc = np.clip(path[:, 1], 0, (branch_mask.shape[1] - 1))
    zz = np.clip(path[:, 2], 0, (branch_mask.shape[2] - 1))
    diameter = np.maximum((2.0 * radius_map[rr, cc, zz]), 0.0).astype(np.float32)
    if (diameter.size >= 7):
        diameter = gaussian_filter1d(
            diameter, sigma = max(0.65, (diameter.size / 180.0)), mode = "nearest"
        )
    cum = path_arclength_mm(path, spacing)
    ref = reference_diameter_profile(diameter, cum)
    area_profile = cross_section_area_profile(
        branch_mask, path, spacing, diameter_hint = diameter
    )
    area_ref_diam = reference_diameter_profile(
        (2.0 * np.sqrt((np.maximum(area_profile, 0.0) / math.pi))), cum
    )
    area_ref = (math.pi * (np.maximum((area_ref_diam / 2.0), 1e-06) ** 2))
    calc_penalty = calcium_penalty_profile(calcium_mask, path, spacing)
    sten = combined_stenosis_profile(
        diameter,
        ref,
        cum,
        area = area_profile,
        area_reference = area_ref,
        calcium_penalty = calc_penalty,
    )
    ratio = np.asarray(sten.get("combined", []), dtype = np.float32)
    return {
        "ok": True,
        "path": path,
        "cum": cum,
        "diameter": diameter,
        "reference": ref,
        "area": area_profile,
        "area_reference": area_ref,
        "calcium_penalty": calc_penalty,
        "ratio": ratio,
        "ratio_raw": sten.get("raw"),
        "ratio_area": sten.get("area_ratio"),
        "method": method,
        "quality": quality,
    }

def extract_calcification(
    hu: np.ndarray,
    vessel_mask: np.ndarray,
    threshold_hu: float,
    voxel_vol: float,
    min_vol_mm3: float,
) -> np.ndarray:
    vessel = vessel_mask.astype(bool)
    if not np.any(vessel):
        return np.zeros_like(vessel, dtype = bool)
    equiv_spacing = float((max(float(voxel_vol), 1e-06) ** (1.0 / 3.0)))
    radius_map = distance_transform_edt(
        vessel, sampling = (equiv_spacing, equiv_spacing, equiv_spacing)
    ).astype(np.float32)
    core = (vessel & (radius_map >= runtime.CALCIFICATION_CORE_MIN_DISTANCE_MM))
    vals = np.asarray(hu[core if np.any(core) else vessel], dtype = np.float32)
    vals = vals[np.isfinite(vals)]
    if (vals.size == 0):
        adaptive_thr = max(float(threshold_hu), runtime.CALCIFICATION_MIN_ADAPTIVE_HU)
    else:
        med = float(np.median(vals))
        mad = float(np.median(np.abs((vals - med))))
        robust_sigma = (1.4826 * mad)
        adaptive_thr = max(
            float(threshold_hu),
            runtime.CALCIFICATION_MIN_ADAPTIVE_HU,
            (med + max(120.0, (2.5 * robust_sigma))),
        )
    shell = (vessel & (radius_map <= runtime.CALCIFICATION_PERIPHERAL_BAND_MM))
    ultra_high = (vessel & (hu >= max(650.0, (adaptive_thr + 150.0))))
    raw = (((hu >= adaptive_thr) & shell) | ultra_high)
    raw &= vessel
    out = remove_small_components_by_volume(raw, voxel_vol, float(min_vol_mm3))
    return out.astype(bool)
