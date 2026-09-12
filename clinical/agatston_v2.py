from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import find_objects, label as ndi_label
from skimage.measure import label as skimage_label

try:
    import cv2
except Exception:
    cv2 = None

import cta_legacy_backend_v1_0_15 as legacy

AGATSTON_ALGORITHM_VERSION = "ccta_wall_core_morphology_volume_conversion_v5"
DEFAULT_RECONSTRUCTION_THICKNESS_MM = 3.0
DEFAULT_MINIMUM_LESION_AREA_MM2 = 1.0
DEFAULT_PERIVASCULAR_RADIUS_MM = 1.25
DEFAULT_VOLUME_TO_AGATSTON_FACTOR = 3.13

@dataclass
class AgatstonBranchResult:
    branch: str
    score: float
    raw_volume_conversion_score: float
    direct_density_score: float
    calcium_volume_mm3: float
    lesion_count: int
    accepted_area_mm2: float
    maximum_hu: float
    shell_threshold_hu: float
    core_threshold_hu: float

@dataclass
class AgatstonResult:
    agatston_score: float
    agatston_score_raw: float
    agatston_score_direct: float
    agatston_score_legacy: float
    agatston_grade: str
    risk_category: str
    calcium_threshold_hu: float
    shell_threshold_hu_median: float
    core_threshold_hu_median: float
    lumen_mean_hu: float
    lumen_std_hu: float
    lumen_sample_count: int
    reconstruction_thickness_mm: float
    minimum_lesion_area_mm2: float
    minimum_component_volume_mm3: float
    calcium_volume_mm3: float
    agatston_lesion_count: int
    direct_lesion_count: int
    method: str
    algorithm_version: str
    calibration_applied: int
    calibration_description: str
    branch_results: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _robust_stats(values: np.ndarray, minimum_scale: float = 8.0) -> Tuple[float, float, int]:
    vals = np.asarray(values, dtype = np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[((vals > -100.0) & (vals < 1500.0))]
    if (vals.size == 0):
        return 250.0, 50.0, 0
    if (vals.size > 250_000):
        vals = np.sort(vals)[np.linspace(0, (vals.size - 1), 250_000, dtype = np.int64)]
    lo, hi = np.percentile(vals, [20.0, 85.0])
    clipped = vals[((vals >= lo) & (vals <= hi))]
    if (clipped.size < 64):
        clipped = vals
    for _ in range(4):
        med = float(np.median(clipped))
        mad = float(np.median(np.abs((clipped - med))))
        sigma = max((1.4826 * mad), minimum_scale)
        keep = ((clipped >= (med - (2.5 * sigma))) & (clipped <= (med + (2.5 * sigma))))
        if (int(np.sum(keep)) < max(32, int((0.45 * clipped.size)))):
            break
        updated = clipped[keep]
        if (updated.size == clipped.size):
            break
        clipped = updated
    return (
        float(np.mean(clipped)),
        float(
            max((np.std(clipped, ddof = 1) if (clipped.size > 1) else minimum_scale), minimum_scale)
        ),
        int(clipped.size),
    )

def _component_filter(
    candidate: np.ndarray,
    shell: np.ndarray,
    image: np.ndarray,
    voxel_volume_mm3: float,
    minimum_volume_mm3: float,
    pixel_area_mm2: float,
    minimum_axial_area_mm2: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    cand = np.asarray(candidate).astype(bool)
    structure = np.ones((3, 3, 3), dtype = np.uint8)
    labeled, count = ndi_label(cand, structure = structure)
    out = np.zeros_like(cand, dtype = bool)
    lesions: List[Dict[str, Any]] = []
    if (int(count) <= 0):
        return out, lesions
    counts = np.bincount(labeled.ravel())
    objects = find_objects(labeled)
    image_arr = np.asarray(image)
    shell_arr = np.asarray(shell).astype(bool)
    for cid in range(1, (int(count) + 1)):
        voxels = (int(counts[cid]) if (cid < counts.size) else 0)
        volume = float((voxels * voxel_volume_mm3))
        if ((voxels <= 0) or ((volume + 1e-9) < float(minimum_volume_mm3))):
            continue
        sl = (objects[(cid - 1)] if ((cid - 1) < len(objects)) else None)
        if (sl is None):
            continue
        local_labels = labeled[sl]
        comp = (local_labels == cid)
        vals = image_arr[sl][comp]
        if (vals.size == 0):
            continue
        max_axial_area = float(
            max(
                ((np.sum(comp[:, :, z]) * float(pixel_area_mm2)) for z in range(comp.shape[2])),
                default = 0.0,
            )
        )
        if ((max_axial_area + 1e-9) < float(minimum_axial_area_mm2)):
            continue
        peak = float(np.max(vals))
        mean = float(np.mean(vals))
        shell_fraction = float(np.mean(shell_arr[sl][comp]))
        if ((shell_fraction < 0.25) and (peak < 700.0)):
            continue
        if ((volume > 220.0) and (peak < 750.0)):
            continue
        out_view = out[sl]
        out_view[comp] = True
        lesions.append(
            {
                "component_id": int(cid),
                "voxel_count": voxels,
                "volume_mm3": volume,
                "peak_hu": peak,
                "mean_hu": mean,
                "shell_fraction": shell_fraction,
            }
        )
    return out, lesions

def _density_weight_contrast(max_hu: float) -> int:
    value = float(max_hu)
    if (value >= 700.0):
        return 4
    if (value >= 550.0):
        return 3
    if (value >= 430.0):
        return 2
    return 1

def _slab_ranges(
    slice_count: int, spacing_z_mm: float, target_thickness_mm: float
) -> List[Tuple[int, int]]:
    count = max(1, int(round((max(target_thickness_mm, spacing_z_mm) / max(spacing_z_mm, 1e-6)))))
    return [(start, min(slice_count, (start + count))) for start in range(0, slice_count, count)]

def _direct_density_score(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    spacing: Sequence[float],
    target_thickness_mm: float,
    minimum_area_mm2: float,
) -> Tuple[float, int, float]:
    pixel_area = (float(spacing[0]) * float(spacing[1]))
    total = 0.0
    lesions = 0
    accepted_area = 0.0
    for start, end in _slab_ranges(image.shape[2], float(spacing[2]), float(target_thickness_mm)):
        mask_2d = np.any(candidate_mask[:, :, start:end], axis = 2)
        if not np.any(mask_2d):
            continue
        labeled, count = skimage_label(mask_2d, background = 0, return_num = True, connectivity = 2)
        for cid in range(1, (int(count) + 1)):
            lesion = (labeled == cid)
            area = float((np.sum(lesion) * pixel_area))
            if ((area + 1e-9) < float(minimum_area_mm2)):
                continue
            native = (candidate_mask[:, :, start:end] & lesion[:, :, None])
            peak = (
                float(np.max(np.asarray(image)[:, :, start:end][native])) if np.any(native) else 0.0
            )
            total += (area * float(_density_weight_contrast(peak)))
            lesions += 1
            accepted_area += area
    return float(total), int(lesions), float(accepted_area)

def _load_calibration(
    calibration: Optional[((Mapping[str, Any] | Path) | str)]
) -> Optional[Dict[str, Any]]:
    if (calibration is None):
        return None
    if isinstance(calibration, Mapping):
        return dict(calibration)
    raw = str(calibration).strip()
    if ((len(raw) >= 2) and (raw[0] == raw[-1]) and (raw[0] in ("'", '"'))):
        raw = raw[1:-1].strip()
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Agatston calibration file does not exist: {path}")
    return json.loads(path.read_text(encoding = "utf-8"))

def _apply_calibration(
    score: float, calibration: Optional[Dict[str, Any]]
) -> Tuple[float, int, str]:
    raw_score = max(float(score), 0.0)
    if not calibration:
        return raw_score, 0, "none"
    model = str(calibration.get("model", "identity")).lower()
    if (raw_score <= 0.0):
        return 0.0, int((model != "identity")), f"{model}: zero_preserved"
    if (model == "identity"):
        return raw_score, 0, "identity_selected_by_cross_validation"
    if (model in ("zero_linear", "linear_zero_intercept")):
        slope = float(calibration.get("slope", 1.0))
        value = (slope * raw_score)
        desc = f"zero_linear: slope={slope:.6g}"
    elif (model in ("zero_power", "power")):
        scale = float(calibration.get("scale", 1.0))
        exponent = float(calibration.get("exponent", 1.0))
        value = (scale * (raw_score ** exponent))
        desc = f"zero_power: scale={scale:.6g}, exponent={exponent:.6g}"
    elif (model == "linear"):
        intercept = float(calibration.get("intercept", 0.0))
        slope = float(calibration.get("slope", 1.0))
        value = (intercept + (slope * raw_score))
        desc = f"linear_with_zero_guard: intercept={intercept:.6g}, slope={slope:.6g}"
    else:
        intercept = float(calibration.get("intercept", 0.0))
        slope = float(calibration.get("slope", 1.0))
        value = math.expm1((intercept + (slope * math.log1p(raw_score))))
        desc = f"log_linear_with_zero_guard: intercept={intercept:.6g}, slope={slope:.6g}"
    return float(max(value, 0.0)), 1, desc

def _bbox_from_label_value(
    labels: np.ndarray,
    label_value: int,
    spacing: Sequence[float],
    margin_mm: float,
) -> Optional[Tuple[slice, slice, slice]]:
    coords = np.where((np.asarray(labels) == int(label_value)))
    if (coords[0].size == 0):
        return None
    shape = np.asarray(labels.shape, dtype = int)
    sp = np.asarray(spacing, dtype = float)
    pad = (np.ceil((float(margin_mm) / np.maximum(sp, 1e-6))).astype(int) + 2)
    lo = np.maximum((np.asarray([int(np.min(a)) for a in coords]) - pad), 0)
    hi = np.minimum(((np.asarray([int(np.max(a)) for a in coords]) + pad) + 1), shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

def _fast_wall_core_candidate(
    image_crop: np.ndarray,
    label_crop: np.ndarray,
    mask: np.ndarray,
    label_value: int,
    spacing: Sequence[float],
    shell_threshold: float,
    core_threshold: float,
    perivascular_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask).astype(np.uint8)
    candidate = np.zeros_like(m, dtype = bool)
    shell_candidate = np.zeros_like(m, dtype = bool)
    sy = max(float(spacing[0]), 1e-6)
    sx = max(float(spacing[1]), 1e-6)
    ry = max(1, int(math.ceil((float(perivascular_radius_mm) / sy))))
    rx = max(1, int(math.ceil((float(perivascular_radius_mm) / sx))))
    ery = max(1, int(math.ceil((0.45 / sy))))
    erx = max(1, int(math.ceil((0.45 / sx))))
    if (cv2 is not None):
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (((2 * rx) + 1), ((2 * ry) + 1)))
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (((2 * erx) + 1), ((2 * ery) + 1)))
    else:
        from scipy.ndimage import binary_dilation, binary_erosion

        yy, xx = np.ogrid[-ry : (ry + 1), -rx : (rx + 1)]
        kd = ((((yy / max(ry, 1)) ** 2) + ((xx / max(rx, 1)) ** 2)) <= 1.0)
        yy, xx = np.ogrid[-ery : (ery + 1), -erx : (erx + 1)]
        ke = ((((yy / max(ery, 1)) ** 2) + ((xx / max(erx, 1)) ** 2)) <= 1.0)
    active = np.where(np.any((m > 0), axis = (0, 1)))[0]
    for z in active:
        sl = m[:, :, int(z)]
        if (cv2 is not None):
            dil = cv2.dilate(sl, kd, iterations = 1).astype(bool)
            ero = cv2.erode(sl, ke, iterations = 1).astype(bool)
        else:
            dil = binary_dilation(sl.astype(bool), structure = kd)
            ero = binary_erosion(sl.astype(bool), structure = ke)
        wall = (dil & (~ero))
        other_branch = ((label_crop[:, :, int(z)] != 0) & (
            label_crop[:, :, int(z)] != int(label_value)
        ))
        wall &= ~other_branch
        img = image_crop[:, :, int(z)]
        shell_hit = ((wall & np.isfinite(img)) & (img >= float(shell_threshold)))
        core_hit = ((ero & np.isfinite(img)) & (img >= float(core_threshold)))
        candidate[:, :, int(z)] = (shell_hit | core_hit)
        shell_candidate[:, :, int(z)] = shell_hit
    return candidate, shell_candidate

def compute_agatston(
    image: np.ndarray,
    aligned_label: np.ndarray,
    branch_specs: Sequence[Mapping[str, Any]],
    spacing: Sequence[float],
    dicom_meta: Optional[Mapping[str, Any]] = None,
    calibration: Optional[((Mapping[str, Any] | Path) | str)] = None,
    reconstruction_thickness_mm: float = DEFAULT_RECONSTRUCTION_THICKNESS_MM,
    minimum_lesion_area_mm2: float = DEFAULT_MINIMUM_LESION_AREA_MM2,
    minimum_component_volume_mm3: float = 0.20,
    perivascular_radius_mm: float = DEFAULT_PERIVASCULAR_RADIUS_MM,
    volume_conversion_factor: float = DEFAULT_VOLUME_TO_AGATSTON_FACTOR,
) -> AgatstonResult:
    image_arr = np.asarray(image, dtype = np.float32)
    labels = np.asarray(aligned_label)
    spacing_tuple = tuple(float(v) for v in spacing)
    voxel_volume = float(np.prod(np.asarray(spacing_tuple, dtype = float)))
    assigned_flat: set[int] = set()
    branch_state: Dict[str, AgatstonBranchResult] = {}
    shell_thresholds: List[float] = []
    core_thresholds: List[float] = []
    lumen_means: List[float] = []
    lumen_stds: List[float] = []
    lumen_counts: List[int] = []
    total_direct_lesions = 0
    global_shape = tuple(int(v) for v in image_arr.shape)

    kvp = float((dicom_meta or {}).get("kvp", np.nan))
    if (np.isfinite(kvp) and (kvp <= 90)):
        shell_floor, core_floor = 750.0, 950.0
    elif (np.isfinite(kvp) and (kvp >= 110)):
        shell_floor, core_floor = 550.0, 720.0
    else:
        shell_floor, core_floor = 550.0, 720.0

    for spec in branch_specs:
        branch = str(spec["branch"])
        label_value = int(spec["label_value"])
        bbox = _bbox_from_label_value(
            labels,
            label_value,
            spacing_tuple,
            margin_mm = (float(perivascular_radius_mm) + 1.5),
        )
        if (bbox is None):
            branch_state[branch] = AgatstonBranchResult(
                branch,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                float(shell_floor),
                float(core_floor),
            )
            continue
        mask = np.ascontiguousarray((labels[bbox] == label_value))
        image_crop = np.ascontiguousarray(image_arr[bbox])
        label_crop = np.ascontiguousarray(labels[bbox])
        if not np.any(mask):
            branch_state[branch] = AgatstonBranchResult(
                branch,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                float(shell_floor),
                float(core_floor),
            )
            continue

        mu, sigma, count = _robust_stats(image_crop[mask])
        lumen_means.append(mu)
        lumen_stds.append(sigma)
        lumen_counts.append(int(count))
        shell_threshold = float(
            np.clip(max(shell_floor, (mu + (2.30 * sigma))), shell_floor, 900.0)
        )
        core_threshold = float(np.clip(max(core_floor, (mu + (4.00 * sigma))), core_floor, 1250.0))
        shell_thresholds.append(shell_threshold)
        core_thresholds.append(core_threshold)

        candidate, shell_candidate = _fast_wall_core_candidate(
            image_crop,
            label_crop,
            mask,
            label_value,
            spacing_tuple,
            shell_threshold,
            core_threshold,
            perivascular_radius_mm,
        )
        coords = np.argwhere(candidate)
        if (coords.size == 0):
            branch_state[branch] = AgatstonBranchResult(
                branch,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                shell_threshold,
                core_threshold,
            )
            continue

        origin = np.asarray([bbox[0].start, bbox[1].start, bbox[2].start], dtype = np.int64)
        global_coords = (coords.astype(np.int64) + origin[None, :])
        flat = np.ravel_multi_index(global_coords.T, global_shape)
        if assigned_flat:
            fresh = np.fromiter(
                ((int(v) not in assigned_flat) for v in flat), dtype = bool, count = flat.size
            )
            coords = coords[fresh]
            global_coords = global_coords[fresh]
            flat = flat[fresh]
        if (coords.size == 0):
            branch_state[branch] = AgatstonBranchResult(
                branch,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                shell_threshold,
                core_threshold,
            )
            continue

        lo = np.maximum((coords.min(axis = 0) - 1), 0)
        hi = np.minimum((coords.max(axis = 0) + 2), np.asarray(candidate.shape))
        small_slices = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
        local = (coords - lo[None, :])
        small_shape = tuple(int((hi[i] - lo[i])) for i in range(3))
        candidate_small = np.zeros(small_shape, dtype = bool)
        candidate_small[local[:, 0], local[:, 1], local[:, 2]] = True
        shell_small_full = shell_candidate[small_slices]
        image_small = np.ascontiguousarray(image_crop[small_slices])
        filtered, lesion_info = _component_filter(
            candidate_small,
            shell_small_full,
            image_small,
            voxel_volume,
            minimum_component_volume_mm3,
            pixel_area_mm2 = float((spacing_tuple[0] * spacing_tuple[1])),
            minimum_axial_area_mm2 = float(minimum_lesion_area_mm2),
        )
        accepted_local = np.argwhere(filtered)
        if accepted_local.size:
            accepted_crop = (accepted_local + lo[None, :])
            accepted_global = (accepted_crop.astype(np.int64) + origin[None, :])
            accepted_ids = np.ravel_multi_index(accepted_global.T, global_shape)
            assigned_flat.update(int(v) for v in accepted_ids.tolist())

        volume = float((np.sum(filtered) * voxel_volume))
        raw_score = float((volume * float(volume_conversion_factor)))
        direct_score, direct_lesions, accepted_area = _direct_density_score(
            image_small,
            filtered,
            spacing_tuple,
            reconstruction_thickness_mm,
            minimum_lesion_area_mm2,
        )
        total_direct_lesions += int(direct_lesions)
        maximum_hu = float(max((item["peak_hu"] for item in lesion_info), default = 0.0))
        branch_state[branch] = AgatstonBranchResult(
            branch = branch,
            score = raw_score,
            raw_volume_conversion_score = raw_score,
            direct_density_score = float(direct_score),
            calcium_volume_mm3 = volume,
            lesion_count = int(len(lesion_info)),
            accepted_area_mm2 = float(accepted_area),
            maximum_hu = maximum_hu,
            shell_threshold_hu = shell_threshold,
            core_threshold_hu = core_threshold,
        )
        del candidate, shell_candidate, coords, global_coords, flat

    raw_total = float(sum(item.raw_volume_conversion_score for item in branch_state.values()))
    direct_total = float(sum(item.direct_density_score for item in branch_state.values()))
    calcium_volume = float(sum(item.calcium_volume_mm3 for item in branch_state.values()))
    total_lesions = int(sum(item.lesion_count for item in branch_state.values()))
    calibration_dict = _load_calibration(calibration)
    final_score, calibration_applied, calibration_description = _apply_calibration(
        raw_total, calibration_dict
    )
    grade, risk = legacy.agatston_grade(final_score)
    branch_results = {branch: asdict(result) for branch, result in branch_state.items()}
    method = (
        "Wall-aware contrast-CCTA calcium extraction using physical axial shell/core "
        "morphology, patient-specific branch lumen/noise thresholds, cropped 3-D "
        "persistence filtering, and the published 3.13 AU/mm3 contrast-CCTA volume "
        "conversion. A density-area score is reported independently; optional paired-"
        "report calibration is explicit and never silently applied."
    )
    return AgatstonResult(
        agatston_score = float(final_score),
        agatston_score_raw = float(raw_total),
        agatston_score_direct = float(direct_total),
        agatston_score_legacy = float("nan"),
        agatston_grade = str(grade),
        risk_category = str(risk),
        calcium_threshold_hu = (
            float(np.median(shell_thresholds)) if shell_thresholds else float(shell_floor)
        ),
        shell_threshold_hu_median = (
            float(np.median(shell_thresholds)) if shell_thresholds else float(shell_floor)
        ),
        core_threshold_hu_median = (
            float(np.median(core_thresholds)) if core_thresholds else float(core_floor)
        ),
        lumen_mean_hu = (
            float(np.average(lumen_means, weights = np.maximum(lumen_counts, 1)))
            if lumen_means
            else 0.0
        ),
        lumen_std_hu = (
            float(np.average(lumen_stds, weights = np.maximum(lumen_counts, 1)))
            if lumen_stds
            else 0.0
        ),
        lumen_sample_count = int(sum(lumen_counts)),
        reconstruction_thickness_mm = float(reconstruction_thickness_mm),
        minimum_lesion_area_mm2 = float(minimum_lesion_area_mm2),
        minimum_component_volume_mm3 = float(minimum_component_volume_mm3),
        calcium_volume_mm3 = float(calcium_volume),
        agatston_lesion_count = int(total_lesions),
        direct_lesion_count = int(total_direct_lesions),
        method = method,
        algorithm_version = AGATSTON_ALGORITHM_VERSION,
        calibration_applied = int(calibration_applied),
        calibration_description = str(calibration_description),
        branch_results = branch_results,
    )
