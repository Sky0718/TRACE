from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label as ndi_label, map_coordinates, median_filter

from .models import (
    CrossSectionMeasurement,
    PlanePatch,
)

def _robust_location_scale(values: np.ndarray, minimum_scale: float = 5.0) -> Tuple[float, float]:
    vals = np.asarray(values, dtype = np.float64)
    vals = vals[np.isfinite(vals)]
    if (vals.size == 0):
        return 0.0, float(minimum_scale)
    med = float(np.median(vals))
    mad = float(np.median(np.abs((vals - med))))
    return med, max((1.4826 * mad), float(minimum_scale))

def _component_from_seed(
    binary: np.ndarray, seed_y: int, seed_x: int, max_distance_px: float = 8.0
) -> np.ndarray:
    b = np.asarray(binary).astype(bool)
    labeled, count = ndi_label(b, structure = np.ones((3, 3), dtype = np.uint8))
    if (count <= 0):
        return np.zeros_like(b, dtype = bool)
    sy = int(np.clip(seed_y, 0, (b.shape[0] - 1)))
    sx = int(np.clip(seed_x, 0, (b.shape[1] - 1)))
    lab = int(labeled[sy, sx])
    if (lab <= 0):
        ys, xs = np.where(b)
        if (xs.size == 0):
            return np.zeros_like(b, dtype = bool)
        d2 = (((xs - sx) ** 2) + ((ys - sy) ** 2))
        k = int(np.argmin(d2))
        if (float(math.sqrt(float(d2[k]))) > float(max_distance_px)):
            return np.zeros_like(b, dtype = bool)
        lab = int(labeled[int(ys[k]), int(xs[k])])
    return ((labeled == lab) if (lab > 0) else np.zeros_like(b, dtype = bool))

def _polygon_area(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype = np.float64)
    if (pts.shape[0] < 3):
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float((0.5 * abs((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))))

def _circular_smooth(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype = np.float64)
    if (vals.size == 0):
        return vals.astype(np.float32)
    if (vals.size < 5):
        return vals.astype(np.float32)
    med = median_filter(vals, size = 5, mode = "wrap")
    sm = gaussian_filter1d(med, sigma = 1.0, mode = "wrap")
    return sm.astype(np.float32)

def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)
    inter = int(np.sum((aa & bb)))
    den = (int(np.sum(aa)) + int(np.sum(bb)))
    return (float(((2.0 * inter) / den)) if (den > 0) else 0.0)

def _contour_from_component(
    patch: np.ndarray,
    outer: np.ndarray,
    component: np.ndarray,
    center_y: float,
    center_x: float,
    pixel_mm: float,
    max_search_mm: float,
    contrast: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    angle_count = 96
    angles = np.linspace(0.0, (2.0 * math.pi), angle_count, endpoint = False)
    radial_mm = np.arange(0.0, (max_search_mm + (0.5 * pixel_mm)), max(pixel_mm, 0.08))
    radial_px = (radial_mm / max(pixel_mm, 1e-6))
    xs = (center_x + (np.cos(angles)[:, None] * radial_px[None, :]))
    ys = (center_y + (np.sin(angles)[:, None] * radial_px[None, :]))
    fy, fx = ys.ravel(), xs.ravel()
    comp_profiles = map_coordinates(
        component.astype(np.float32), [fy, fx], order = 0, mode = "constant", cval = 0.0
    ).reshape(angle_count, -1)
    outer_profiles = map_coordinates(
        outer.astype(np.float32), [fy, fx], order = 0, mode = "constant", cval = 0.0
    ).reshape(angle_count, -1)
    intensity_profiles = map_coordinates(patch, [fy, fx], order = 1, mode = "nearest").reshape(
        angle_count, -1
    )
    gradients = -np.gradient(
        gaussian_filter1d(intensity_profiles, sigma = 1.0, axis = 1, mode = "nearest"), axis = 1
    )
    radii = np.full(angle_count, np.nan, dtype = np.float32)
    valid_ray = np.zeros(angle_count, dtype = bool)
    for i in range(angle_count):
        contiguous: List[int] = []
        for j in range(radial_mm.size):
            if (outer_profiles[i, j] < 0.5):
                break
            if (comp_profiles[i, j] >= 0.5):
                contiguous.append(j)
            elif (j >= 2):
                if (((j + 1) < radial_mm.size) and (comp_profiles[i, (j + 1)] >= 0.5)):
                    contiguous.append(j)
                    continue
                break
        if contiguous:
            radii[i] = float(radial_mm[contiguous[-1]])
            valid_ray[i] = True
            continue
        idx = np.where(((radial_mm >= 0.30) & (outer_profiles[i] >= 0.5)))[0]
        if idx.size:
            edge = int(idx[np.argmax(gradients[i, idx])])
            if (gradients[i, edge] >= max(10.0, (0.14 * contrast))):
                radii[i] = float(radial_mm[edge])
                valid_ray[i] = True
    if not np.any(valid_ray):
        return (
            np.zeros(angle_count, dtype = np.float32),
            np.zeros((0, 2), dtype = np.float32),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    good = np.where(valid_ray)[0]
    missing = np.where(~valid_ray)[0]
    if missing.size:
        radii[missing] = np.interp(
            missing,
            np.concatenate([(good - angle_count), good, (good + angle_count)]),
            np.tile(radii[good], 3),
        )
    med = float(np.nanmedian(radii))
    mad = float(np.nanmedian(np.abs((radii - med))))
    if (mad > 1e-5):
        radii = np.clip(radii, max(0.05, (med - (3.0 * mad))), (med + (3.0 * mad)))
    radii = _circular_smooth(radii)
    contour = np.column_stack([(radii * np.cos(angles)), (radii * np.sin(angles))]).astype(
        np.float32
    )
    area = _polygon_area(contour)
    half = (angle_count // 2)
    diameters = (radii[:half] + radii[half:])
    dmin = (float(np.nanpercentile(diameters, 10.0)) if diameters.size else 0.0)
    dmax = (float(np.nanpercentile(diameters, 90.0)) if diameters.size else 0.0)
    deq = (float((2.0 * math.sqrt((max(area, 0.0) / math.pi)))) if (area > 0) else 0.0)
    return radii, contour, area, dmin, dmax, deq, float(np.mean(valid_ray))

def _section_identity_metrics(
    patch: np.ndarray,
    outer: np.ndarray,
    component: np.ndarray,
    center_y: float,
    center_x: float,
    radii_mm: np.ndarray,
    pixel_mm: float,
    contrast: float,
    tracking_overlap: float,
    tracking_area_ratio: float,
    tracking_center_delta_mm: float,
    calc_fraction: float,
) -> Dict[str, float]:
    image = np.asarray(patch, dtype = np.float32)
    outer_mask = np.asarray(outer).astype(bool)
    comp = np.asarray(component).astype(bool)
    h, w = image.shape
    cy = float(np.clip(center_y, 0.0, (h - 1.0)))
    cx = float(np.clip(center_x, 0.0, (w - 1.0)))
    iy, ix = int(round(cy)), int(round(cx))
    center_inside = (int(bool(comp[iy, ix])) if comp.size else 0)

    if np.any(outer_mask):
        core_distance = (distance_transform_edt(outer_mask) * float(pixel_mm))
        core = (core_distance >= 0.25)
        core_overlap = float((np.sum((comp & core)) / max(int(np.sum(comp)), 1)))
    else:
        core_overlap = 0.0

    radii = np.asarray(radii_mm, dtype = np.float32)
    radii = radii[(np.isfinite(radii) & (radii > 0))]
    if radii.size:
        radial_cv = float((np.std(radii) / max(float(np.mean(radii)), 1e-6)))
        half = (radii.size // 2)
        if (half >= 2):
            diam = (radii[:half] + radii[half : (half + half)])
            d10 = float(np.percentile(diam, 10.0))
            d90 = float(np.percentile(diam, 90.0))
            eccentricity = float((d90 / max(d10, 1e-6)))
        else:
            eccentricity = 1.0
    else:
        radial_cv = float("inf")
        eccentricity = float("inf")

    edge_support = 0.0
    edge_strength = 0.0
    radii_full = np.asarray(radii_mm, dtype = np.float32)
    if ((radii_full.size >= 12) and np.any((np.isfinite(radii_full) & (radii_full > 0)))):
        angles = np.linspace(0.0, (2.0 * math.pi), radii_full.size, endpoint = False)
        r = np.nan_to_num(radii_full, nan = 0.0, posinf = 0.0, neginf = 0.0)
        inside_r = np.maximum(0.10, (r - max(0.22, (1.2 * float(pixel_mm)))))
        outside_r = (r + max(0.24, (1.4 * float(pixel_mm))))
        inside_x = (cx + ((np.cos(angles) * inside_r) / max(float(pixel_mm), 1e-6)))
        inside_y = (cy + ((np.sin(angles) * inside_r) / max(float(pixel_mm), 1e-6)))
        outside_x = (cx + ((np.cos(angles) * outside_r) / max(float(pixel_mm), 1e-6)))
        outside_y = (cy + ((np.sin(angles) * outside_r) / max(float(pixel_mm), 1e-6)))
        inside = map_coordinates(image, [inside_y, inside_x], order = 1, mode = "nearest")
        outside = map_coordinates(image, [outside_y, outside_x], order = 1, mode = "nearest")
        drops = (inside - outside)
        finite = (np.isfinite(drops) & (r > 0))
        if np.any(finite):
            scale = max(float(contrast), 30.0)
            normalized = (drops[finite] / scale)
            edge_support = float(np.mean((normalized >= 0.10)))
            edge_strength = float(np.clip(np.median(normalized), -0.5, 1.5))

    overlap = (float(tracking_overlap) if np.isfinite(tracking_overlap) else 0.50)
    area_ratio = (float(tracking_area_ratio) if np.isfinite(tracking_area_ratio) else 0.50)
    center_delta = (float(tracking_center_delta_mm) if np.isfinite(tracking_center_delta_mm) else 0.0)
    shape_score = float(np.clip((1.0 - (max((radial_cv - 0.18), 0.0) / 0.65)), 0.0, 1.0))
    eccentric_score = float(np.clip((1.0 - (max((eccentricity - 2.0), 0.0) / 4.0)), 0.0, 1.0))
    longitudinal = ((
        (0.42 * np.clip(overlap, 0.0, 1.0)) + (0.33 * np.clip(area_ratio, 0.0, 1.0))
    ) + (0.25 * np.clip((1.0 - (center_delta / 1.20)), 0.0, 1.0)))
    identity = ((
        (
            (
                (
                    (
                        (
                            (0.19 * float(center_inside)) + (0.17 * np.clip((core_overlap / 0.55), 0.0, 1.0))
                        ) + (0.20 * np.clip((edge_support / 0.70), 0.0, 1.0))
                    ) + (0.10 * np.clip(((edge_strength + 0.05) / 0.45), 0.0, 1.0))
                ) + (0.17 * np.clip(longitudinal, 0.0, 1.0))
            ) + (0.09 * shape_score)
        ) + (0.05 * eccentric_score)
    ) + (0.03 * np.clip((1.0 - float(calc_fraction)), 0.0, 1.0)))
    return {
        "gradient_boundary_support": float(np.clip(edge_support, 0.0, 1.0)),
        "gradient_boundary_strength": float(edge_strength),
        "core_overlap_fraction": float(np.clip(core_overlap, 0.0, 1.0)),
        "center_inside_component": int(center_inside),
        "radial_cv": float(radial_cv),
        "eccentricity": float(eccentricity),
        "lumen_identity_score": float(np.clip(identity, 0.0, 1.0)),
    }

def _evaluate_component(
    patch: np.ndarray,
    outer: np.ndarray,
    calc: np.ndarray,
    component: np.ndarray,
    center_y: float,
    center_x: float,
    nominal_center: Tuple[float, float],
    axis_mm: np.ndarray,
    pixel_mm: float,
    lumen_mu: float,
    lumen_sigma: float,
    bg_mu: float,
    contrast: float,
    prior: Optional[PlanePatch],
    tracking_status: str,
    index: int,
    distance_mm: float,
    center_voxel: np.ndarray,
) -> PlanePatch:
    h, w = patch.shape
    cy, cx = nominal_center
    yy, xx = np.indices(patch.shape, dtype = np.float32)
    max_search_mm = min(float(np.max(np.abs(axis_mm))), 5.2)
    radii, contour, area, dmin, dmax, deq, angular_support = _contour_from_component(
        patch, outer, component, center_y, center_x, pixel_mm, max_search_mm, contrast
    )
    comp_area = float(((np.sum(component) * pixel_mm) * pixel_mm))
    agreement = (
        float((1.0 - (abs((comp_area - area)) / max(comp_area, area, 1e-6))))
        if ((comp_area > 0) and (area > 0))
        else 0.0
    )
    shift_mm = float((math.hypot((center_y - cy), (center_x - cx)) * pixel_mm))
    rr = (np.sqrt((((yy - center_y) ** 2) + ((xx - center_x) ** 2))) * pixel_mm)
    region = (rr <= max((deq / 2.0), 0.8))
    calc_fraction = (float(np.mean(calc[region])) if np.any(region) else 0.0)
    cnr = float(((lumen_mu - bg_mu) / max(lumen_sigma, 10.0)))
    outer_fraction = float(np.mean(outer))
    center_hu = float(map_coordinates(patch, [[center_y], [center_x]], order = 1, mode = "nearest")[0])

    overlap = float("nan")
    area_ratio = float("nan")
    center_delta = float("nan")
    continuity_score = 0.50
    if (prior is not None):
        overlap = _mask_dice(component, prior.lumen_component)
        prior_area = float(prior.measurement.lumen_area_mm2)
        if (np.isfinite(prior_area) and (prior_area > 0) and (area > 0)):
            area_ratio = float((min(area, prior_area) / max(area, prior_area)))
        py, px = prior.refined_center_pixel
        center_delta = float((math.hypot((center_y - py), (center_x - px)) * pixel_mm))
        continuity_score = ((
            (0.45 * np.clip(overlap, 0.0, 1.0)) + (0.30 * np.clip((area_ratio if np.isfinite(area_ratio) else 0.0), 0.0, 1.0))
        ) + (
            0.25 * np.clip(
                (1.0 - ((center_delta if np.isfinite(center_delta) else 2.0) / 1.20)), 0.0, 1.0
            )
        ))

    identity = _section_identity_metrics(
        patch,
        outer,
        component,
        center_y,
        center_x,
        radii,
        pixel_mm,
        contrast,
        overlap,
        area_ratio,
        center_delta,
        calc_fraction,
    )
    edge_support = float(identity["gradient_boundary_support"])
    identity_score = float(identity["lumen_identity_score"])
    core_overlap = float(identity["core_overlap_fraction"])
    center_inside = int(identity["center_inside_component"])

    quality = ((
        (
            (
                (
                    (
                        (
                            (0.14 * np.clip(angular_support, 0.0, 1.0)) + (0.16 * np.clip((edge_support / 0.65), 0.0, 1.0))
                        ) + (0.13 * np.clip(agreement, 0.0, 1.0))
                    ) + (0.12 * np.clip((cnr / 4.0), 0.0, 1.0))
                ) + (0.09 * np.clip((1.0 - (shift_mm / 1.7)), 0.0, 1.0))
            ) + (0.10 * np.clip((1.0 - calc_fraction), 0.0, 1.0))
        ) + (0.12 * np.clip(continuity_score, 0.0, 1.0))
    ) + (0.14 * np.clip(identity_score, 0.0, 1.0)))
    unresolved = int(((area <= 0.08) or (deq <= 0.30) or (dmin <= 0.15) or (np.sum(component) < 4)))
    prior_consistent = True
    if (prior is not None):
        severe_centered_rescue = (
            (center_inside == 1)
            and (edge_support >= 0.55)
            and (not np.isfinite(center_delta) or (center_delta <= 0.75))
        )
        prior_consistent = bool(
            (
                (not np.isfinite(center_delta) or (center_delta <= 1.05))
                and (
                    (np.isfinite(overlap) and (overlap >= 0.055))
                    or (np.isfinite(area_ratio) and (area_ratio >= 0.20))
                    or severe_centered_rescue
                )
            )
        )
    strict_valid = bool(
        (
            not unresolved
            and (angular_support >= 0.62)
            and (edge_support >= 0.23)
            and (agreement >= 0.34)
            and (cnr >= 1.10)
            and (shift_mm <= 1.75)
            and (core_overlap >= 0.10)
            and (identity_score >= 0.46)
            and ((center_inside == 1) or (core_overlap >= 0.34))
            and prior_consistent
        )
    )
    flags: List[str] = []
    if (angular_support < 0.62):
        flags.append("low_angular_contour_coverage")
    if (edge_support < 0.23):
        flags.append("weak_gradient_boundary_support")
    if (agreement < 0.34):
        flags.append("component_contour_disagreement")
    if (cnr < 1.10):
        flags.append("low_lumen_contrast")
    if (shift_mm > 1.75):
        flags.append("large_center_shift")
    if (core_overlap < 0.10):
        flags.append("weak_branch_core_overlap")
    if (center_inside == 0):
        flags.append("propagated_center_outside_component")
    if (identity_score < 0.46):
        flags.append("low_target_lumen_identity")
    if not prior_consistent:
        flags.append("longitudinal_identity_break")
    if ((prior is not None) and np.isfinite(area_ratio) and (area_ratio < 0.22)):
        flags.append("longitudinal_area_jump")
    if (calc_fraction > 0.25):
        flags.append("calcification_blooming_risk")
    if unresolved:
        flags.append("unresolved_lumen")
    if not flags:
        flags.append("pass")

    out_area = (float(area) if strict_valid else float("nan"))
    out_deq = (float(deq) if strict_valid else float("nan"))
    out_dmin = (float(dmin) if strict_valid else float("nan"))
    out_dmax = (float(dmax) if strict_valid else float("nan"))
    tracking_score = float(
        np.clip(
            (((0.55 * quality) + (0.25 * continuity_score)) + (0.20 * identity_score)), 0.0, 1.0
        )
    )
    measurement = CrossSectionMeasurement(
        index = int(index),
        distance_mm = float(distance_mm),
        center_voxel_x = float(center_voxel[0]),
        center_voxel_y = float(center_voxel[1]),
        center_voxel_z = float(center_voxel[2]),
        lumen_area_mm2 = out_area,
        equivalent_diameter_mm = out_deq,
        min_diameter_mm = out_dmin,
        max_diameter_mm = out_dmax,
        lumen_mean_hu = float(lumen_mu),
        background_mean_hu = float(bg_mu),
        lumen_noise_hu = float(lumen_sigma),
        contrast_to_noise = float(cnr),
        boundary_support = float(angular_support),
        contour_component_agreement = float(agreement),
        calcification_fraction = float(calc_fraction),
        center_shift_mm = float(shift_mm),
        quality_score = float(np.clip(quality, 0.0, 1.0)),
        valid = int(strict_valid),
        quality_flags = ";".join(dict.fromkeys(flags)),
        refined_center_u_mm = float(((center_x - cx) * pixel_mm)),
        refined_center_v_mm = float(((center_y - cy) * pixel_mm)),
        component_area_mm2 = float(comp_area),
        tracking_overlap = float(overlap),
        tracking_area_ratio = float(area_ratio),
        tracking_center_delta_mm = float(center_delta),
        tracking_score = tracking_score,
        tracking_status = str(tracking_status),
        outer_mask_fraction = float(outer_fraction),
        center_hu = float(center_hu),
        unresolved_lumen = int(unresolved),
        gradient_boundary_support = float(edge_support),
        core_overlap_fraction = float(core_overlap),
        center_inside_component = int(center_inside),
        radial_cv = float(identity["radial_cv"]),
        eccentricity = float(identity["eccentricity"]),
        lumen_identity_score = float(identity_score),
    )
    return PlanePatch(
        image = np.asarray(patch, dtype = np.float32),
        outer_mask = np.asarray(outer).astype(bool),
        calc_mask = np.asarray(calc).astype(bool),
        axis_mm = np.asarray(axis_mm, dtype = np.float32),
        center_pixel = (float(cy), float(cx)),
        refined_center_pixel = (float(center_y), float(center_x)),
        lumen_component = np.asarray(component).astype(bool),
        contour_xy_mm = np.asarray(contour, dtype = np.float32),
        radii_mm = np.asarray(radii, dtype = np.float32),
        measurement = measurement,
    )
