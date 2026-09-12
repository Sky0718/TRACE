from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_opening

from . import contours as contours_ops

from .models import (
    ALGORITHM_VERSION,
    CenterlineResult,
    CrossSectionMeasurement,
    PlanePatch,
    estimate_branch_lumen_reference,
    sample_plane_batch,
)

def segment_lumen_patch(
    image_patch: np.ndarray,
    outer_mask_patch: np.ndarray,
    axis_mm: np.ndarray,
    branch_lumen_reference: Mapping[str, float],
    calc_threshold_hu: float,
    index: int,
    distance_mm: float,
    center_voxel: np.ndarray,
    pixel_mm: float,
    prior: Optional[PlanePatch] = None,
    tracking_status: str = "independent",
) -> PlanePatch:
    patch = np.asarray(image_patch, dtype = np.float32)
    outer = np.asarray(outer_mask_patch).astype(bool)
    h, w = patch.shape
    cy, cx = ((h - 1) / 2.0), ((w - 1) / 2.0)
    yy, xx = np.indices(patch.shape, dtype = np.float32)
    calc = ((patch >= float(calc_threshold_hu)) & outer)
    branch_mu = float(branch_lumen_reference.get("mean_hu", 250.0))
    branch_sigma = float(branch_lumen_reference.get("std_hu", 40.0))

    if ((prior is not None) and (int(prior.measurement.valid) == 1)):
        py, px = prior.refined_center_pixel
        py = float(np.clip(py, 0, (h - 1)))
        px = float(np.clip(px, 0, (w - 1)))
    else:
        py, px = cy, cx

    r_nominal = (np.sqrt((((yy - cy) ** 2) + ((xx - cx) ** 2))) * pixel_mm)
    central = ((((r_nominal <= 1.8) & outer) & (~calc)) & np.isfinite(patch))
    vals = patch[central]
    seed_threshold = max(
        (float(np.percentile(vals, 55.0)) if vals.size else branch_mu),
        (branch_mu - (1.7 * branch_sigma)),
    )
    bright = ((central & (patch >= seed_threshold)) & (patch < 1200.0))
    if np.any(bright):
        weights = np.clip((patch[bright] - max(0.0, (branch_mu - (2.0 * branch_sigma)))), 1.0, None)
        iy = float(np.average(yy[bright], weights = weights))
        ix = float(np.average(xx[bright], weights = weights))
    else:
        iy, ix = cy, cx
    if (prior is not None):
        iy = ((0.70 * py) + (0.30 * iy))
        ix = ((0.70 * px) + (0.30 * ix))
        delta_px = math.hypot((iy - py), (ix - px))
        max_delta_px = (1.1 / max(pixel_mm, 1e-6))
        if (delta_px > max_delta_px):
            scale = (max_delta_px / max(delta_px, 1e-6))
            iy = (py + ((iy - py) * scale))
            ix = (px + ((ix - px) * scale))

    rr = (np.sqrt((((yy - iy) ** 2) + ((xx - ix) ** 2))) * pixel_mm)
    core_vals = patch[((((rr <= 0.85) & outer) & (~calc)) & np.isfinite(patch))]
    if core_vals.size:
        local = core_vals[(core_vals >= np.percentile(core_vals, 40.0))]
        local_mu, local_sigma = contours_ops._robust_location_scale(local, 8.0)
    else:
        local_mu, local_sigma = branch_mu, branch_sigma
    lumen_mu = float(np.clip(((0.72 * local_mu) + (0.28 * branch_mu)), 25.0, 1200.0))
    lumen_sigma = float(max(((0.72 * local_sigma) + (0.28 * branch_sigma)), 8.0))
    ann = patch[(((rr >= 2.0) & (rr <= 4.8)) & np.isfinite(patch))]
    if ann.size:
        bg_vals = ann[(ann <= np.percentile(ann, 65.0))]
        bg_mu, _ = contours_ops._robust_location_scale(bg_vals, 10.0)
    else:
        finite = patch[np.isfinite(patch)]
        bg_mu = (float(np.percentile(finite, 30.0)) if finite.size else -50.0)
    contrast = max((lumen_mu - bg_mu), 1.0)
    base_threshold = max((bg_mu + (0.48 * contrast)), (lumen_mu - (2.6 * lumen_sigma)))
    base_threshold = float(np.clip(base_threshold, 25.0, min(max((lumen_mu - 4.0), 30.0), 850.0)))
    max_search_mm = min(float(np.max(np.abs(axis_mm))), 5.2)

    thresholds = [
        base_threshold,
        max(25.0, (base_threshold - (0.12 * contrast))),
        min(850.0, (base_threshold + (0.10 * contrast))),
        max(25.0, (lumen_mu - (3.2 * lumen_sigma))),
    ]
    unique_thresholds: List[float] = []
    for threshold in thresholds:
        if not any((abs((threshold - old)) < 2.0) for old in unique_thresholds):
            unique_thresholds.append(float(threshold))

    ranked: List[Tuple[float, int, float, np.ndarray, float, float]] = []
    prior_area = (
        float(prior.measurement.component_area_mm2) if (prior is not None) else float("nan")
    )
    prior_component = (prior.lumen_component if (prior is not None) else None)
    for attempt, threshold in enumerate(unique_thresholds):
        candidate = ((((patch >= threshold) & outer) & (rr <= max_search_mm)) & (~calc))
        candidate = binary_closing(candidate, structure = np.ones((3, 3), dtype = bool), iterations = 1)
        candidate = binary_opening(candidate, structure = np.ones((2, 2), dtype = bool), iterations = 1)
        component = binary_fill_holes(
            contours_ops._component_from_seed(candidate, int(round(iy)), int(round(ix)), max_distance_px = 8.0)
        )
        voxel_count = int(np.sum(component))
        if (voxel_count < 4):
            continue
        coords = np.argwhere(component)
        component_weights = np.clip(((patch[component] - threshold) + 1.0), 1.0, None)
        ry = float(np.average(coords[:, 0], weights = component_weights))
        rx = float(np.average(coords[:, 1], weights = component_weights))
        if (prior is not None):
            dpx = math.hypot((ry - py), (rx - px))
            max_dpx = (1.2 / max(pixel_mm, 1e-6))
            if (dpx > max_dpx):
                scale = (max_dpx / max(dpx, 1e-6))
                ry = (py + ((ry - py) * scale))
                rx = (px + ((rx - px) * scale))

        area_mm2 = float(((voxel_count * pixel_mm) * pixel_mm))
        center_delta_mm = float(
            (
                math.hypot(
                    (ry - (py if (prior is not None) else cy)),
                    (rx - (px if (prior is not None) else cx)),
                ) * pixel_mm
            )
        )
        centrality = float(np.exp((-0.5 * ((center_delta_mm / 1.10) ** 2))))
        comp_vals = patch[component]
        comp_mu = (float(np.median(comp_vals)) if comp_vals.size else bg_mu)
        intensity_score = float(np.clip(((comp_mu - bg_mu) / max(contrast, 20.0)), 0.0, 1.2))
        calc_fraction = (float(np.mean(calc[component])) if np.any(component) else 1.0)
        calc_score = float(np.clip((1.0 - calc_fraction), 0.0, 1.0))
        plausible_area = float(
            np.exp((-0.5 * (((math.log(max(area_mm2, 0.05)) - math.log(4.5)) / 1.45) ** 2)))
        )
        if ((prior_component is not None) and np.any(prior_component)):
            overlap = contours_ops._mask_dice(component, prior_component)
            if (np.isfinite(prior_area) and (prior_area > 0)):
                area_ratio = float((min(area_mm2, prior_area) / max(area_mm2, prior_area)))
            else:
                area_ratio = 0.5
            cheap_score = ((
                (
                    (
                        (
                            (0.33 * np.clip(overlap, 0.0, 1.0)) + (0.24 * np.clip(area_ratio, 0.0, 1.0))
                        ) + (0.18 * centrality)
                    ) + (0.15 * np.clip(intensity_score, 0.0, 1.0))
                ) + (0.06 * calc_score)
            ) + (0.04 * plausible_area))
        else:
            cheap_score = ((
                (
                    ((0.39 * centrality) + (0.29 * np.clip(intensity_score, 0.0, 1.0))) + (0.16 * plausible_area)
                ) + (0.10 * calc_score)
            ) + (0.06 * float(np.clip((voxel_count / 24.0), 0.0, 1.0))))
        ranked.append(
            (float(cheap_score), int(attempt), float(threshold), component, float(ry), float(rx))
        )

    ranked.sort(key = lambda item: item[0], reverse = True)
    best: Optional[PlanePatch] = None
    best_score = -1e9
    evaluation_limit = min(4, len(ranked))
    for rank_index, (_, attempt, threshold, component, ry, rx) in enumerate(
        ranked[:evaluation_limit]
    ):
        plane = contours_ops._evaluate_component(
            patch,
            outer,
            calc,
            component,
            ry,
            rx,
            (cy, cx),
            axis_mm,
            pixel_mm,
            lumen_mu,
            lumen_sigma,
            bg_mu,
            contrast,
            prior,
            (tracking_status if (attempt == 0) else f"{tracking_status}_rescue_{attempt}"),
            index,
            distance_mm,
            center_voxel,
        )
        score = float(plane.measurement.tracking_score)
        if (int(plane.measurement.valid) == 1):
            score += 0.25
        if ((prior is not None) and np.isfinite(plane.measurement.tracking_area_ratio)):
            if ((plane.measurement.tracking_area_ratio < 0.22) and (
                plane.measurement.boundary_support < 0.75
            )):
                score -= 0.25
        if (score > best_score):
            best_score = score
            best = plane
        if ((rank_index == 0) and (int(plane.measurement.valid) == 1) and (score >= 0.82)):
            break

    if (best is not None):
        return best

    measurement = CrossSectionMeasurement(
        index = int(index),
        distance_mm = float(distance_mm),
        center_voxel_x = float(center_voxel[0]),
        center_voxel_y = float(center_voxel[1]),
        center_voxel_z = float(center_voxel[2]),
        lumen_area_mm2 = float("nan"),
        equivalent_diameter_mm = float("nan"),
        min_diameter_mm = float("nan"),
        max_diameter_mm = float("nan"),
        lumen_mean_hu = float(lumen_mu),
        background_mean_hu = float(bg_mu),
        lumen_noise_hu = float(lumen_sigma),
        contrast_to_noise = float(((lumen_mu - bg_mu) / max(lumen_sigma, 10.0))),
        boundary_support = 0.0,
        contour_component_agreement = 0.0,
        calcification_fraction = (float(np.mean(calc[outer])) if np.any(outer) else 0.0),
        center_shift_mm = float((math.hypot((iy - cy), (ix - cx)) * pixel_mm)),
        quality_score = 0.0,
        valid = 0,
        quality_flags = "unresolved_lumen;segmentation_failed",
        refined_center_u_mm = float(((ix - cx) * pixel_mm)),
        refined_center_v_mm = float(((iy - cy) * pixel_mm)),
        component_area_mm2 = float("nan"),
        tracking_overlap = float("nan"),
        tracking_area_ratio = float("nan"),
        tracking_center_delta_mm = float("nan"),
        tracking_score = 0.0,
        tracking_status = f"{tracking_status}_failed",
        outer_mask_fraction = float(np.mean(outer)),
        center_hu = float(
            patch[int(round(np.clip(iy, 0, (h - 1)))), int(round(np.clip(ix, 0, (w - 1))))]
        ),
        unresolved_lumen = 1,
    )
    return PlanePatch(
        patch,
        outer,
        calc,
        np.asarray(axis_mm),
        (cy, cx),
        (iy, ix),
        np.zeros_like(outer),
        np.zeros((0, 2), dtype = np.float32),
        np.zeros(0, dtype = np.float32),
        measurement,
    )

def _dilate_plane_search_mask(
    mask_patch: np.ndarray, pixel_mm: float, radius_mm: float = 1.25
) -> np.ndarray:
    mask = np.asarray(mask_patch).astype(bool)
    radius_px = max(1, int(math.ceil((float(radius_mm) / max(float(pixel_mm), 1e-6)))))
    yy, xx = np.ogrid[-radius_px : (radius_px + 1), -radius_px : (radius_px + 1)]
    structure = (((xx * xx) + (yy * yy)) <= (radius_px * radius_px))
    return binary_dilation(mask, structure = structure, iterations = 1)

def _sample_all_patches(
    image: np.ndarray,
    branch_mask: np.ndarray,
    centerline: CenterlineResult,
    spacing: Sequence[float],
    pixel_mm: float,
    half_width_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_patches, axis = sample_plane_batch(
        image,
        centerline.path_physical_mm,
        centerline.normal_u,
        centerline.normal_v,
        spacing,
        half_width_mm,
        pixel_mm,
        1,
        -1000.0,
    )
    mask_patches, _ = sample_plane_batch(
        np.asarray(branch_mask).astype(bool),
        centerline.path_physical_mm,
        centerline.normal_u,
        centerline.normal_v,
        spacing,
        half_width_mm,
        pixel_mm,
        0,
        0.0,
    )
    return (
        np.asarray(image_patches, dtype = np.float32),
        np.asarray((mask_patches >= 0.5), dtype = bool),
        np.asarray(axis, dtype = np.float32),
    )

def _longest_true_run(mask: np.ndarray, step_mm: float) -> Tuple[int, float]:
    m = np.asarray(mask).astype(bool)
    best = current = 0
    for value in m:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best), float((max((best - 1), 0) * step_mm))

def _choose_better_section(primary: PlanePatch, fallback: PlanePatch) -> PlanePatch:

    def key(patch: PlanePatch) -> Tuple[int, float, float, float]:
        m = patch.measurement
        return (
            int(m.valid),
            float(m.tracking_score),
            float(m.quality_score),
            float(m.boundary_support),
        )

    return (primary if (key(primary) >= key(fallback)) else fallback)

def _repair_short_invalid_gaps(
    patches: List[PlanePatch],
    image_patches: np.ndarray,
    mask_patches: np.ndarray,
    axis: np.ndarray,
    reference: Mapping[str, float],
    calc_threshold_hu: float,
    centerline: CenterlineResult,
    pixel_mm: float,
    maximum_gap_mm: float = 1.8,
) -> None:
    n = len(patches)
    if (n == 0):
        return
    step = (
        float(np.median(np.diff(centerline.cumulative_mm)))
        if (centerline.cumulative_mm.size >= 2)
        else 0.55
    )
    max_count = max(1, int(round((float(maximum_gap_mm) / max(step, 0.2)))))
    valid = np.asarray([(int(p.measurement.valid) == 1) for p in patches], dtype = bool)
    i = 0
    while (i < n):
        if valid[i]:
            i += 1
            continue
        j = i
        while (((j + 1) < n) and not valid[(j + 1)]):
            j += 1
        gap_count = ((j - i) + 1)
        left = (i - 1)
        right = (j + 1)
        if (
            (gap_count <= max_count)
            and (left >= 0)
            and (right < n)
            and valid[left]
            and valid[right]
        ):
            prior = patches[left]
            forward: Dict[int, PlanePatch] = {}
            for k in range(i, (j + 1)):
                outer = _dilate_plane_search_mask(mask_patches[k], pixel_mm, radius_mm = 0.90)
                trial = segment_lumen_patch(
                    image_patches[k],
                    outer,
                    axis,
                    reference,
                    calc_threshold_hu,
                    k,
                    float(centerline.cumulative_mm[k]),
                    centerline.path_voxel[k],
                    pixel_mm,
                    prior = prior,
                    tracking_status = "short_gap_forward_rescue",
                )
                forward[k] = trial
                if (int(trial.measurement.valid) == 1):
                    prior = trial
            prior = patches[right]
            backward: Dict[int, PlanePatch] = {}
            for k in range(j, (i - 1), -1):
                outer = _dilate_plane_search_mask(mask_patches[k], pixel_mm, radius_mm = 0.90)
                trial = segment_lumen_patch(
                    image_patches[k],
                    outer,
                    axis,
                    reference,
                    calc_threshold_hu,
                    k,
                    float(centerline.cumulative_mm[k]),
                    centerline.path_voxel[k],
                    pixel_mm,
                    prior = prior,
                    tracking_status = "short_gap_backward_rescue",
                )
                backward[k] = trial
                if (int(trial.measurement.valid) == 1):
                    prior = trial
            for k in range(i, (j + 1)):
                chosen = _choose_better_section(forward[k], backward[k])
                if (int(chosen.measurement.valid) == 1):
                    chosen.measurement.tracking_status += "_accepted"
                    patches[k] = chosen
                    valid[k] = True
        i = (j + 1)

def _measure_cross_sections_once(
    image: np.ndarray,
    branch_mask: np.ndarray,
    centerline: CenterlineResult,
    spacing: Sequence[float],
    calc_threshold_hu: float,
    pixel_mm: float,
    half_width_mm: float,
    keep_patches: bool,
) -> Tuple[pd.DataFrame, Dict[int, PlanePatch]]:
    if not centerline.ok:
        return pd.DataFrame(), {}
    reference = estimate_branch_lumen_reference(image, centerline, spacing)
    image_patches, mask_patches, axis = _sample_all_patches(
        image, branch_mask, centerline, spacing, pixel_mm, half_width_mm
    )
    n = int(image_patches.shape[0])
    if (n == 0):
        return pd.DataFrame(), {}

    stride = max(1, int(math.ceil((n / 48.0))))
    anchors = set(range(0, n, stride))
    anchors.update({0, max(0, (n // 4)), max(0, (n // 2)), min((n - 1), ((3 * n) // 4)), (n - 1)})
    anchor_patches: Dict[int, PlanePatch] = {}
    for i in sorted(anchors):
        outer = _dilate_plane_search_mask(mask_patches[i], pixel_mm, radius_mm = 0.90)
        anchor_patches[i] = segment_lumen_patch(
            image_patches[i],
            outer,
            axis,
            reference,
            calc_threshold_hu,
            i,
            float(centerline.cumulative_mm[i]),
            centerline.path_voxel[i],
            pixel_mm,
            prior = None,
            tracking_status = "independent_anchor",
        )

    central_lo = int((0.08 * n))
    central_hi = max((central_lo + 1), int((0.92 * n)))
    valid_anchor = [
        i
        for i, patch in anchor_patches.items()
        if ((central_lo <= i < central_hi) and (int(patch.measurement.valid) == 1))
    ]
    if not valid_anchor:
        valid_anchor = [
            i for i, patch in anchor_patches.items() if (int(patch.measurement.valid) == 1)
        ]
    if valid_anchor:
        anchor = max(
            valid_anchor,
            key = lambda i: (
                float(anchor_patches[i].measurement.quality_score),
                float(anchor_patches[i].measurement.boundary_support),
                float(anchor_patches[i].measurement.contrast_to_noise),
            ),
        )
    else:
        anchor = min(anchor_patches, key = lambda i: abs((i - (n // 2))))

    tracked: List[Optional[PlanePatch]] = ([None] * n)
    tracked[anchor] = anchor_patches[anchor]
    for direction in (1, -1):
        anchor_patch = tracked[anchor]
        prior = (
            anchor_patch
            if ((anchor_patch is not None) and (int(anchor_patch.measurement.valid) == 1))
            else None
        )
        gap_count = 0
        i = (anchor + direction)
        while (0 <= i < n):
            outer = _dilate_plane_search_mask(mask_patches[i], pixel_mm, radius_mm = 0.90)
            candidate = segment_lumen_patch(
                image_patches[i],
                outer,
                axis,
                reference,
                calc_threshold_hu,
                i,
                float(centerline.cumulative_mm[i]),
                centerline.path_voxel[i],
                pixel_mm,
                prior = prior,
                tracking_status = ("forward_track" if (direction > 0) else "backward_track"),
            )
            need_rescue = (
                (int(candidate.measurement.valid) == 0)
                or (float(candidate.measurement.quality_score) < 0.46)
                or (float(candidate.measurement.boundary_support) < 0.42)
            )
            if need_rescue:
                rescue = anchor_patches.get(i)
                if (rescue is None):
                    rescue = segment_lumen_patch(
                        image_patches[i],
                        outer,
                        axis,
                        reference,
                        calc_threshold_hu,
                        i,
                        float(centerline.cumulative_mm[i]),
                        centerline.path_voxel[i],
                        pixel_mm,
                        prior = None,
                        tracking_status = "independent_rescue",
                    )
                candidate = _choose_better_section(candidate, rescue)
                if (candidate is rescue):
                    candidate.measurement.tracking_status = "independent_rescue_selected"
            tracked[i] = candidate
            if (int(candidate.measurement.valid) == 1):
                prior = candidate
                gap_count = 0
            else:
                gap_count += 1
                maximum_gap_count = max(
                    2,
                    int(
                        round(
                            (
                                1.8 / max(
                                    (
                                        float(np.median(np.diff(centerline.cumulative_mm)))
                                        if (centerline.cumulative_mm.size > 1)
                                        else 0.55
                                    ),
                                    0.2,
                                )
                            )
                        )
                    ),
                )
                if (gap_count > maximum_gap_count):
                    prior = None
            i += direction

    patches_list: List[PlanePatch] = []
    for i in range(n):
        patch = tracked[i]
        if (patch is None):
            patch = anchor_patches.get(i)
        if (patch is None):
            outer = _dilate_plane_search_mask(mask_patches[i], pixel_mm, radius_mm = 0.90)
            patch = segment_lumen_patch(
                image_patches[i],
                outer,
                axis,
                reference,
                calc_threshold_hu,
                i,
                float(centerline.cumulative_mm[i]),
                centerline.path_voxel[i],
                pixel_mm,
                prior = None,
                tracking_status = "independent_unreached",
            )
        patches_list.append(patch)

    _repair_short_invalid_gaps(
        patches_list,
        image_patches,
        mask_patches,
        axis,
        reference,
        calc_threshold_hu,
        centerline,
        pixel_mm,
    )
    step_mm = (
        float(np.median(np.diff(centerline.cumulative_mm)))
        if (centerline.cumulative_mm.size >= 2)
        else 0.55
    )
    _invalidate_longitudinal_outliers(patches_list, step_mm)

    rows: List[Dict[str, Any]] = []
    patches: Dict[int, PlanePatch] = {}
    valid_arr = np.asarray([(int(p.measurement.valid) == 1) for p in patches_list], dtype = bool)
    run_id = np.full(n, -1, dtype = int)
    current = -1
    for i in range(n):
        if valid_arr[i]:
            if ((i == 0) or not valid_arr[(i - 1)]):
                current += 1
            run_id[i] = current
    run_lengths: Dict[int, int] = {
        int(rid): int(np.sum((run_id == rid))) for rid in np.unique(run_id[(run_id >= 0)])
    }
    for i, plane in enumerate(patches_list):
        row = asdict(plane.measurement)
        rid = int(run_id[i])
        count = (int(run_lengths.get(rid, 0)) if (rid >= 0) else 0)
        navigation_area_mm2 = float(((np.sum(mask_patches[i]) * pixel_mm) * pixel_mm))
        navigation_diameter_mm = (
            float((2.0 * math.sqrt((max(navigation_area_mm2, 0.0) / math.pi))))
            if (navigation_area_mm2 > 0)
            else float("nan")
        )
        row.update(
            {
                "navigation_mask_area_mm2": navigation_area_mm2,
                "navigation_mask_equivalent_diameter_mm": navigation_diameter_mm,
                "branch_lumen_reference_hu": float(reference["mean_hu"]),
                "branch_lumen_reference_std_hu": float(reference["std_hu"]),
                "valid_run_id": rid,
                "valid_run_count": count,
                "valid_run_mm": float((max((count - 1), 0) * step_mm)),
                "algorithm_version": ALGORITHM_VERSION,
            }
        )
        rows.append(row)
        if keep_patches:
            patches[i] = plane
    return pd.DataFrame(rows), patches

def _invalidate_longitudinal_outliers(patches: List[PlanePatch], step_mm: float) -> None:
    n = len(patches)
    if (n == 0):
        return
    valid = np.asarray([(int(p.measurement.valid) == 1) for p in patches], dtype = bool)
    diam = np.asarray([p.measurement.min_diameter_mm for p in patches], dtype = float)
    area = np.asarray([p.measurement.lumen_area_mm2 for p in patches], dtype = float)
    identity = np.asarray(
        [getattr(p.measurement, "lumen_identity_score", 0.0) for p in patches], dtype = float
    )
    edge = np.asarray(
        [getattr(p.measurement, "gradient_boundary_support", 0.0) for p in patches], dtype = float
    )
    tracking = np.asarray([p.measurement.tracking_score for p in patches], dtype = float)
    to_invalidate: set[int] = set()

    for i in range(n):
        if (not valid[i] or not np.isfinite(diam[i]) or (diam[i] <= 0)):
            continue
        if ((identity[i] < 0.43) or (edge[i] < 0.18)):
            to_invalidate.add(i)
            continue
        if ((tracking[i] < 0.38) and (identity[i] < 0.60)):
            to_invalidate.add(i)
            continue
        radius = max(4, int(round((2.4 / max(float(step_mm), 0.2)))))
        lo, hi = max(0, (i - radius)), min(n, ((i + radius) + 1))
        neigh = np.arange(lo, hi)
        neigh = neigh[
            ((((neigh != i) & valid[neigh]) & np.isfinite(diam[neigh])) & (diam[neigh] > 0))
        ]
        if (neigh.size < 4):
            continue
        local_d = float(np.median(diam[neigh]))
        local_a_vals = area[neigh][(np.isfinite(area[neigh]) & (area[neigh] > 0))]
        local_a = (float(np.median(local_a_vals)) if local_a_vals.size else float("nan"))
        ratio_d = float((diam[i] / max(local_d, 1e-6)))
        ratio_a = (
            float((area[i] / max(local_a, 1e-6)))
            if (np.isfinite(local_a) and np.isfinite(area[i]))
            else 1.0
        )
        concordant = []
        for j in ((i - 2), (i - 1), (i + 1), (i + 2)):
            if ((0 <= j < n) and valid[j] and np.isfinite(diam[j]) and (diam[j] > 0)):
                concordant.append(
                    (
                        ((diam[j] / max(local_d, 1e-6)) < 0.72)
                        and (identity[j] >= 0.45)
                        and (edge[j] >= 0.18)
                    )
                )
        sustained = (sum(bool(v) for v in concordant) >= 1)
        if ((
            (ratio_d < 0.40) or (ratio_d > 2.35) or (ratio_a < 0.20) or (ratio_a > 3.2)
        ) and not sustained):
            to_invalidate.add(i)
        elif (((ratio_d < 0.55) or (ratio_a < 0.36)) and (identity[i] < 0.58) and not sustained):
            to_invalidate.add(i)

    for i in sorted(to_invalidate):
        m = patches[i].measurement
        m.valid = 0
        m.lumen_area_mm2 = float("nan")
        m.equivalent_diameter_mm = float("nan")
        m.min_diameter_mm = float("nan")
        m.max_diameter_mm = float("nan")
        m.quality_flags = ";".join(
            dict.fromkeys(
                (m.quality_flags + ";longitudinal_identity_outlier").strip(";").split(";")
            )
        )
        m.tracking_status = (str(m.tracking_status) + "_identity_invalidated")

def _robust_profile_noise_ratio(
    values: np.ndarray, reference: np.ndarray, valid: np.ndarray, step_mm: float
) -> float:
    vals = np.asarray(values, dtype = float)
    refv = np.asarray(reference, dtype = float)
    ok = ((
        ((np.asarray(valid).astype(bool) & np.isfinite(vals)) & np.isfinite(refv)) & (vals > 0)
    ) & (refv > 0))
    if (int(np.sum(ok)) < 8):
        return 0.08
    ratio = np.full(vals.size, np.nan, dtype = float)
    ratio[ok] = (vals[ok] / refv[ok])
    good = ratio[ok]
    if (good.size < 8):
        return 0.08
    diffs = np.abs(np.diff(good))
    if (diffs.size == 0):
        return 0.05
    base = (float(np.percentile(diffs, 35.0)) / math.sqrt(2.0))
    return float(np.clip(max((base * 1.4826), 0.025), 0.025, 0.20))
