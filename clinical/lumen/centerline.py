from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label as ndi_label
from scipy.spatial import cKDTree
import root_topology_backend_v2 as ref

from . import sections as sections_ops

from .models import (
    CenterlineResult,
    DEFAULT_CENTERLINE_STEP_MM,
    DEFAULT_CROSS_SECTION_PIXEL_MM,
    DEFAULT_PATCH_HALF_WIDTH_MM,
    PARENT_BRANCHES,
    PlanePatch,
    RootAssignment,
    base_branch_name,
    parallel_transport_frames,
)

def parent_contact_root(
    branch: str,
    branch_mask: np.ndarray,
    branch_masks: Mapping[str, np.ndarray],
    spacing: Sequence[float],
    contact_distance_mm: float = 1.1,
) -> Optional[RootAssignment]:
    base = base_branch_name(branch)
    parents = PARENT_BRANCHES.get(base, ())
    if not parents:
        return None
    child = np.asarray(branch_mask).astype(bool)
    child_coords_all = np.argwhere(child)
    if (child_coords_all.shape[0] == 0):
        return None
    max_child = 120_000
    child_coords = (
        child_coords_all
        if (child_coords_all.shape[0] <= max_child)
        else child_coords_all[
            np.linspace(0, (child_coords_all.shape[0] - 1), max_child, dtype = np.int64)
        ]
    )
    sp = np.asarray(spacing, dtype = np.float64)
    for parent_base in parents:
        parent_key = next((k for k in branch_masks if (base_branch_name(k) == parent_base)), None)
        if (parent_key is None):
            continue
        parent_coords_all = np.argwhere(np.asarray(branch_masks[parent_key]).astype(bool))
        if (parent_coords_all.shape[0] == 0):
            continue
        max_parent = 120_000
        parent_coords = (
            parent_coords_all
            if (parent_coords_all.shape[0] <= max_parent)
            else parent_coords_all[
                np.linspace(0, (parent_coords_all.shape[0] - 1), max_parent, dtype = np.int64)
            ]
        )
        tree = cKDTree((parent_coords.astype(np.float64) * sp[None, :]))
        distances, _ = tree.query((child_coords.astype(np.float64) * sp[None, :]), k = 1, workers = -1)
        keep = (distances <= float(contact_distance_mm))
        if not np.any(keep):
            continue
        contact = child_coords[keep]
        contact_dist = distances[keep]
        pad = np.maximum(np.ceil((4.0 / np.maximum(sp, 1e-6))).astype(int), 3)
        lo = np.maximum((contact.min(axis = 0) - pad), 0)
        hi = np.minimum(((contact.max(axis = 0) + pad) + 1), np.asarray(child.shape))
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        radius_crop = distance_transform_edt(
            child[sl], sampling = tuple(float(v) for v in spacing)
        ).astype(np.float32)
        local = (contact - lo[None, :])
        radii = radius_crop[local[:, 0], local[:, 1], local[:, 2]]
        score = (radii - (0.35 * contact_dist))
        best = contact[int(np.argmax(score))].astype(np.float32)
        root_radius = float(max(radii[int(np.argmax(score))], 0.5))
        min_mm = (3.0 if (base in ("D1", "D2", "RAMUS", "RAD")) else 4.0)
        max_mm = (7.0 if (base in ("D1", "D2", "RAMUS", "RAD")) else 10.0)
        return RootAssignment(
            branch = branch,
            source = f"parent_contact_{parent_base}",
            root_voxel_x = float(best[0]),
            root_voxel_y = float(best[1]),
            root_voxel_z = float(best[2]),
            nearest_distance_mm = float(np.min(contact_dist)),
            parent_branch = str(parent_key),
            seam_exclusion_mm = float(np.clip((2.5 + (2.0 * root_radius)), min_mm, max_mm)),
        )
    return None

def assign_branch_roots_from_labels(
    labels: np.ndarray,
    label_map: Mapping[str, int],
    point_values: Sequence[int],
    spacing: Sequence[float],
    contact_distance_mm: float = 1.25,
    maximum_sample: int = 60_000,
) -> Dict[str, RootAssignment]:
    label_arr = np.asarray(labels)
    sp = np.asarray(spacing, dtype = np.float64)
    sampled: Dict[str, np.ndarray] = {}
    full_counts: Dict[str, int] = {}
    for branch, value in label_map.items():
        coords = np.argwhere((label_arr == int(value))).astype(np.int32, copy = False)
        full_counts[str(branch)] = int(coords.shape[0])
        if (coords.shape[0] > int(maximum_sample)):
            take = np.linspace(0, (coords.shape[0] - 1), int(maximum_sample), dtype = np.int64)
            coords = coords[take]
        sampled[str(branch)] = coords

    point_coords_parts = [
        np.argwhere((label_arr == int(value))).astype(np.int32, copy = False)
        for value in point_values
    ]
    point_coords_parts = [c for c in point_coords_parts if (c.shape[0] > 0)]
    centroids: List[Tuple[int, np.ndarray]] = []
    if point_coords_parts:
        pc = np.concatenate(point_coords_parts, axis = 0)
        lo = np.maximum((pc.min(axis = 0) - 1), 0)
        hi = np.minimum((pc.max(axis = 0) + 2), np.asarray(label_arr.shape))
        local = np.zeros(tuple((hi - lo).astype(int)), dtype = bool)
        lc = (pc - lo[None, :])
        local[lc[:, 0], lc[:, 1], lc[:, 2]] = True
        labs, count = ndi_label(local, structure = np.ones((3, 3, 3), dtype = np.uint8))
        for cid in range(1, (int(count) + 1)):
            cc = np.argwhere((labs == cid))
            if cc.shape[0]:
                centroids.append(
                    (cid, (cc.mean(axis = 0).astype(np.float32) + lo.astype(np.float32)))
                )

    roots: Dict[str, RootAssignment] = {}
    ostial = [
        name
        for name in sampled
        if ((base_branch_name(name) in ("RCA", "LM")) and sampled[name].shape[0])
    ]
    trees: Dict[str, cKDTree] = {
        name: cKDTree((sampled[name].astype(np.float64) * sp[None, :])) for name in ostial
    }
    for cid, centroid in centroids:
        best: Optional[Tuple[float, str, np.ndarray]] = None
        cp = (centroid.astype(np.float64) * sp)
        for name, tree in trees.items():
            dist, k = tree.query(cp, k = 1, workers = 1)
            item = (float(dist), name, sampled[name][int(k)].astype(np.float32))
            if ((best is None) or (item[0] < best[0])):
                best = item
        if ((best is None) or (best[0] > 5.0)):
            continue
        dist, name, voxel = best
        old = roots.get(name)
        if ((old is None) or (dist < old.nearest_distance_mm)):
            roots[name] = RootAssignment(
                branch = name,
                source = "POINT_component",
                root_voxel_x = float(voxel[0]),
                root_voxel_y = float(voxel[1]),
                root_voxel_z = float(voxel[2]),
                nearest_distance_mm = float(dist),
                point_component_id = int(cid),
            )

    parent_trees: Dict[str, cKDTree] = {}
    for child in label_map:
        if ((child in roots) or (sampled.get(child, np.zeros((0, 3))).shape[0] == 0)):
            continue
        base = base_branch_name(child)
        parents = PARENT_BRANCHES.get(base, ())
        if not parents:
            continue
        child_coords = sampled[child]
        child_phys = (child_coords.astype(np.float64) * sp[None, :])
        best_parent: Optional[Tuple[float, str, np.ndarray, np.ndarray]] = None
        for parent_base in parents:
            parent = next(
                (name for name in label_map if (base_branch_name(name) == parent_base)), None
            )
            if ((parent is None) or (sampled.get(parent, np.zeros((0, 3))).shape[0] == 0)):
                continue
            if (parent not in parent_trees):
                parent_trees[parent] = cKDTree((sampled[parent].astype(np.float64) * sp[None, :]))
            distances, _ = parent_trees[parent].query(child_phys, k = 1, workers = 1)
            keep = (np.isfinite(distances) & (distances <= float(contact_distance_mm)))
            if not np.any(keep):
                continue
            item = (float(np.min(distances[keep])), parent, child_coords[keep], distances[keep])
            if ((best_parent is None) or (item[0] < best_parent[0])):
                best_parent = item
        if (best_parent is None):
            continue
        minimum, parent, contacts, contact_distances = best_parent
        order = np.argsort(contact_distances)[: min(800, contact_distances.size)]
        contacts = contacts[order]
        contact_distances = contact_distances[order]
        pad = np.maximum(np.ceil((4.0 / np.maximum(sp, 1e-6))).astype(int), 3)
        lo = np.maximum((contacts.min(axis = 0) - pad), 0)
        hi = np.minimum(((contacts.max(axis = 0) + pad) + 1), np.asarray(label_arr.shape))
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        child_crop = (label_arr[sl] == int(label_map[child]))
        radius = distance_transform_edt(
            child_crop, sampling = tuple(float(v) for v in spacing)
        ).astype(np.float32)
        local = (contacts - lo[None, :])
        radii = radius[local[:, 0], local[:, 1], local[:, 2]]
        k = int(np.argmax((radii - (0.35 * contact_distances))))
        voxel = contacts[k].astype(np.float32)
        root_radius = float(max(radii[k], 0.5))
        min_mm = (3.0 if (base in ("D1", "D2", "RAMUS", "RAD")) else 4.0)
        max_mm = (7.0 if (base in ("D1", "D2", "RAMUS", "RAD")) else 10.0)
        roots[str(child)] = RootAssignment(
            branch = str(child),
            source = f"parent_contact_{base_branch_name(parent)}",
            root_voxel_x = float(voxel[0]),
            root_voxel_y = float(voxel[1]),
            root_voxel_z = float(voxel[2]),
            nearest_distance_mm = float(minimum),
            parent_branch = str(parent),
            seam_exclusion_mm = float(np.clip((2.5 + (2.0 * root_radius)), min_mm, max_mm)),
        )
    return roots

def extract_rooted_centerline(
    branch: str,
    branch_mask: np.ndarray,
    spacing: Sequence[float],
    root_assignment: Optional[RootAssignment] = None,
    step_mm: float = DEFAULT_CENTERLINE_STEP_MM,
) -> CenterlineResult:
    mask = np.asarray(branch_mask).astype(bool)
    coords = np.argwhere(mask)
    if (coords.shape[0] == 0):
        return ref.extract_rooted_centerline(
            branch, mask, spacing, root_assignment = root_assignment, step_mm = step_mm
        )
    sp = np.asarray(spacing, dtype = np.float64)
    pad = np.maximum(np.ceil((3.0 / np.maximum(sp, 1e-6))).astype(int), 2)
    lo = np.maximum((coords.min(axis = 0) - pad), 0).astype(int)
    hi = np.minimum(((coords.max(axis = 0) + pad) + 1), np.asarray(mask.shape)).astype(int)
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    cropped = np.ascontiguousarray(mask[sl])

    local_root: Optional[RootAssignment] = None
    if (root_assignment is not None):
        local_root = RootAssignment(
            **{
                **asdict(root_assignment),
                "root_voxel_x": float((root_assignment.root_voxel_x - lo[0])),
                "root_voxel_y": float((root_assignment.root_voxel_y - lo[1])),
                "root_voxel_z": float((root_assignment.root_voxel_z - lo[2])),
            }
        )
    local = ref.extract_rooted_centerline(
        branch, cropped, spacing, root_assignment = local_root, step_mm = step_mm
    )
    if not local.ok:
        topology = dict(local.topology)
        topology.update(
            {
                "crop_origin_x": int(lo[0]),
                "crop_origin_y": int(lo[1]),
                "crop_origin_z": int(lo[2]),
                "crop_shape": tuple(int(v) for v in cropped.shape),
            }
        )
        return CenterlineResult(
            branch = local.branch,
            path_voxel = local.path_voxel,
            path_physical_mm = local.path_physical_mm,
            cumulative_mm = local.cumulative_mm,
            tangent = local.tangent,
            normal_u = local.normal_u,
            normal_v = local.normal_v,
            method = (str(local.method) + "+cropped_bbox"),
            quality_flag = local.quality_flag,
            topology = topology,
            root_assignment = root_assignment,
        )
    shift_vox = lo.astype(np.float32)
    shift_phys = (shift_vox * np.asarray(spacing, dtype = np.float32))
    topology = dict(local.topology)
    topology.update(
        {
            "crop_origin_x": int(lo[0]),
            "crop_origin_y": int(lo[1]),
            "crop_origin_z": int(lo[2]),
            "crop_shape": tuple(int(v) for v in cropped.shape),
            "full_shape": tuple(int(v) for v in mask.shape),
        }
    )
    return CenterlineResult(
        branch = local.branch,
        path_voxel = (local.path_voxel + shift_vox[None, :]).astype(np.float32),
        path_physical_mm = (local.path_physical_mm + shift_phys[None, :]).astype(np.float32),
        cumulative_mm = local.cumulative_mm.astype(np.float32),
        tangent = local.tangent.astype(np.float32),
        normal_u = local.normal_u.astype(np.float32),
        normal_v = local.normal_v.astype(np.float32),
        method = (str(local.method) + "+cropped_bbox"),
        quality_flag = local.quality_flag,
        topology = topology,
        root_assignment = root_assignment,
    )

def _clone_centerline(
    centerline: CenterlineResult,
    path_phys: np.ndarray,
    spacing: Sequence[float],
    topology_updates: Mapping[str, Any],
    quality_append: Sequence[str] = (),
) -> CenterlineResult:
    path_phys = np.asarray(path_phys, dtype = np.float32)
    path_vox = (path_phys / np.asarray(spacing, dtype = np.float32)[None, :])
    cum = (
        np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(path_phys, axis = 0), axis = 1))]
        ).astype(np.float32)
        if (path_phys.shape[0] >= 2)
        else np.zeros(path_phys.shape[0], dtype = np.float32)
    )
    tangent, u, v = parallel_transport_frames(path_phys)
    topology = dict(centerline.topology)
    topology.update(dict(topology_updates))
    flags = [f for f in str(centerline.quality_flag).split(";") if (f and (f != "pass"))]
    flags.extend([f for f in quality_append if f])
    quality = ("pass" if not flags else ";".join(dict.fromkeys(flags)))
    return CenterlineResult(
        branch = centerline.branch,
        path_voxel = path_vox.astype(np.float32),
        path_physical_mm = path_phys.astype(np.float32),
        cumulative_mm = cum,
        tangent = tangent,
        normal_u = u,
        normal_v = v,
        method = (str(centerline.method) + "+CTA_recentered"),
        quality_flag = quality,
        topology = topology,
        root_assignment = centerline.root_assignment,
    )

def refine_centerline_with_cta(
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    spacing: Sequence[float],
) -> CenterlineResult:
    if (not centerline.ok or profile.empty or (len(profile) != centerline.path_physical_mm.shape[0])):
        return centerline
    valid = (pd.to_numeric(profile["valid"], errors = "coerce").fillna(0).to_numpy(int) > 0)
    uoff = pd.to_numeric(
        profile.get("refined_center_u_mm", pd.Series(np.nan, index = profile.index)), errors = "coerce"
    ).to_numpy(float)
    voff = pd.to_numeric(
        profile.get("refined_center_v_mm", pd.Series(np.nan, index = profile.index)), errors = "coerce"
    ).to_numpy(float)
    quality = (
        pd.to_numeric(
            profile.get("quality_score", pd.Series(0.0, index = profile.index)), errors = "coerce"
        )
        .fillna(0)
        .to_numpy(float)
    )
    usable = (((valid & np.isfinite(uoff)) & np.isfinite(voff)) & (quality >= 0.48))
    n = len(profile)
    if (np.sum(usable) < max(5, int((0.20 * n)))):
        topology = {
            "cta_support_fraction": (float(np.mean(valid)) if n else 0.0),
            "cta_recenter_status": "insufficient_valid_sections",
        }
        return _clone_centerline(
            centerline, centerline.path_physical_mm, spacing, topology, ["cta_lumen_continuity_low"]
        )
    idx = np.arange(n)
    u = np.interp(idx, idx[usable], uoff[usable])
    v = np.interp(idx, idx[usable], voff[usable])
    u = gaussian_filter1d(u, sigma = max(1.2, (n / 160.0)), mode = "nearest")
    v = gaussian_filter1d(v, sigma = max(1.2, (n / 160.0)), mode = "nearest")
    magnitude = np.sqrt(((u * u) + (v * v)))
    scale = np.ones(n, dtype = float)
    too_large = (magnitude > 1.4)
    scale[too_large] = (1.4 / np.maximum(magnitude[too_large], 1e-6))
    u *= scale
    v *= scale
    shifted = (
        (centerline.path_physical_mm + (u[:, None] * centerline.normal_u)) + (v[:, None] * centerline.normal_v)
    ).astype(np.float32)
    if (n >= 5):
        sigma = max(0.8, (n / 220.0))
        for d in range(3):
            shifted[:, d] = gaussian_filter1d(shifted[:, d], sigma = sigma, mode = "nearest")

    step_mm = (
        float(np.median(np.diff(centerline.cumulative_mm)))
        if (centerline.cumulative_mm.size >= 2)
        else 0.55
    )
    valid_run_count, valid_run_mm = sections_ops._longest_true_run(valid, step_mm)
    original_length = (float(centerline.cumulative_mm[-1]) if centerline.cumulative_mm.size else 0.0)
    trim_end = n
    last_valid = (int(np.where(valid)[0][-1]) if np.any(valid) else -1)
    if (
        (last_valid >= 0)
        and ((((n - 1) - last_valid) * step_mm) >= 5.0)
        and (last_valid >= max(5, int((0.35 * n))))
    ):
        trim_end = min(n, ((last_valid + max(2, int(round((1.5 / max(step_mm, 0.2)))))) + 1))
    trimmed = shifted[:trim_end]
    trimmed_mm = max(
        0.0,
        (
            original_length - (float(centerline.cumulative_mm[(trim_end - 1)]) if (trim_end > 0) else 0.0)
        ),
    )
    support_fraction = (float(np.mean(valid)) if n else 0.0)
    flags: List[str] = []
    if ((support_fraction < 0.35) or (valid_run_mm < 12.0)):
        flags.append("cta_lumen_continuity_low")
    topology = {
        "cta_support_fraction": support_fraction,
        "cta_longest_valid_run_count": int(valid_run_count),
        "cta_longest_valid_run_mm": float(valid_run_mm),
        "cta_recentered_mean_shift_mm": float(
            np.mean(np.sqrt(((u[usable] ** 2) + (v[usable] ** 2))))
        ),
        "cta_recentered_max_shift_mm": float(
            np.max(np.sqrt(((u[usable] ** 2) + (v[usable] ** 2))))
        ),
        "cta_trimmed_tip_mm": float(trimmed_mm),
        "cta_recenter_status": "success",
    }
    return _clone_centerline(centerline, trimmed, spacing, topology, flags)

def analyze_lumen_cross_sections(
    image: np.ndarray,
    branch_mask: np.ndarray,
    centerline: CenterlineResult,
    spacing: Sequence[float],
    calc_threshold_hu: float,
    pixel_mm: float = DEFAULT_CROSS_SECTION_PIXEL_MM,
    half_width_mm: float = DEFAULT_PATCH_HALF_WIDTH_MM,
    keep_patches: bool = False,
) -> Tuple[CenterlineResult, pd.DataFrame, Dict[int, PlanePatch], Dict[str, Any]]:
    first_profile, first_patches = sections_ops._measure_cross_sections_once(
        image,
        branch_mask,
        centerline,
        spacing,
        calc_threshold_hu,
        pixel_mm,
        half_width_mm,
        keep_patches = keep_patches,
    )
    refined = refine_centerline_with_cta(centerline, first_profile, spacing)
    first_valid_fraction = (
        float(
            pd.to_numeric(first_profile.get("valid", pd.Series(dtype = float)), errors = "coerce")
            .fillna(0)
            .mean()
        )
        if not first_profile.empty
        else 0.0
    )
    mean_shift = float(refined.topology.get("cta_recentered_mean_shift_mm", 0.0))
    max_shift = float(refined.topology.get("cta_recentered_max_shift_mm", 0.0))
    trimmed_tip = float(refined.topology.get("cta_trimmed_tip_mm", 0.0))
    need_second_pass = bool(
        (
            refined.ok
            and (refined.path_voxel.shape[0] <= 280)
            and ((mean_shift > 0.55) or (max_shift > 1.35) or (trimmed_tip > 1.5))
        )
    )

    if need_second_pass:
        final_profile, patches = sections_ops._measure_cross_sections_once(
            image,
            branch_mask,
            refined,
            spacing,
            calc_threshold_hu,
            pixel_mm,
            half_width_mm,
            keep_patches = keep_patches,
        )
        pass_count = 2
    else:
        final_profile = first_profile.copy()
        patches = first_patches
        pass_count = 1
        final_n = (int(refined.path_voxel.shape[0]) if refined.ok else int(len(final_profile)))
        if (final_n < len(final_profile)):
            final_profile = final_profile.iloc[:final_n].reset_index(drop = True)
            patches = {i: p for i, p in patches.items() if (int(i) < final_n)}
        if (refined.ok and (len(final_profile) == refined.path_voxel.shape[0])):
            final_profile["distance_mm"] = refined.cumulative_mm.astype(float)
            final_profile["center_voxel_x"] = refined.path_voxel[:, 0].astype(float)
            final_profile["center_voxel_y"] = refined.path_voxel[:, 1].astype(float)
            final_profile["center_voxel_z"] = refined.path_voxel[:, 2].astype(float)

    valid = (
        pd.to_numeric(final_profile.get("valid", pd.Series(dtype = float)), errors = "coerce")
        .fillna(0)
        .to_numpy(int)
        > 0
    )
    step = (
        float(np.median(np.diff(refined.cumulative_mm)))
        if (refined.cumulative_mm.size >= 2)
        else 0.55
    )
    run_count, run_mm = sections_ops._longest_true_run(valid, step)
    qc = {
        "measurement_pass_count": int(pass_count),
        "second_pass_triggered": int(need_second_pass),
        "first_pass_valid_fraction": float(first_valid_fraction),
        "final_valid_fraction": (float(np.mean(valid)) if valid.size else 0.0),
        "longest_valid_run_count": int(run_count),
        "longest_valid_run_mm": float(run_mm),
        "centerline_quality": str(refined.quality_flag),
        **{str(k): v for k, v in refined.topology.items() if str(k).startswith("cta_")},
    }
    return refined, final_profile, patches, qc
