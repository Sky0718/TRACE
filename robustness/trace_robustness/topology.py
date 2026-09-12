from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from scipy.ndimage import distance_transform_edt
import numpy as np
from skimage.measure import label as skimage_label
from . import runtime, graph, measurements, quality_tables

def split_left_coronary_centroid(
    left_mask: np.ndarray,
    bifurcation_min_area_px: int = runtime.DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    lm_mask = np.zeros_like(left_mask, dtype = bool)
    lad_mask = np.zeros_like(left_mask, dtype = bool)
    lcx_mask = np.zeros_like(left_mask, dtype = bool)
    info = {
        "method": "slice_connected_component_centroid_tracking",
        "bifurcation_z": None,
        "warning": "",
    }
    active_z = np.where(left_mask.any(axis = (0, 1)))[0]
    if (active_z.size == 0):
        info["warning"] = "empty_left_mask"
        return (lm_mask, lad_mask, lcx_mask, info)
    bif_z = None
    prev_lad = None
    prev_lcx = None
    for z in active_z:
        (labeled, num) = skimage_label(
            left_mask[:, :, z], background = 0, return_num = True, connectivity = 2
        )
        comps = []
        for cid in range(1, (num + 1)):
            region = (labeled == cid)
            area = int(np.sum(region))
            if (area < int(bifurcation_min_area_px)):
                continue
            (ys, xs) = np.where(region)
            comps.append((cid, area, np.array([ys.mean(), xs.mean()], dtype = np.float32)))
        if (len(comps) >= 2):
            comps.sort(key = lambda x: x[1], reverse = True)
            (c1, c2) = (comps[0], comps[1])
            pair = sorted([c1, c2], key = lambda x: x[2][0])
            bif_z = int(z)
            prev_lad = pair[0][2]
            prev_lcx = pair[1][2]
            lad_mask[:, :, z] = (labeled == pair[0][0])
            lcx_mask[:, :, z] = (labeled == pair[1][0])
            break
    if (bif_z is None):
        lad_mask[:] = left_mask
        info["warning"] = (
            "left_bifurcation_not_detected; entire_left_component_assigned_to_LAD"
        )
        return (lm_mask, lad_mask, lcx_mask, info)
    info["bifurcation_z"] = int(bif_z)
    pre_z = active_z[(active_z < bif_z)]
    lm_mask[:, :, pre_z] = left_mask[:, :, pre_z]
    for z in active_z[(active_z > bif_z)]:
        (labeled, num) = skimage_label(
            left_mask[:, :, z], background = 0, return_num = True, connectivity = 2
        )
        comps = []
        for cid in range(1, (num + 1)):
            region = (labeled == cid)
            if (int(np.sum(region)) < int(bifurcation_min_area_px)):
                continue
            (ys, xs) = np.where(region)
            centroid = np.array([ys.mean(), xs.mean()], dtype = np.float32)
            comps.append((cid, region, centroid))
        for cid, region, centroid in comps:
            d_lad = (
                float(np.linalg.norm((centroid - prev_lad))) if (prev_lad is not None) else 0.0
            )
            d_lcx = (
                float(np.linalg.norm((centroid - prev_lcx))) if (prev_lcx is not None) else 0.0
            )
            if (d_lad <= d_lcx):
                lad_mask[:, :, z] |= region
                prev_lad = centroid
            else:
                lcx_mask[:, :, z] |= region
                prev_lcx = centroid
    return (lm_mask, lad_mask, lcx_mask, info)

def separate_components_by_size(
    vessel_mask: np.ndarray,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    (labeled, num) = measurements.connected_components_3d(vessel_mask)
    info = {
        "method": "3d_connected_components_volume_sort_plus_left_centroid_tracking",
        "warnings": [],
    }
    if (num == 0):
        info["warnings"].append("no_vessel_component")
        return ({}, info)
    sizes = np.bincount(labeled.ravel())[1:]
    valid_ids = (np.where((sizes >= int(min_branch_voxels)))[0] + 1)
    if (valid_ids.size == 0):
        valid_ids = np.array([(int(np.argmax(sizes)) + 1)], dtype = np.int32)
        info["warnings"].append(
            "no_component_passed_min_branch_voxels; using_largest_component_only"
        )
    sorted_ids = valid_ids[np.argsort(sizes[(valid_ids - 1)])[::-1]]
    branches: Dict[str, np.ndarray] = {}
    branches["RCA"] = (labeled == sorted_ids[0])
    if (len(sorted_ids) >= 2):
        left_mask = (labeled == sorted_ids[1])
        (lm, lad, lcx, left_info) = split_left_coronary(left_mask)
        info["left_split"] = left_info
        if np.any(lm):
            branches["LM"] = lm
        if np.any(lad):
            branches["LAD"] = lad
        if np.any(lcx):
            branches["LCX"] = lcx
    else:
        info["warnings"].append("only_one_major_component_detected; no_left_system_split")
    for idx, cid in enumerate(sorted_ids[2:], start = 1):
        branches[f"OTHER{idx}"] = (labeled == cid)
    return (branches, info)

def _physical_coords(coords: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    if (coords.size == 0):
        return np.zeros((0, 3), dtype = np.float32)
    return (coords.astype(np.float32) * np.asarray(spacing, dtype = np.float32)[None, :])

def _branch_component_masks(
    mask: np.ndarray, min_voxels: int
) -> List[Tuple[int, int, np.ndarray]]:
    (labeled, num) = measurements.connected_components_3d(mask.astype(bool))
    if (num == 0):
        return []
    sizes = np.bincount(labeled.ravel())
    ids = [i for i in range(1, len(sizes)) if (int(sizes[i]) >= int(min_voxels))]
    if not ids:
        ids = [(int(np.argmax(sizes[1:])) + 1)] if (len(sizes) > 1) else []
    items = [(int(i), int(sizes[i]), (labeled == i)) for i in ids]
    items.sort(key = lambda x: x[1], reverse = True)
    return items

def _mask_centroid_physical(
    mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    coords = np.argwhere(mask.astype(bool))
    if (coords.shape[0] == 0):
        return np.zeros(3, dtype = np.float32)
    return np.mean(_physical_coords(coords, spacing), axis = 0).astype(np.float32)

def _split_score(
    lm: np.ndarray, lad: np.ndarray, lcx: np.ndarray, info: Dict[str, Any]
) -> float:
    lm_n = int(np.sum(lm))
    lad_n = int(np.sum(lad))
    lcx_n = int(np.sum(lcx))
    score = 0.0
    if (info.get("bifurcation_z") is not None):
        score += 4.0
    if (lad_n > runtime.DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX):
        score += 3.0
    if (lcx_n > runtime.DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX):
        score += 3.0
    if (lm_n > runtime.LM_REPAIR_MIN_VOXELS):
        score += 1.0
    if ((lad_n > 0) and (lcx_n > 0)):
        balance = (min(lad_n, lcx_n) / max(max(lad_n, lcx_n), 1))
        score += (2.0 * float(balance))
    warning = quality_tables.clean_warning_value(info.get("warning", ""))
    if warning:
        score -= 1.0
    return float(score)

def _repair_lm_from_left_mask(
    left_mask: np.ndarray,
    lm_mask: np.ndarray,
    lad_mask: np.ndarray,
    lcx_mask: np.ndarray,
    info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    lm = lm_mask.copy().astype(bool)
    lad = lad_mask.copy().astype(bool)
    lcx = lcx_mask.copy().astype(bool)
    active_z = np.where(left_mask.any(axis = (0, 1)))[0]
    if (active_z.size == 0):
        return (lm, lad, lcx, info)
    if (int(np.sum(lm)) >= runtime.LM_REPAIR_MIN_VOXELS):
        return (lm, lad, lcx, info)
    bif_z = info.get("bifurcation_z", None)
    if (bif_z is None):
        return (lm, lad, lcx, info)
    z0 = int(active_z[0])
    z1 = int(
        min(active_z[-1], (max(int(bif_z), z0) + max(1, int(round((0.04 * active_z.size))))))
    )
    band_z = active_z[((active_z >= z0) & (active_z <= z1))]
    if (band_z.size == 0):
        return (lm, lad, lcx, info)
    candidate = np.zeros_like(left_mask, dtype = bool)
    candidate[:, :, band_z] = left_mask[:, :, band_z]
    if (int(np.sum(candidate)) > (0.25 * int(np.sum(left_mask)))):
        band_z = active_z[: max(1, min(2, active_z.size))]
        candidate[:] = False
        candidate[:, :, band_z] = left_mask[:, :, band_z]
    if (int(np.sum(candidate)) > 0):
        lm |= candidate
        lad &= ~candidate
        lcx &= ~candidate
        info = dict(info)
        old_warning = quality_tables.clean_warning_value(info.get("warning", ""))
        repair_note = "LM_repaired_by_proximal_band"
        info["lm_repair"] = repair_note
        info["warning"] = "; ".join([x for x in [old_warning, repair_note] if x])
    return (lm, lad, lcx, info)

def split_left_coronary_repaired(
    left_mask: np.ndarray,
    bifurcation_min_area_px: int = runtime.DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    (lm, lad, lcx, info) = split_left_coronary_centroid(
        left_mask, bifurcation_min_area_px = bifurcation_min_area_px
    )
    info = dict(info)
    (lm, lad, lcx, info) = _repair_lm_from_left_mask(left_mask, lm, lad, lcx, info)
    if (np.any(left_mask) and (not np.any(lad) or not np.any(lcx))):
        active_z = np.where(left_mask.any(axis = (0, 1)))[0]
        if (active_z.size >= 2):
            start_idx = max(0, int((0.15 * active_z.size)))
            distal_z = active_z[start_idx:]
            coords = np.argwhere(left_mask[:, :, distal_z].astype(bool))
            if (coords.shape[0] >= 20):
                coords_full = coords.copy()
                coords_full[:, 2] = distal_z[coords[:, 2]]
                xy = coords_full[:, :2].astype(np.float32)
                xy_center = xy.mean(axis = 0)
                try:
                    (_, _, vt) = np.linalg.svd((xy - xy_center), full_matrices = False)
                    axis = vt[0]
                    proj = ((xy - xy_center) @ axis)
                    threshold = float(np.median(proj))
                    m1_coords = coords_full[(proj <= threshold)]
                    m2_coords = coords_full[(proj > threshold)]
                    if ((m1_coords.shape[0] > 0) and (m2_coords.shape[0] > 0)):
                        m1 = np.zeros_like(left_mask, dtype = bool)
                        m2 = np.zeros_like(left_mask, dtype = bool)
                        m1[m1_coords[:, 0], m1_coords[:, 1], m1_coords[:, 2]] = True
                        m2[m2_coords[:, 0], m2_coords[:, 1], m2_coords[:, 2]] = True
                        c1 = np.mean(m1_coords[:, 0])
                        c2 = np.mean(m2_coords[:, 0])
                        if (c1 <= c2):
                            (lad, lcx) = (m1, m2)
                        else:
                            (lad, lcx) = (m2, m1)
                        (lm, lad, lcx, info) = _repair_lm_from_left_mask(
                            left_mask, lm, lad, lcx, info
                        )
                        info["distal_split_rescue"] = "pca_median_axis"
                        old_warning = quality_tables.clean_warning_value(
                            info.get("warning", "")
                        )
                        info["warning"] = "; ".join(
                            [
                                x
                                for x in [
                                    old_warning,
                                    "LAD_LCX_rescued_by_PCA_distal_split_low_confidence",
                                ]
                                if x
                            ]
                        )
                except Exception:
                    pass
    return (lm.astype(bool), lad.astype(bool), lcx.astype(bool), info)

def _graph_voronoi_split_single_component(
    vessel_mask: np.ndarray, spacing: Tuple[float, float, float], k: int = 3
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    info = {
        "method": "single_component_skeleton_voronoi_fallback",
        "warnings": ["single_component_graph_voronoi_fallback_low_confidence"],
    }
    (skel, _) = graph.skeletonize_3d_safe(vessel_mask)
    skel = graph.largest_skeleton_component(skel)
    (coords, _, adj) = graph.build_graph_from_skeleton(skel, spacing)
    if (coords.shape[0] < 10):
        info["warnings"].append("single_component_fallback_failed_small_skeleton")
        return ({}, info)
    deg = np.array([len(a) for a in adj], dtype = np.int32)
    endpoints = np.where((deg <= 1))[0]
    if (endpoints.size < k):
        endpoints = np.arange(coords.shape[0], dtype = np.int32)
    phys = _physical_coords(coords, spacing)
    ep_phys = phys[endpoints]
    seed_local = [int(np.argmax(np.linalg.norm((ep_phys - ep_phys.mean(axis = 0)), axis = 1)))]
    while (len(seed_local) < min(k, endpoints.size)):
        dmin = np.full(endpoints.size, np.inf, dtype = np.float32)
        for s in seed_local:
            dmin = np.minimum(dmin, np.linalg.norm((ep_phys - ep_phys[s]), axis = 1))
        seed_local.append(int(np.argmax(dmin)))
    seeds = endpoints[np.asarray(seed_local, dtype = np.int32)]
    dist_stack = []
    for s in seeds:
        (_, dist, _) = graph.dijkstra_farthest(adj, int(s))
        dist_stack.append(dist)
    D = np.vstack(dist_stack)
    skel_labels = (np.argmin(D, axis = 0).astype(np.int16) + 1)
    label_vol = np.zeros_like(vessel_mask, dtype = np.int16)
    label_vol[coords[:, 0], coords[:, 1], coords[:, 2]] = skel_labels
    seed_mask = (label_vol > 0)
    if not np.any(seed_mask):
        return ({}, info)
    try:
        (_, inds) = distance_transform_edt(~seed_mask, return_indices = True)
        full_labels = label_vol[inds[0], inds[1], inds[2]]
        full_labels = full_labels.astype(np.int16)
        full_labels[~vessel_mask.astype(bool)] = 0
    except Exception:
        info["warnings"].append("single_component_voronoi_distance_transform_failed")
        return ({}, info)
    masks = []
    for lab in range(1, (int(np.max(skel_labels)) + 1)):
        m = (full_labels == lab)
        if (int(np.sum(m)) > 0):
            masks.append(m)
    if (len(masks) < 2):
        return ({}, info)
    masks.sort(key = lambda m: int(np.sum(m)), reverse = True)
    out: Dict[str, np.ndarray] = {"RCA": masks[0]}
    remaining = masks[1:3]
    if (len(remaining) == 1):
        out["LAD"] = remaining[0]
    elif (len(remaining) >= 2):
        c0 = _mask_centroid_physical(remaining[0], spacing)
        c1 = _mask_centroid_physical(remaining[1], spacing)
        axis = 0 if (abs((c0[0] - c1[0])) >= abs((c0[1] - c1[1]))) else 1
        if (c0[axis] <= c1[axis]):
            (out["LAD"], out["LCX"]) = (remaining[0], remaining[1])
        else:
            (out["LAD"], out["LCX"]) = (remaining[1], remaining[0])
    for i, extra in enumerate(masks[3:], start = 1):
        out[f"OTHER{i}"] = extra
    return (out, info)

def separate_branches_validated(
    vessel_mask: np.ndarray,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    comps = _branch_component_masks(vessel_mask, min_branch_voxels)
    info: Dict[str, Any] = {
        "method": "multi_hypothesis_component_scoring_plus_left_split_repair",
        "warnings": [],
        "component_count": int(len(comps)),
    }
    if not comps:
        info["warnings"].append("no_vessel_component")
        return ({}, info)
    if (len(comps) == 1):
        (fallback, fb_info) = _graph_voronoi_split_single_component(
            comps[0][2], (1.0, 1.0, 1.0), k = 3
        )
        if fallback:
            info["method"] = fb_info.get("method", info["method"])
            info["warnings"].extend(fb_info.get("warnings", []))
            return (fallback, info)
        info["warnings"].append(
            "single_component_fallback_failed; using_legacy_largest_component_rule"
        )
        return separate_components_by_size(vessel_mask, min_branch_voxels = min_branch_voxels)
    candidates = []
    for cid, size, cmask in comps[: min(len(comps), 6)]:
        (lm, lad, lcx, linfo) = split_left_coronary(cmask)
        score = _split_score(lm, lad, lcx, linfo)
        candidates.append(
            {
                "cid": cid,
                "size": size,
                "mask": cmask,
                "lm": lm,
                "lad": lad,
                "lcx": lcx,
                "info": linfo,
                "score": score,
            }
        )
    candidates.sort(key = lambda d: (float(d["score"]), int(d["size"])), reverse = True)
    left_cand = candidates[0]
    left_id = int(left_cand["cid"])
    branches: Dict[str, np.ndarray] = {}
    remaining = [(cid, size, cmask) for (cid, size, cmask) in comps if (int(cid) != left_id)]
    if remaining:
        (rca_id, rca_size, rca_mask) = sorted(remaining, key = lambda x: x[1], reverse = True)[
            0
        ]
        branches["RCA"] = rca_mask
    else:
        info["warnings"].append("no_remaining_component_for_RCA")
    if np.any(left_cand["lm"]):
        branches["LM"] = left_cand["lm"].astype(bool)
    if np.any(left_cand["lad"]):
        branches["LAD"] = left_cand["lad"].astype(bool)
    if np.any(left_cand["lcx"]):
        branches["LCX"] = left_cand["lcx"].astype(bool)
    info["left_split"] = left_cand["info"]
    info["left_component_id"] = left_id
    info["left_component_score"] = float(left_cand["score"])
    if (("LAD" not in branches) or ("LCX" not in branches)):
        info["warnings"].append("left_system_incomplete_after_multi_hypothesis_split")
    used_ids = {left_id}
    if remaining:
        used_ids.add(int(sorted(remaining, key = lambda x: x[1], reverse = True)[0][0]))
    other_index = 1
    for cid, size, cmask in comps:
        if (int(cid) not in used_ids):
            branches[f"OTHER{other_index}"] = cmask
            other_index += 1
    return (branches, info)

def _graph_split_left_coronary(
    left_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    lm = np.zeros_like(left_mask, dtype = bool)
    lad = np.zeros_like(left_mask, dtype = bool)
    lcx = np.zeros_like(left_mask, dtype = bool)
    info: Dict[str, Any] = {
        "method": "graph_bifurcation_left_split_v1",
        "warning": "",
        "confidence": "low_to_medium",
    }
    if not np.any(left_mask):
        info["warning"] = "empty_left_mask"
        return (lm, lad, lcx, info)
    try:
        (skel, skel_method) = graph.skeletonize_3d_safe(left_mask.astype(bool))
        skel = graph.largest_skeleton_component(skel)
        (coords, _, adj) = graph.build_graph_from_skeleton(skel, (1.0, 1.0, 1.0))
        if (coords.shape[0] < 12):
            info["warning"] = "graph_left_split_failed_small_skeleton"
            return (lm, lad, lcx, info)
        degrees = np.array([len(a) for a in adj], dtype = np.int32)
        endpoints = np.where((degrees <= 1))[0]
        branchpoints = np.where((degrees >= 3))[0]
        if ((endpoints.size < 3) or (branchpoints.size < 1)):
            info["warning"] = "graph_left_split_failed_no_clear_bifurcation"
            return (lm, lad, lcx, info)
        radius_map = distance_transform_edt(left_mask.astype(bool)).astype(np.float32)
        ep_coords = coords[endpoints]
        ep_radius = radius_map[ep_coords[:, 0], ep_coords[:, 1], ep_coords[:, 2]]
        root = int(endpoints[int(np.argmax(ep_radius))])
        (dist_root, parent_root) = graph._dijkstra_all(adj, root)
        valid_bp = [
            int(bp)
            for bp in branchpoints
            if (np.isfinite(dist_root[int(bp)]) and (float(dist_root[int(bp)]) >= 2.0))
        ]
        if not valid_bp:
            info["warning"] = "graph_left_split_failed_no_downstream_bifurcation"
            return (lm, lad, lcx, info)
        bp = min(valid_bp, key = lambda n: float(dist_root[n]))
        lm_nodes = graph.reconstruct_graph_path(parent_root, root, bp)
        if (len(lm_nodes) < 2):
            info["warning"] = "graph_left_split_failed_lm_path"
            return (lm, lad, lcx, info)
        (dist_bp, parent_bp) = graph._dijkstra_all(adj, bp)
        distal_candidates = []
        for ep in endpoints:
            ep = int(ep)
            if ((ep == root) or not np.isfinite(dist_bp[ep])):
                continue
            p = graph.reconstruct_graph_path(parent_bp, bp, ep)
            if (len(p) >= 3):
                distal_candidates.append((float(dist_bp[ep]), ep, p))
        if (len(distal_candidates) < 2):
            info["warning"] = "graph_left_split_failed_less_than_two_distal_arms"
            return (lm, lad, lcx, info)
        distal_candidates.sort(key = lambda x: x[0], reverse = True)
        arm1 = distal_candidates[0][2]
        arm2 = distal_candidates[1][2]
        seed = np.zeros_like(left_mask, dtype = np.uint8)
        c_lm = coords[np.asarray(lm_nodes, dtype = np.int32)]
        c1 = coords[np.asarray(arm1, dtype = np.int32)]
        c2 = coords[np.asarray(arm2, dtype = np.int32)]
        seed[c_lm[:, 0], c_lm[:, 1], c_lm[:, 2]] = 1
        seed[c1[:, 0], c1[:, 1], c1[:, 2]] = 2
        seed[c2[:, 0], c2[:, 1], c2[:, 2]] = 3
        seed_mask = (seed > 0)
        if not np.any(seed_mask):
            info["warning"] = "graph_left_split_failed_empty_seed"
            return (lm, lad, lcx, info)
        (_, inds) = distance_transform_edt(~seed_mask, return_indices = True)
        lab = seed[inds[0], inds[1], inds[2]]
        lab[~left_mask.astype(bool)] = 0
        lm = (lab == 1)
        arm_m1 = (lab == 2)
        arm_m2 = (lab == 3)
        if (len(arm1) >= len(arm2)):
            (lad, lcx) = (arm_m1, arm_m2)
        else:
            (lad, lcx) = (arm_m2, arm_m1)
        info.update(
            {
                "skeleton_method": skel_method,
                "root_node": int(root),
                "bifurcation_node": int(bp),
                "lm_skeleton_points": int(len(lm_nodes)),
                "lad_lcx_assignment_rule": "longer_distal_arm_as_LAD",
            }
        )
        return (lm.astype(bool), lad.astype(bool), lcx.astype(bool), info)
    except Exception as exc:
        info["warning"] = f"graph_left_split_exception:{exc}"
        return (lm, lad, lcx, info)

def split_left_coronary(
    left_mask: np.ndarray,
    bifurcation_min_area_px: int = runtime.DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    (glm, glad, glcx, ginfo) = _graph_split_left_coronary(left_mask)
    graph_score = _split_score(glm, glad, glcx, ginfo)
    if (((graph_score >= 6.0) and np.any(glad)) and np.any(glcx)):
        return (glm, glad, glcx, ginfo)
    (lm, lad, lcx, info) = split_left_coronary_repaired(
        left_mask, bifurcation_min_area_px = bifurcation_min_area_px
    )
    score = _split_score(lm, lad, lcx, info)
    if (((graph_score > score) and np.any(glad)) and np.any(glcx)):
        warn = quality_tables.clean_warning_value(ginfo.get("warning", ""))
        ginfo["warning"] = "; ".join(
            [x for x in [warn, "graph_split_used_despite_low_score"] if x]
        )
        return (glm, glad, glcx, ginfo)
    return (lm, lad, lcx, info)

def separate_branches(
    vessel_mask: np.ndarray,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    comps = _branch_component_masks(vessel_mask, min_branch_voxels)
    info: Dict[str, Any] = {
        "method": "v1_3_graph_left_split_component_scoring",
        "warnings": [],
        "component_count": int(len(comps)),
    }
    if not comps:
        info["warnings"].append("no_vessel_component")
        return ({}, info)
    if (len(comps) == 1):
        (branches, fb_info) = separate_branches_validated(
            vessel_mask, min_branch_voxels = min_branch_voxels
        )
        info.update(fb_info if isinstance(fb_info, dict) else {})
        info.setdefault("warnings", [])
        info["warnings"].append("single_component_semantic_branch_labels_low_confidence")
        return (branches, info)
    candidates = []
    for cid, size, cmask in comps[: min(len(comps), 6)]:
        (lm, lad, lcx, linfo) = split_left_coronary(cmask)
        score = _split_score(lm, lad, lcx, linfo)
        if (np.any(lm) and (int(np.sum(lm)) < runtime.LM_REPAIR_MIN_VOXELS)):
            score -= 1.0
        candidates.append(
            {
                "cid": cid,
                "size": size,
                "mask": cmask,
                "lm": lm,
                "lad": lad,
                "lcx": lcx,
                "info": linfo,
                "score": score,
            }
        )
    candidates.sort(key = lambda d: (float(d["score"]), int(d["size"])), reverse = True)
    left_cand = candidates[0]
    branches: Dict[str, np.ndarray] = {}
    left_id = int(left_cand["cid"])
    remaining = [(cid, size, cmask) for (cid, size, cmask) in comps if (int(cid) != left_id)]
    if remaining:
        (rca_id, rca_size, rca_mask) = sorted(remaining, key = lambda x: x[1], reverse = True)[
            0
        ]
        branches["RCA"] = rca_mask
    else:
        info["warnings"].append("no_remaining_component_for_RCA")
    if np.any(left_cand["lm"]):
        branches["LM"] = left_cand["lm"].astype(bool)
    if np.any(left_cand["lad"]):
        branches["LAD"] = left_cand["lad"].astype(bool)
    if np.any(left_cand["lcx"]):
        branches["LCX"] = left_cand["lcx"].astype(bool)
    info["left_split"] = left_cand["info"]
    info["left_component_id"] = int(left_id)
    info["left_component_score"] = float(left_cand["score"])
    info["semantic_branch_note"] = (
        "LM/LAD/LCX/RCA are algorithmic estimates from a binary coronary mask, not native ImageCAS labels."
    )
    if ("LM" not in branches):
        info["warnings"].append("LM_not_confidently_identified")
    if (("LAD" not in branches) or ("LCX" not in branches)):
        info["warnings"].append("left_system_incomplete_after_graph_split")
    used_ids = {left_id}
    if remaining:
        used_ids.add(int(sorted(remaining, key = lambda x: x[1], reverse = True)[0][0]))
    other_index = 1
    for cid, size, cmask in comps:
        if (int(cid) not in used_ids):
            branches[f"OTHER{other_index}"] = cmask
            other_index += 1
    return (branches, info)
