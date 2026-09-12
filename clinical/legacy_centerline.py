from __future__ import annotations

import math
import heapq
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.measure import label as skimage_label

try:
    from skimage.morphology import skeletonize

    SKIMAGE_SKELETON_OK = True
except Exception:
    SKIMAGE_SKELETON_OK = False

NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not ((dx == 0) and (dy == 0) and (dz == 0))
]

def connected_components_3d(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    labeled, num = skimage_label(mask.astype(bool), background = 0, return_num = True, connectivity = 3)
    return labeled.astype(np.int32, copy = False), int(num)

def skeletonize_3d_safe(mask: np.ndarray) -> Tuple[np.ndarray, str]:
    m = mask.astype(bool)
    if not np.any(m):
        return np.zeros_like(m, dtype = bool), "empty_mask"
    if not SKIMAGE_SKELETON_OK:
        return skeletonize_2d_fallback(m), "slice_wise_2d_fallback_no_skeletonize"
    try:
        sk = skeletonize(m, method = "lee")
        return sk.astype(bool), "skimage_skeletonize_lee_3d"
    except Exception:
        try:
            sk = skeletonize(m)
            if (sk.shape == m.shape):
                return sk.astype(bool), "skimage_skeletonize_default"
        except Exception:
            pass
    return skeletonize_2d_fallback(m), "slice_wise_2d_fallback"

def skeletonize_2d_fallback(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype = bool)
    if not SKIMAGE_SKELETON_OK:
        return out
    for z in range(mask.shape[2]):
        if np.any(mask[:, :, z]):
            try:
                out[:, :, z] = skeletonize(mask[:, :, z].astype(bool)).astype(bool)
            except Exception:
                pass
    return out

def largest_skeleton_component(skel: np.ndarray) -> np.ndarray:
    labeled, num = connected_components_3d(skel.astype(bool))
    if (num == 0):
        return np.zeros_like(skel, dtype = bool)
    sizes = np.bincount(labeled.ravel())[1:]
    keep = (int(np.argmax(sizes)) + 1)
    return (labeled == keep)

def build_graph_from_skeleton(
    skel: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int], List[List[Tuple[int, float]]]]:
    coords = np.argwhere(skel.astype(bool)).astype(np.int32)
    index = {(int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(coords.tolist())}
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(coords.shape[0])]
    sx, sy, sz = map(float, spacing)
    shape = skel.shape
    for i, (x, y, z) in enumerate(coords.tolist()):
        for dx, dy, dz in NEIGHBORS_26:
            nx, ny, nz = int((x + dx)), int((y + dy)), int((z + dz))
            if ((0 <= nx < shape[0]) and (0 <= ny < shape[1]) and (0 <= nz < shape[2])):
                j = index.get((nx, ny, nz))
                if (j is not None):
                    w = math.sqrt(((((dx * sx) ** 2) + ((dy * sy) ** 2)) + ((dz * sz) ** 2)))
                    adj[i].append((j, float(w)))
    return coords, index, adj

def centroid_axis_fallback_path(
    mask: np.ndarray, spacing: Tuple[float, float, float]
) -> np.ndarray:
    coords = np.argwhere(mask.astype(bool))
    if (coords.shape[0] == 0):
        return np.zeros((0, 3), dtype = np.int32)
    points = []
    for z in np.unique(coords[:, 2]):
        sl = coords[(coords[:, 2] == z)]
        if (sl.shape[0] == 0):
            continue
        x = int(round(float(np.mean(sl[:, 0]))))
        y = int(round(float(np.mean(sl[:, 1]))))
        points.append((x, y, int(z)))
    if (len(points) < 2):
        physical = (coords.astype(np.float32) * np.asarray(spacing, dtype = np.float32)[None, :])
        center = physical.mean(axis = 0)
        u, s, vt = np.linalg.svd((physical - center), full_matrices = False)
        axis = vt[0]
        proj = ((physical - center) @ axis)
        order = np.argsort(proj)
        pts = coords[order]
        if (pts.shape[0] > 80):
            pts = pts[np.linspace(0, (pts.shape[0] - 1), 80).round().astype(np.int32)]
        return pts.astype(np.int32)
    return np.asarray(points, dtype = np.int32)

def path_arclength_mm(path: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    if (path.shape[0] == 0):
        return np.array([], dtype = np.float32)
    phys = (path.astype(np.float32) * np.asarray(spacing, dtype = np.float32)[None, :])
    if (phys.shape[0] == 1):
        return np.array([0.0], dtype = np.float32)
    step = np.sqrt(np.sum((np.diff(phys, axis = 0) ** 2), axis = 1))
    return np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)

def _clip_path_to_mask(path: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if (path.size == 0):
        return path.astype(np.int32)
    keep = []
    sx, sy, sz = mask.shape
    for p in np.asarray(path, dtype = np.int32):
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        if ((0 <= x < sx) and (0 <= y < sy) and (0 <= z < sz) and bool(mask[x, y, z])):
            keep.append((x, y, z))
    if not keep:
        return np.zeros((0, 3), dtype = np.int32)
    out = np.asarray(keep, dtype = np.int32)
    uniq = [out[0]]
    for i in range(1, out.shape[0]):
        if not np.array_equal(out[i], out[(i - 1)]):
            uniq.append(out[i])
    return np.asarray(uniq, dtype = np.int32)

def _resample_voxel_path(
    path: np.ndarray, spacing: Tuple[float, float, float], step_mm: float = 0.65
) -> np.ndarray:
    path = np.asarray(path, dtype = np.float32)
    if (path.shape[0] < 2):
        return path.astype(np.int32)
    sp = np.asarray(spacing, dtype = np.float32)
    phys = (path * sp[None, :])
    seg = np.sqrt(np.sum((np.diff(phys, axis = 0) ** 2), axis = 1))
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(cum[-1])
    if (total <= 1e-6):
        return np.rint(path).astype(np.int32)
    step_mm = float(np.clip(step_mm, 0.35, 1.20))
    count = max(2, (int(round((total / step_mm))) + 1))
    new_cum = np.linspace(0.0, total, count, dtype = np.float32)
    out_phys = np.zeros((count, 3), dtype = np.float32)
    for d in range(3):
        out_phys[:, d] = np.interp(new_cum, cum, phys[:, d])
    out = np.rint((out_phys / sp[None, :])).astype(np.int32)
    uniq = [out[0]]
    for i in range(1, out.shape[0]):
        if not np.array_equal(out[i], out[(i - 1)]):
            uniq.append(out[i])
    return np.asarray(uniq, dtype = np.int32)

def _smooth_voxel_path(
    path: np.ndarray, spacing: Tuple[float, float, float], sigma_mm: float = 1.2
) -> np.ndarray:
    path = np.asarray(path, dtype = np.float32)
    if (path.shape[0] < 5):
        return np.rint(path).astype(np.int32)
    sp = np.asarray(spacing, dtype = np.float32)
    phys = (path * sp[None, :])
    cum = path_arclength_mm(path.astype(np.int32), spacing)
    step = (float(np.median(np.diff(cum))) if (cum.size >= 2) else 0.7)
    sigma = max((float(sigma_mm) / max(step, 1e-3)), 0.5)
    for d in range(3):
        phys[:, d] = gaussian_filter1d(phys[:, d], sigma = sigma, mode = "nearest")
    out = np.rint((phys / sp[None, :])).astype(np.int32)
    uniq = [out[0]]
    for i in range(1, out.shape[0]):
        if not np.array_equal(out[i], out[(i - 1)]):
            uniq.append(out[i])
    return np.asarray(uniq, dtype = np.int32)

def _dijkstra_all(adj: List[List[Tuple[int, float]]], start: int) -> Tuple[List[float], List[int]]:
    n = len(adj)
    dist = ([float("inf")] * n)
    parent = ([-1] * n)
    dist[start] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, int(start))]
    while pq:
        d, u = heapq.heappop(pq)
        if (d != dist[u]):
            continue
        for v, w in adj[u]:
            nd = (d + float(w))
            if (nd < dist[int(v)]):
                dist[int(v)] = nd
                parent[int(v)] = int(u)
                heapq.heappush(pq, (nd, int(v)))
    return dist, parent

def _reconstruct_idx_path(parent: List[int], start: int, end: int) -> List[int]:
    path = []
    cur = int(end)
    seen = set()
    while ((cur != -1) and (cur not in seen)):
        seen.add(cur)
        path.append(cur)
        if (cur == int(start)):
            break
        cur = (int(parent[cur]) if (cur < len(parent)) else -1)
    if (not path or (path[-1] != int(start))):
        return []
    path.reverse()
    return path

def _path_quality_metrics(
    path: np.ndarray, spacing: Tuple[float, float, float]
) -> Dict[str, float]:
    path = np.asarray(path, dtype = np.float64)
    if (path.shape[0] < 2):
        return {
            "length_mm": 0.0,
            "endpoint_distance_mm": 0.0,
            "tortuosity": 0.0,
            "sharp_turn_fraction": 0.0,
            "turn_angle_p95_deg": 0.0,
            "turn_angle_max_deg": 0.0,
        }
    sp = np.asarray(spacing, dtype = np.float64)
    phys = (path * sp[None, :])
    if (phys.shape[0] >= 5):
        sigma = max(0.8, (phys.shape[0] / 140.0))
        for k in range(3):
            phys[:, k] = gaussian_filter1d(phys[:, k], sigma = sigma, mode = "nearest")
    seg = np.diff(phys, axis = 0)
    step = np.linalg.norm(seg, axis = 1)
    length = float(np.sum(step))
    endpoint = float(np.linalg.norm((phys[-1] - phys[0])))
    tort = (float((length / max(endpoint, 1e-6))) if (length > 0) else 0.0)
    if (path.shape[0] >= 5):
        target_mm = 2.5
        median_step = (float(np.median(step[(step > 1e-6)])) if np.any((step > 1e-6)) else target_mm)
        off = max(1, int(round((target_mm / max(median_step, 1e-6)))))
        tangents = []
        for i in range(off, (phys.shape[0] - off)):
            v = (phys[(i + off)] - phys[(i - off)])
            nv = np.linalg.norm(v)
            if (nv > 1e-6):
                tangents.append((v / nv))
        unit = np.asarray(tangents, dtype = np.float64)
    else:
        unit = np.zeros((0, 3), dtype = np.float64)
    if (unit.shape[0] >= 2):
        dots = np.sum((unit[:-1] * unit[1:]), axis = 1)
        ang = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    else:
        ang = np.zeros(0, dtype = np.float64)
    return {
        "length_mm": float(length),
        "endpoint_distance_mm": float(endpoint),
        "tortuosity": float(tort),
        "sharp_turn_fraction": (float(np.mean((ang > 70.0))) if ang.size else 0.0),
        "turn_angle_p95_deg": (float(np.percentile(ang, 95.0)) if ang.size else 0.0),
        "turn_angle_max_deg": (float(np.max(ang)) if ang.size else 0.0),
    }

def _graph_local_radius_score(
    endpoint: int,
    adj: List[List[Tuple[int, float]]],
    coords: np.ndarray,
    radius_map: np.ndarray,
    radius_mm: float = 5.0,
) -> float:
    dist, _ = _dijkstra_all(adj, int(endpoint))
    idx = np.asarray(
        [i for i, d in enumerate(dist) if (np.isfinite(d) and (d <= float(radius_mm)))],
        dtype = np.int32,
    )
    if (idx.size == 0):
        p = coords[int(endpoint)]
        return float(radius_map[p[0], p[1], p[2]])
    pts = coords[idx]
    vals = radius_map[pts[:, 0], pts[:, 1], pts[:, 2]]
    return (float(np.mean(vals)) if vals.size else 0.0)

def _radius_reexpansion_fraction(radius: np.ndarray) -> float:
    r = np.asarray(radius, dtype = np.float32)
    if (r.size < 8):
        return 0.0
    rs = gaussian_filter1d(r, sigma = max(1.0, (r.size / 120.0)), mode = "nearest")
    start = max(1, int(round((0.20 * rs.size))))
    dr = np.diff(rs[start:])
    if (dr.size == 0):
        return 0.0
    scale = max(float(np.median(rs)), 0.5)
    return float(np.mean((dr > max(0.08, (0.04 * scale)))))

def _extract_centerline_root_to_tip(
    branch_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    branch_name: str = "",
    root_hint: Optional[np.ndarray] = None,
    root_source: str = "",
) -> Tuple[np.ndarray, str, str, Dict[str, float]]:
    if (int(np.sum(branch_mask)) < 8):
        empty = np.zeros((0, 3), dtype = np.int32)
        return (
            empty,
            "too_small",
            "too_small_branch_mask",
            {
                "skeleton_endpoint_count": 0.0,
                "candidate_path_count": 0.0,
                "root_radius_mm": 0.0,
                "tip_radius_mm": 0.0,
                "path_skeleton_fraction": 0.0,
                "radius_reexpansion_fraction": 0.0,
                "root_ambiguity_ratio": 0.0,
            },
        )

    skel, skel_method = skeletonize_3d_safe(branch_mask)
    skel = largest_skeleton_component(skel)
    coords, index, adj = build_graph_from_skeleton(skel, spacing)
    if (coords.shape[0] < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (
            approx,
            "centroid_axis_fallback",
            "skeleton_too_small",
            {
                "skeleton_endpoint_count": 0.0,
                "candidate_path_count": 0.0,
                "root_radius_mm": 0.0,
                "tip_radius_mm": 0.0,
                "path_skeleton_fraction": 0.0,
                "radius_reexpansion_fraction": 0.0,
                "root_ambiguity_ratio": 0.0,
            },
        )

    degrees = np.asarray([len(a) for a in adj], dtype = np.int32)
    endpoints = np.where((degrees <= 1))[0]
    if (endpoints.size < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (
            approx,
            "centroid_axis_fallback",
            "no_endpoint_pair",
            {
                "skeleton_endpoint_count": float(endpoints.size),
                "candidate_path_count": 0.0,
                "root_radius_mm": 0.0,
                "tip_radius_mm": 0.0,
                "path_skeleton_fraction": 0.0,
                "radius_reexpansion_fraction": 0.0,
                "root_ambiguity_ratio": 0.0,
            },
        )

    radius_map = distance_transform_edt(branch_mask.astype(bool), sampling = spacing).astype(
        np.float32
    )
    endpoint_pts = coords[endpoints]
    endpoint_r = radius_map[endpoint_pts[:, 0], endpoint_pts[:, 1], endpoint_pts[:, 2]].astype(
        np.float32
    )
    local_r = np.asarray(
        [
            _graph_local_radius_score(int(ep), adj, coords, radius_map, radius_mm = 5.0)
            for ep in endpoints
        ],
        dtype = np.float32,
    )
    sp = np.asarray(spacing, dtype = np.float32)

    if ((root_hint is not None) and (np.asarray(root_hint).size == 3)):
        hint = np.asarray(root_hint, dtype = np.float32)
        d = np.linalg.norm(
            ((endpoint_pts.astype(np.float32) - hint[None, :]) * sp[None, :]), axis = 1
        )
        root_scores = ((-d + (0.30 * endpoint_r)) + (0.15 * local_r))
        root_pos = int(np.argmax(root_scores))
        source = str((root_source or "parent_contact"))
    else:
        root_scores = ((3.5 * endpoint_r) + (2.0 * local_r))
        root_pos = int(np.argmax(root_scores))
        source = str((root_source or "largest_radius_endpoint"))

    sorted_root_scores = np.sort(root_scores)[::-1]
    if (sorted_root_scores.size >= 2):
        denom = max(abs(float(sorted_root_scores[0])), 1e-6)
        root_ambiguity = float(np.clip((abs(float(sorted_root_scores[1])) / denom), 0.0, 2.0))
    else:
        root_ambiguity = 0.0

    root_idx = int(endpoints[root_pos])
    root_pt = coords[root_idx]
    root_radius = float(radius_map[root_pt[0], root_pt[1], root_pt[2]])
    dist, parent = _dijkstra_all(adj, root_idx)

    best_score = -1e18
    best_nodes: List[int] = []
    best_metrics: Dict[str, float] = {}
    candidate_count = 0
    for tip_idx in endpoints:
        tip_idx = int(tip_idx)
        if ((tip_idx == root_idx) or not np.isfinite(dist[tip_idx])):
            continue
        node_path = _reconstruct_idx_path(parent, root_idx, tip_idx)
        if (len(node_path) < 2):
            continue
        candidate_count += 1
        path = coords[np.asarray(node_path, dtype = np.int32)]
        q = _path_quality_metrics(path, spacing)
        rr = radius_map[path[:, 0], path[:, 1], path[:, 2]].astype(np.float32)
        if (rr.size >= 5):
            rr_s = gaussian_filter1d(rr, sigma = max(0.8, (rr.size / 140.0)), mode = "nearest")
        else:
            rr_s = rr
        p25_r = (float(np.percentile(rr_s, 25.0)) if rr_s.size else 0.0)
        med_r = (float(np.median(rr_s)) if rr_s.size else 0.0)
        tip_r = (float(rr_s[-1]) if rr_s.size else 0.0)
        reexp = _radius_reexpansion_fraction(rr_s)
        caliber_length = float((q["length_mm"] * (max(med_r, 0.15) ** 0.65)))
        score = ((
            (
                (
                    ((caliber_length + (0.30 * float(q["endpoint_distance_mm"]))) + (7.0 * p25_r)) + (2.5 * max((root_radius - tip_r), 0.0))
                ) - (22.0 * max((float(q["tortuosity"]) - 1.8), 0.0))
            ) - (26.0 * reexp)
        ) - (0.05 * float(q["turn_angle_p95_deg"])))
        if (score > best_score):
            best_score = float(score)
            best_nodes = node_path
            best_metrics = dict(q)
            best_metrics.update(
                {
                    "tip_radius_mm": tip_r,
                    "root_radius_mm": root_radius,
                    "radius_reexpansion_fraction": reexp,
                }
            )

    if (len(best_nodes) < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (
            approx,
            "centroid_axis_fallback",
            "root_to_tip_candidate_failed",
            {
                "skeleton_endpoint_count": float(endpoints.size),
                "candidate_path_count": float(candidate_count),
                "root_radius_mm": root_radius,
                "tip_radius_mm": 0.0,
                "path_skeleton_fraction": 0.0,
                "radius_reexpansion_fraction": 0.0,
                "root_ambiguity_ratio": root_ambiguity,
            },
        )

    raw_path = coords[np.asarray(best_nodes, dtype = np.int32)]
    resampled = _resample_voxel_path(raw_path, spacing, step_mm = 0.55)
    smoothed = _smooth_voxel_path(resampled, spacing, sigma_mm = 1.0)
    smoothed = _clip_path_to_mask(smoothed, branch_mask.astype(bool))
    if (smoothed.shape[0] < 2):
        smoothed = _clip_path_to_mask(resampled, branch_mask.astype(bool))
    if (smoothed.shape[0] < 2):
        smoothed = raw_path.astype(np.int32)

    sm_q = _path_quality_metrics(smoothed, spacing)
    path_fraction = float((len(best_nodes) / max(coords.shape[0], 1)))
    reexp = float(best_metrics.get("radius_reexpansion_fraction", 0.0))
    tip_radius = float(best_metrics.get("tip_radius_mm", 0.0))

    flags: List[str] = []
    if (endpoints.size > 14):
        flags.append("complex_skeleton")
    if ((root_ambiguity > 0.92) and (root_hint is None)):
        flags.append("ambiguous_root")
    if ((path_fraction < 0.10) and (endpoints.size > 4)):
        flags.append("small_main_path_fraction")
    if (reexp > 0.35):
        flags.append("radius_reexpansion")
    if (float(sm_q.get("tortuosity", 0.0)) > 3.5):
        flags.append("high_tortuosity")
    if (float(sm_q.get("turn_angle_p95_deg", 0.0)) > 80.0):
        flags.append("high_scale_turn_angle")
    if (smoothed.shape[0] < 10):
        flags.append("short_centerline")
    quality = ("pass" if not flags else ";".join(flags))

    topo = dict(sm_q)
    topo.update(
        {
            "skeleton_endpoint_count": float(endpoints.size),
            "candidate_path_count": float(candidate_count),
            "root_radius_mm": float(root_radius),
            "tip_radius_mm": float(tip_radius),
            "path_skeleton_fraction": float(path_fraction),
            "radius_reexpansion_fraction": float(reexp),
            "root_ambiguity_ratio": float(root_ambiguity),
            "root_endpoint_index": float(root_idx),
        }
    )
    method = (
        f"{skel_method}+root_to_tip_radius_weighted"
        f"|root={source}|endpoints={int(endpoints.size)}"
        f"|candidates={int(candidate_count)}|score={best_score:.3f}"
    )
    return smoothed.astype(np.int32), method, quality, topo

def extract_centerline_path(
    branch_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    branch_name: str = "",
    root_hint: Optional[np.ndarray] = None,
    root_source: str = "",
) -> Tuple[np.ndarray, str, str]:
    path, method, quality, _ = _extract_centerline_root_to_tip(
        branch_mask, spacing, branch_name = branch_name, root_hint = root_hint, root_source = root_source
    )
    return path, method, quality
