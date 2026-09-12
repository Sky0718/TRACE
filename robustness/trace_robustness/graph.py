from __future__ import annotations
from typing import Dict
from typing import List
from typing import Tuple
import heapq
import math
import numpy as np
from . import runtime, measurements, topology

def skeletonize_3d_safe(mask: np.ndarray) -> Tuple[np.ndarray, str]:
    m = mask.astype(bool)
    if not np.any(m):
        return (np.zeros_like(m, dtype = bool), "empty_mask")
    if not runtime.SKIMAGE_SKELETON_OK:
        return (skeletonize_2d_fallback(m), "slice_wise_2d_fallback_no_skeletonize")
    try:
        sk = runtime.skeletonize(m, method = "lee")
        return (sk.astype(bool), "skimage_skeletonize_lee_3d")
    except Exception:
        try:
            sk = runtime.skeletonize(m)
            if (sk.shape == m.shape):
                return (sk.astype(bool), "skimage_skeletonize_default")
        except Exception:
            pass
    return (skeletonize_2d_fallback(m), "slice_wise_2d_fallback")

def skeletonize_2d_fallback(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype = bool)
    if not runtime.SKIMAGE_SKELETON_OK:
        return out
    for z in range(mask.shape[2]):
        if np.any(mask[:, :, z]):
            try:
                out[:, :, z] = runtime.skeletonize(mask[:, :, z].astype(bool)).astype(bool)
            except Exception:
                pass
    return out

def largest_skeleton_component(skel: np.ndarray) -> np.ndarray:
    (labeled, num) = measurements.connected_components_3d(skel.astype(bool))
    if (num == 0):
        return np.zeros_like(skel, dtype = bool)
    sizes = np.bincount(labeled.ravel())[1:]
    keep = (int(np.argmax(sizes)) + 1)
    return (labeled == keep)

def build_graph_from_skeleton(
    skel: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int], List[List[Tuple[int, float]]]]:
    coords = np.argwhere(skel.astype(bool)).astype(np.int32)
    index = {(int(x), int(y), int(z)): i for (i, (x, y, z)) in enumerate(coords.tolist())}
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(coords.shape[0])]
    (sx, sy, sz) = map(float, spacing)
    shape = skel.shape
    for i, (x, y, z) in enumerate(coords.tolist()):
        for dx, dy, dz in runtime.NEIGHBORS_26:
            (nx, ny, nz) = (int((x + dx)), int((y + dy)), int((z + dz)))
            if (((0 <= nx < shape[0]) and (0 <= ny < shape[1])) and (0 <= nz < shape[2])):
                j = index.get((nx, ny, nz))
                if (j is not None):
                    w = math.sqrt(((((dx * sx) ** 2) + ((dy * sy) ** 2)) + ((dz * sz) ** 2)))
                    adj[i].append((j, float(w)))
    return (coords, index, adj)

def dijkstra_farthest(
    adj: List[List[Tuple[int, float]]], start: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    n = len(adj)
    dist = np.full(n, np.inf, dtype = np.float64)
    parent = np.full(n, -1, dtype = np.int32)
    dist[int(start)] = 0.0
    pq = [(0.0, int(start))]
    while pq:
        (d, u) = heapq.heappop(pq)
        if (d > dist[u]):
            continue
        for v, w in adj[u]:
            nd = (d + float(w))
            if (nd < dist[v]):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    finite = np.where(np.isfinite(dist))[0]
    if (finite.size == 0):
        return (int(start), dist, parent)
    farthest = int(finite[np.argmax(dist[finite])])
    return (farthest, dist, parent)

def reconstruct_graph_path(parent: np.ndarray, start: int, end: int) -> List[int]:
    path = []
    cur = int(end)
    while (cur >= 0):
        path.append(cur)
        if (cur == int(start)):
            break
        cur = int(parent[cur])
    if ((len(path) == 0) or (path[-1] != int(start))):
        return []
    path.reverse()
    return path

def extract_centerline_diameter_path(
    branch_mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, str, str]:
    if (int(np.sum(branch_mask)) < 8):
        return (np.zeros((0, 3), dtype = np.int32), "too_small", "too_small_branch_mask")
    (skel, skel_method) = skeletonize_3d_safe(branch_mask)
    skel = largest_skeleton_component(skel)
    (coords, index, adj) = build_graph_from_skeleton(skel, spacing)
    if (coords.shape[0] < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (approx, "centroid_axis_fallback", "skeleton_too_small")
    degrees = np.array([len(a) for a in adj], dtype = np.int32)
    endpoints = np.where((degrees <= 1))[0]
    if (endpoints.size >= 2):
        start0 = int(endpoints[0])
    else:
        start0 = 0
    (a, _, _) = dijkstra_farthest(adj, start0)
    (b, dist_b, parent_b) = dijkstra_farthest(adj, a)
    node_path = reconstruct_graph_path(parent_b, a, b)
    if (len(node_path) < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (approx, "centroid_axis_fallback", "graph_path_failed")
    return (
        coords[np.asarray(node_path, dtype = np.int32)],
        (skel_method + "+graph_diameter_path"),
        "ok",
    )

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
        physical = (
            coords.astype(np.float32) * np.asarray(spacing, dtype = np.float32)[None, :]
        )
        center = physical.mean(axis = 0)
        (u, s, vt) = np.linalg.svd((physical - center), full_matrices = False)
        axis = vt[0]
        proj = ((physical - center) @ axis)
        order = np.argsort(proj)
        pts = coords[order]
        if (pts.shape[0] > 80):
            pts = pts[np.linspace(0, (pts.shape[0] - 1), 80).round().astype(np.int32)]
        return pts.astype(np.int32)
    return np.asarray(points, dtype = np.int32)

def _stable_endpoint_path(
    coords: np.ndarray,
    adj: List[List[Tuple[int, float]]],
    spacing: Tuple[float, float, float],
) -> List[int]:
    if (coords.shape[0] < 2):
        return []
    degrees = np.array([len(a) for a in adj], dtype = np.int32)
    endpoints = np.where((degrees <= 1))[0]
    if (endpoints.size < 2):
        endpoints = np.arange(coords.shape[0], dtype = np.int32)
    phys = topology._physical_coords(coords, spacing)
    ep_phys = phys[endpoints]
    if (endpoints.size > 40):
        chosen = [int(np.argmax(np.linalg.norm((ep_phys - ep_phys.mean(axis = 0)), axis = 1)))]
        while (len(chosen) < 40):
            dmin = np.full(endpoints.size, np.inf, dtype = np.float32)
            for ci in chosen:
                dmin = np.minimum(dmin, np.linalg.norm((ep_phys - ep_phys[ci]), axis = 1))
            chosen.append(int(np.argmax(dmin)))
        endpoints = endpoints[np.asarray(chosen, dtype = np.int32)]
        ep_phys = phys[endpoints]
    if (endpoints.size < 2):
        return []
    chord_mat_max = 0.0
    for i in range(endpoints.size):
        d = (
            np.linalg.norm((ep_phys[(i + 1) :] - ep_phys[i]), axis = 1)
            if ((i + 1) < endpoints.size)
            else np.array([])
        )
        if d.size:
            chord_mat_max = max(chord_mat_max, float(np.max(d)))
    min_chord = max(8.0, (0.22 * chord_mat_max))
    best = None
    best_score = -np.inf
    fallback_best = None
    fallback_chord = -np.inf
    for si, start in enumerate(endpoints):
        (far, dist, parent) = dijkstra_farthest(adj, int(start))
        for end in endpoints[(si + 1) :]:
            length = (
                float(dist[int(end)])
                if ((int(end) < dist.size) and np.isfinite(dist[int(end)]))
                else np.inf
            )
            if (not np.isfinite(length) or (length <= 0)):
                continue
            chord = float(np.linalg.norm((phys[int(start)] - phys[int(end)])))
            tort = (length / max(chord, 1e-06))
            if (chord > fallback_chord):
                path_fb = reconstruct_graph_path(parent, int(start), int(end))
                if (len(path_fb) >= 2):
                    fallback_best = path_fb
                    fallback_chord = chord
            eligible = ((chord >= min_chord) and (tort <= 12.0))
            score = (((0.7 * length) + (0.3 * chord)) - ((max(0.0, (tort - 5.0)) * 0.08) * length))
            if (eligible and (score > best_score)):
                path = reconstruct_graph_path(parent, int(start), int(end))
                if (len(path) >= 2):
                    best = path
                    best_score = score
    if (best is not None):
        return best
    return fallback_best if (fallback_best is not None) else []

def extract_centerline_path(
    branch_mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, str, str]:
    if (int(np.sum(branch_mask)) < 8):
        return (np.zeros((0, 3), dtype = np.int32), "too_small", "too_small_branch_mask")
    (skel, skel_method) = skeletonize_3d_safe(branch_mask)
    skel = largest_skeleton_component(skel)
    (coords, index, adj) = build_graph_from_skeleton(skel, spacing)
    if (coords.shape[0] < 2):
        approx = centroid_axis_fallback_path(branch_mask, spacing)
        return (approx, "centroid_axis_fallback", "skeleton_too_small")
    node_path = _stable_endpoint_path(coords, adj, spacing)
    if (len(node_path) < 2):
        return extract_centerline_diameter_path(branch_mask, spacing)
    return (
        coords[np.asarray(node_path, dtype = np.int32)],
        (skel_method + "+stable_endpoint_graph_path"),
        "ok",
    )

def _dijkstra_all(
    adj: List[List[Tuple[int, float]]], start: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    dist = np.full(n, np.inf, dtype = np.float64)
    parent = np.full(n, -1, dtype = np.int32)
    dist[int(start)] = 0.0
    pq = [(0.0, int(start))]
    while pq:
        (d, u) = heapq.heappop(pq)
        if (d > dist[u]):
            continue
        for v, w in adj[u]:
            nd = (d + float(w))
            if (nd < dist[v]):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return (dist, parent)
