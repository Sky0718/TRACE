from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

import cta_legacy_backend_v1_0_15 as legacy

DEFAULT_CENTERLINE_STEP_MM = 0.55

PARENT_BRANCHES: Dict[str, Tuple[str, ...]] = {
    "LAD": ("LM",),
    "LCX": ("LM",),
    "D1": ("LAD",),
    "D2": ("LAD",),
    "RAMUS": ("LM", "LAD", "LCX"),
    "RAD": ("LM", "LAD", "LCX"),
}

def base_branch_name(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name).upper().strip())

def severity_interval(ratio: float) -> str:
    pct = float((np.clip(ratio, 0.0, 1.0) * 100.0))
    if (pct < 1.0):
        return "0%"
    if (pct < 25.0):
        return "1-24%"
    if (pct < 50.0):
        return "25-49%"
    if (pct < 70.0):
        return "50-69%"
    if (pct < 100.0):
        return "70-99%"
    return "100%"

@dataclass
class RootAssignment:
    branch: str
    source: str
    root_voxel_x: float
    root_voxel_y: float
    root_voxel_z: float
    nearest_distance_mm: float = float("nan")
    point_component_id: int = -1
    parent_branch: str = ""
    seam_exclusion_mm: float = 0.0

    @property
    def voxel(self) -> np.ndarray:
        return np.asarray(
            [self.root_voxel_x, self.root_voxel_y, self.root_voxel_z], dtype = np.float32
        )

@dataclass
class CenterlineResult:
    branch: str
    path_voxel: np.ndarray
    path_physical_mm: np.ndarray
    cumulative_mm: np.ndarray
    tangent: np.ndarray
    normal_u: np.ndarray
    normal_v: np.ndarray
    method: str
    quality_flag: str
    topology: Dict[str, Any] = field(default_factory = dict)
    root_assignment: Optional[RootAssignment] = None

    @property
    def ok(self) -> bool:
        return ((self.path_voxel.ndim == 2) and (self.path_voxel.shape[0] >= 2))

def _safe_normalize(vec: np.ndarray, fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype = np.float64)
    n = float(np.linalg.norm(arr))
    if (n > 1e-10):
        return (arr / n)
    fb = np.asarray(fallback, dtype = np.float64)
    return (fb / max(float(np.linalg.norm(fb)), 1e-10))

def _physical_distance(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> float:
    return float(np.linalg.norm(((np.asarray(a) - np.asarray(b)) * np.asarray(spacing))))

def _crop_mask(mask: np.ndarray, pad: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(np.asarray(mask).astype(bool))
    if (coords.shape[0] == 0):
        return np.zeros((0, 0, 0), dtype = bool), np.zeros(3, dtype = np.int32)
    lo = np.maximum((coords.min(axis = 0) - int(pad)), 0)
    hi = np.minimum(((coords.max(axis = 0) + int(pad)) + 1), np.asarray(mask.shape))
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    return np.ascontiguousarray(mask[sl].astype(bool)), lo.astype(np.int32)

def _resample_smooth_path(
    path_voxel: np.ndarray, spacing: Sequence[float], step_mm: float, sigma_mm: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(path_voxel, dtype = np.float64)
    if (path.shape[0] < 2):
        return (
            path.astype(np.float32),
            path.astype(np.float32),
            np.zeros(path.shape[0], dtype = np.float32),
        )
    sp = np.asarray(spacing, dtype = np.float64)
    phys = (path * sp[None, :])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(phys, axis = 0), axis = 1))])
    total = float(cum[-1])
    count = max(2, (int(math.ceil((total / max(step_mm, 0.2)))) + 1))
    target = np.linspace(0.0, total, count)
    out_phys = np.column_stack([np.interp(target, cum, phys[:, d]) for d in range(3)])
    actual_step = (total / max((count - 1), 1))
    if (count >= 5):
        sigma = max((sigma_mm / max(actual_step, 1e-4)), 0.5)
        for d in range(3):
            out_phys[:, d] = gaussian_filter1d(out_phys[:, d], sigma = sigma, mode = "nearest")
    out_vox = (out_phys / sp[None, :])
    cum2 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(out_phys, axis = 0), axis = 1))])
    return out_vox.astype(np.float32), out_phys.astype(np.float32), cum2.astype(np.float32)

def parallel_transport_frames(
    path_physical_mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phys = np.asarray(path_physical_mm, dtype = np.float64)
    n = phys.shape[0]
    if (n < 2):
        z = np.zeros((n, 3), dtype = np.float32)
        return z, z, z
    tangent = np.asarray([_safe_normalize(t) for t in np.gradient(phys, axis = 0)])
    u = np.zeros_like(tangent)
    v = np.zeros_like(tangent)
    refs = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    ref = min(refs, key = lambda r: abs(float(np.dot(tangent[0], r))))
    u[0] = _safe_normalize(np.cross(tangent[0], ref))
    v[0] = _safe_normalize(np.cross(tangent[0], u[0]), (0.0, 1.0, 0.0))
    for i in range(1, n):
        projected = (u[(i - 1)] - (np.dot(u[(i - 1)], tangent[i]) * tangent[i]))
        if (np.linalg.norm(projected) < 1e-6):
            ref = min(refs, key = lambda r: abs(float(np.dot(tangent[i], r))))
            projected = np.cross(tangent[i], ref)
        u[i] = _safe_normalize(projected, u[(i - 1)])
        v[i] = _safe_normalize(np.cross(tangent[i], u[i]), v[(i - 1)])
        u[i] = _safe_normalize(np.cross(v[i], tangent[i]), u[i])
    return tangent.astype(np.float32), u.astype(np.float32), v.astype(np.float32)

def extract_rooted_centerline(
    branch: str,
    branch_mask: np.ndarray,
    spacing: Sequence[float],
    root_assignment: Optional[RootAssignment] = None,
    step_mm: float = DEFAULT_CENTERLINE_STEP_MM,
) -> CenterlineResult:
    crop, origin = _crop_mask(branch_mask, 3)
    if (crop.size == 0):
        e = np.zeros((0, 3), dtype = np.float32)
        return CenterlineResult(branch, e, e, np.zeros(0), e, e, e, "empty", "empty_mask")
    root_local = (
        (root_assignment.voxel - origin.astype(np.float32))
        if (root_assignment is not None)
        else None
    )
    root_source = (
        root_assignment.source if (root_assignment is not None) else "largest_radius_endpoint"
    )
    try:
        path_local, method, quality, topology = legacy._extract_centerline_root_to_tip(
            crop,
            tuple(float(v) for v in spacing),
            branch_name = branch,
            root_hint = root_local,
            root_source = root_source,
        )
    except Exception as exc:
        path_local, method, quality = legacy.extract_centerline_path(
            crop, tuple(float(v) for v in spacing)
        )
        topology = {"fallback_exception": str(exc)}
    path = (np.asarray(path_local, dtype = np.float32) + origin.astype(np.float32)[None, :])
    if ((path.shape[0] >= 2) and (root_assignment is not None)):
        if (_physical_distance(path[-1], root_assignment.voxel, spacing) < _physical_distance(
            path[0], root_assignment.voxel, spacing
        )):
            path = path[::-1].copy()
    path_vox, path_phys, cum = _resample_smooth_path(path, spacing, step_mm)
    if ((path_vox.shape[0] >= 2) and (root_assignment is not None)):
        if (_physical_distance(path_vox[-1], root_assignment.voxel, spacing) < _physical_distance(
            path_vox[0], root_assignment.voxel, spacing
        )):
            path_vox, path_phys = path_vox[::-1].copy(), path_phys[::-1].copy()
            cum = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(path_phys, axis = 0), axis = 1))]
            ).astype(np.float32)
    tangent, u, v = parallel_transport_frames(path_phys)
    topology = dict(topology)
    if (path_phys.shape[0] >= 2):
        length = float(cum[-1])
        endpoint = float(np.linalg.norm((path_phys[-1] - path_phys[0])))
        topology.update(
            {
                "resampled_length_mm": length,
                "endpoint_distance_mm": endpoint,
                "tortuosity": (length / max(endpoint, 1e-6)),
                "root_source": root_source,
            }
        )
    return CenterlineResult(
        branch,
        path_vox,
        path_phys,
        cum,
        tangent,
        u,
        v,
        str(method),
        str(quality),
        topology,
        root_assignment,
    )

def _plane_grid(half_width_mm: float, pixel_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-half_width_mm, (half_width_mm + (0.5 * pixel_mm)), pixel_mm, dtype = np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing = "xy")
    return axis, uu, vv

def sample_plane_batch(
    volume: np.ndarray,
    centers_physical_mm: np.ndarray,
    normals_u: np.ndarray,
    normals_v: np.ndarray,
    spacing: Sequence[float],
    half_width_mm: float,
    pixel_mm: float,
    order: int,
    cval: float,
) -> Tuple[np.ndarray, np.ndarray]:
    axis, uu, vv = _plane_grid(half_width_mm, pixel_mm)
    centers = np.asarray(centers_physical_mm, dtype = np.float64)
    u = np.asarray(normals_u, dtype = np.float64)
    v = np.asarray(normals_v, dtype = np.float64)
    phys = ((centers[:, None, None, :] + (uu[None, ..., None] * u[:, None, None, :])) + (
        vv[None, ..., None] * v[:, None, None, :]
    ))
    vox = (phys / np.asarray(spacing, dtype = np.float64)[None, None, None, :])
    sampled = map_coordinates(
        np.asarray(volume),
        [vox[..., 0].ravel(), vox[..., 1].ravel(), vox[..., 2].ravel()],
        order = order,
        mode = "constant",
        cval = float(cval),
        prefilter = bool((order > 1)),
    ).reshape(vox.shape[:-1])
    return np.asarray(sampled), axis

def _robust_location_scale(values: np.ndarray, minimum_scale: float = 5.0) -> Tuple[float, float]:
    vals = np.asarray(values, dtype = np.float64)
    vals = vals[np.isfinite(vals)]
    if (vals.size == 0):
        return 0.0, minimum_scale
    med = float(np.median(vals))
    mad = float(np.median(np.abs((vals - med))))
    return med, max((1.4826 * mad), minimum_scale)

def estimate_branch_lumen_reference(
    image: np.ndarray, centerline: CenterlineResult, spacing: Sequence[float]
) -> Dict[str, float]:
    path = centerline.path_voxel
    if (path.shape[0] == 0):
        return {"mean_hu": 250.0, "std_hu": 40.0, "sample_count": 0}
    vals = map_coordinates(
        np.asarray(image, dtype = np.float32),
        [path[:, 0], path[:, 1], path[:, 2]],
        order = 1,
        mode = "nearest",
    )
    clean = vals[((np.isfinite(vals) & (vals > 40.0)) & (vals < 1000.0))]
    if (clean.size == 0):
        return {"mean_hu": 250.0, "std_hu": 50.0, "sample_count": 0}
    lo, hi = np.percentile(clean, [45.0, 92.0])
    trimmed = clean[((clean >= lo) & (clean <= hi))]
    if (trimmed.size < 5):
        trimmed = clean
    mu, sigma = _robust_location_scale(trimmed, 12.0)
    return {"mean_hu": mu, "std_hu": sigma, "sample_count": int(trimmed.size)}
