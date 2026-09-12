from __future__ import annotations

import re
import gc
import zipfile
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from skimage.measure import label as skimage_label

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

try:
    import pydicom

    PYDICOM_OK = True
except Exception:
    pydicom = None
    PYDICOM_OK = False

from legacy_centerline import connected_components_3d

DEFAULT_MIN_BRANCH_COMPONENT_VOXELS = 1500

DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX = 20

BRANCH_RGB = {
    "RCA": (155, 89, 182),
    "LM": (241, 196, 15),
    "LAD": (46, 204, 113),
    "LCX": (52, 152, 219),
    "RAMUS": (216, 101, 79),
    "RAD": (216, 101, 79),
    "D1": (221, 130, 101),
    "D2": (190, 115, 95),
}

def ensure_dir(path: (Path | str)) -> Path:
    p = Path(path)
    p.mkdir(parents = True, exist_ok = True)
    return p

def agatston_grade(score: float) -> Tuple[str, str]:
    s = max(0.0, float(score))
    if (s == 0.0):
        return "none", "very_low_risk"
    if (s <= 100.0):
        return "mild", "low_risk"
    if (s <= 400.0):
        return "moderate", "intermediate_risk"
    return "severe", "high_risk"

def split_left_coronary(
    left_mask: np.ndarray, bifurcation_min_area_px: int = DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX
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
        return lm_mask, lad_mask, lcx_mask, info

    bif_z = None
    prev_lad = None
    prev_lcx = None

    for z in active_z:
        labeled, num = skimage_label(
            left_mask[:, :, z], background = 0, return_num = True, connectivity = 2
        )
        comps = []
        for cid in range(1, (num + 1)):
            region = (labeled == cid)
            area = int(np.sum(region))
            if (area < int(bifurcation_min_area_px)):
                continue
            ys, xs = np.where(region)
            comps.append((cid, area, np.array([ys.mean(), xs.mean()], dtype = np.float32)))
        if (len(comps) >= 2):
            comps.sort(key = lambda x: x[1], reverse = True)
            c1, c2 = comps[0], comps[1]
            pair = sorted([c1, c2], key = lambda x: x[2][0])
            bif_z = int(z)
            prev_lad = pair[0][2]
            prev_lcx = pair[1][2]
            lad_mask[:, :, z] = (labeled == pair[0][0])
            lcx_mask[:, :, z] = (labeled == pair[1][0])
            break

    if (bif_z is None):
        lad_mask[:] = left_mask
        info["warning"] = "left_bifurcation_not_detected; entire_left_component_assigned_to_LAD"
        return lm_mask, lad_mask, lcx_mask, info

    info["bifurcation_z"] = int(bif_z)
    pre_z = active_z[(active_z < bif_z)]
    lm_mask[:, :, pre_z] = left_mask[:, :, pre_z]

    for z in active_z[(active_z > bif_z)]:
        labeled, num = skimage_label(
            left_mask[:, :, z], background = 0, return_num = True, connectivity = 2
        )
        comps = []
        for cid in range(1, (num + 1)):
            region = (labeled == cid)
            if (int(np.sum(region)) < int(bifurcation_min_area_px)):
                continue
            ys, xs = np.where(region)
            centroid = np.array([ys.mean(), xs.mean()], dtype = np.float32)
            comps.append((cid, region, centroid))
        for cid, region, centroid in comps:
            d_lad = (float(np.linalg.norm((centroid - prev_lad))) if (prev_lad is not None) else 0.0)
            d_lcx = (float(np.linalg.norm((centroid - prev_lcx))) if (prev_lcx is not None) else 0.0)
            if (d_lad <= d_lcx):
                lad_mask[:, :, z] |= region
                prev_lad = centroid
            else:
                lcx_mask[:, :, z] |= region
                prev_lcx = centroid

    return lm_mask, lad_mask, lcx_mask, info

def separate_branches(
    vessel_mask: np.ndarray, min_branch_voxels: int = DEFAULT_MIN_BRANCH_COMPONENT_VOXELS
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    labeled, num = connected_components_3d(vessel_mask)
    info = {
        "method": "3d_connected_components_volume_sort_plus_left_centroid_tracking",
        "warnings": [],
    }
    if (num == 0):
        info["warnings"].append("no_vessel_component")
        return {}, info

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
        lm, lad, lcx, left_info = split_left_coronary(left_mask)
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

    return branches, info

def branch_sort_key(branch: str) -> Tuple[int, str]:
    order = {"LM": 1, "LAD": 2, "D1": 3, "LCX": 4, "RAMUS": 5, "RAD": 6, "RCA": 7}
    b = str(branch).upper()
    return (order.get(b, 100), b)

def branch_display_color(branch: str) -> Tuple[int, int, int]:
    key = str(branch).upper()

    key = re.sub(r"_\d+$", "", key)

    colors = {
        "RCA": (155, 89, 182),
        "LM": (241, 196, 15),
        "LAD": (46, 204, 113),
        "LCX": (52, 152, 219),
        "RAMUS": (216, 101, 79),
        "RAD": (216, 101, 79),
        "D1": (221, 130, 101),
        "D2": (190, 115, 95),
    }

    return colors.get(key, (149, 165, 166))

def discover_dicom_series(dicom_folder: Path) -> Dict[str, List[Path]]:
    if not PYDICOM_OK:
        raise RuntimeError(
            "pydicom is required for CTA DICOM folder input. Install with: pip install pydicom"
        )
    folder = Path(dicom_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"DICOM folder does not exist: {folder}")
    series: Dict[str, List[Path]] = {}
    for fp in sorted(folder.rglob("*"), key = lambda p: str(p).lower()):
        if not fp.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels = True, force = True)
            if not (
                hasattr(ds, "SeriesInstanceUID") and hasattr(ds, "Rows") and hasattr(ds, "Columns")
            ):
                continue
            sid = str(getattr(ds, "SeriesInstanceUID", "unknown_series"))
            series.setdefault(sid, []).append(fp)
        except Exception:
            continue
    if not series:
        raise FileNotFoundError(f"No readable DICOM slices were found in: {folder}")
    return series

def _dicom_slice_sort_positions(headers: List[Any]) -> Optional[np.ndarray]:
    try:
        first = headers[0]
        iop = np.asarray([float(v) for v in first.ImageOrientationPatient], dtype = np.float64)
        row_cos = iop[:3]
        col_cos = iop[3:]
        normal = np.cross(row_cos, col_cos)
        positions = []
        for ds in headers:
            ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype = np.float64)
            positions.append(float(np.dot(ipp, normal)))
        return np.asarray(positions, dtype = np.float64)
    except Exception:
        return None

def _dicom_instance_number(ds: Any, fallback: int) -> int:
    try:
        return int(getattr(ds, "InstanceNumber"))
    except Exception:
        return int(fallback)

def load_dicom_volume(
    dicom_folder: Path, preferred_series_uid: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    series = discover_dicom_series(Path(dicom_folder))
    if (preferred_series_uid and (preferred_series_uid in series)):
        chosen_uid = str(preferred_series_uid)
    else:
        chosen_uid = max(series.keys(), key = lambda sid: len(series[sid]))
    files = list(series[chosen_uid])
    headers = [pydicom.dcmread(str(fp), stop_before_pixels = True, force = True) for fp in files]
    positions = _dicom_slice_sort_positions(headers)
    if ((positions is not None) and (positions.size == len(files))):
        order = np.argsort(positions)
        sorted_positions = positions[order]
    else:
        order = np.argsort([_dicom_instance_number(ds, i) for i, ds in enumerate(headers)])
        sorted_positions = None
    files = [files[int(i)] for i in order]
    headers = [headers[int(i)] for i in order]

    slices: List[np.ndarray] = []
    for fp, _ in zip(files, headers):
        ds = pydicom.dcmread(str(fp), force = True)
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode DICOM pixels: {fp}\n"
                "If this is a compressed DICOM, install pixel decoding support such as pylibjpeg pylibjpeg-libjpeg.\n"
                f"Original error: {exc}"
            )
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = ((arr * slope) + intercept)
        if (arr.ndim == 2):
            slices.append(arr.astype(np.float32))
        elif (arr.ndim == 3):
            for k in range(arr.shape[0]):
                slices.append(arr[k].astype(np.float32))
        else:
            raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape} in {fp}")

    if not slices:
        raise RuntimeError(f"No pixel data loaded from series {chosen_uid}")
    volume = np.stack(slices, axis = 2).astype(np.float32)

    first = headers[0]
    try:
        row_spacing = float(first.PixelSpacing[0])
        col_spacing = float(first.PixelSpacing[1])
    except Exception:
        row_spacing, col_spacing = 1.0, 1.0
    if ((sorted_positions is not None) and (sorted_positions.size >= 2)):
        diffs = np.diff(sorted_positions)
        diffs = diffs[(np.abs(diffs) > 1e-6)]
        slice_spacing = (
            float(np.median(np.abs(diffs)))
            if diffs.size
            else float(getattr(first, "SliceThickness", 1.0))
        )
    else:
        try:
            slice_spacing = float(getattr(first, "SpacingBetweenSlices"))
        except Exception:
            slice_spacing = (
                float(getattr(first, "SliceThickness", 1.0))
                if hasattr(first, "SliceThickness")
                else 1.0
            )
    spacing = (float(row_spacing), float(col_spacing), float(abs(slice_spacing)))
    meta = {
        "dicom_folder": str(dicom_folder),
        "series_uid": chosen_uid,
        "series_count": int(len(series)),
        "series_slice_counts": {str(k): int(len(v)) for k, v in series.items()},
        "selected_file_count": int(len(files)),
        "volume_shape": tuple(int(x) for x in volume.shape),
        "spacing": spacing,
        "first_file": str(files[0]),
        "last_file": str(files[-1]),
    }
    return volume, spacing, meta

PUBLICATION_BG = "white"

PUBLICATION_FG = "black"

def label_projection_rgb(
    multilabel: np.ndarray,
    label_map: Dict[str, int],
    axis: int = 2,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    proj = np.max(np.asarray(multilabel, dtype = np.uint16), axis = int(axis))
    rgb = np.empty((proj.shape + (3,)), dtype = np.uint8)
    rgb[:] = np.asarray(background, dtype = np.uint8)
    for branch, value in label_map.items():
        color = branch_display_color(branch)
        rgb[(proj == int(value))] = np.asarray(color, dtype = np.uint8)
    return np.rot90(rgb)

def parse_segmentation_color_table(path: Path) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for raw in Path(path).read_text(encoding = "utf-8-sig", errors = "replace").splitlines():
        line = raw.strip()
        if (not line or line.startswith("#")):
            continue
        tokens = line.split()
        if (len(tokens) < 6):
            continue
        try:
            value = int(tokens[0])
            rgba = [int(v) for v in tokens[-4:]]
        except Exception:
            continue
        name = " ".join(tokens[1:-4]).strip()
        if not name:
            name = f"LABEL_{value}"
        entries[value] = {
            "value": int(value),
            "name": str(name),
            "rgba": tuple(int(np.clip(v, 0, 255)) for v in rgba),
        }
    if not entries:
        raise ValueError(f"No label entries could be parsed from color table: {path}")
    return dict(sorted(entries.items(), key = lambda kv: int(kv[0])))

def normalize_packaged_branch_name(name: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().upper()).strip("_")
    aliases = {
        "RAMUS_INTERMEDIUS": "RAMUS",
        "RI": "RAMUS",
        "DIAGONAL_1": "D1",
        "DIAGONAL1": "D1",
    }
    return aliases.get(raw, (raw if raw else "UNKNOWN"))

def is_vessel_color_table_label(name: str) -> bool:
    upper = str(name).strip().upper()
    excluded_tokens = (
        "POINT",
        "MARKER",
        "LESION",
        "STENOSIS",
        "BACKGROUND",
        "UNLABELED",
        "UNLABELLED",
        "NONE",
        "AIR",
    )
    return not any((token in upper) for token in excluded_tokens)

def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = ensure_dir(destination).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if ((target != destination) and (destination not in target.parents)):
                raise ValueError(f"Unsafe path in ZIP archive: {info.filename}")
        zf.extractall(destination)

def align_multilabel_to_dicom(
    multilabel: np.ndarray, image: np.ndarray, vessel_values: List[int]
) -> Tuple[np.ndarray, str, float]:
    src = np.asarray(multilabel)
    if not np.issubdtype(src.dtype, np.integer):
        src = np.rint(src).astype(np.int32)
    target_shape = tuple(int(v) for v in image.shape)
    valid_values = np.asarray([int(v) for v in vessel_values], dtype = np.int32)
    source_mask = np.isin(src, valid_values)
    source_coords = np.argwhere(source_mask)
    if (source_coords.shape[0] == 0):
        raise ValueError(
            "The segmentation contains none of the vessel values listed in the color table."
        )
    if (source_coords.shape[0] > 120000):
        take = np.linspace(0, (source_coords.shape[0] - 1), 120000, dtype = np.int64)
        sample_coords = source_coords[take]
    else:
        sample_coords = source_coords

    candidates: List[Tuple[float, Tuple[int, int, int], Tuple[bool, bool, bool]]] = []
    for perm in itertools.permutations(range(3)):
        if (tuple(src.shape[i] for i in perm) != target_shape):
            continue
        perm_coords = sample_coords[:, list(perm)].astype(np.int64, copy = True)
        for flips in itertools.product((False, True), repeat = 3):
            coords = perm_coords.copy()
            for axis, flip in enumerate(flips):
                if flip:
                    coords[:, axis] = ((target_shape[axis] - 1) - coords[:, axis])
            vals = image[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
            finite = vals[np.isfinite(vals)]
            if (finite.size == 0):
                score = -1e12
            else:
                clipped = np.clip(finite, -300.0, 1200.0)
                score = ((
                    float(np.median(clipped)) + (90.0 * float(np.mean(((clipped >= 80.0) & (clipped <= 1000.0)))))
                ) - (120.0 * float(np.mean((clipped < -200.0)))))
            candidates.append(
                (float(score), tuple(int(x) for x in perm), tuple(bool(x) for x in flips))
            )
    if not candidates:
        raise ValueError(
            f"Segmentation/DICOM shape mismatch: segmentation={src.shape}, DICOM={target_shape}. "
            "No axis permutation can make the shapes agree."
        )
    best_score, best_perm, best_flips = max(candidates, key = lambda x: x[0])
    del source_mask, source_coords, sample_coords, candidates
    gc.collect()
    aligned = np.transpose(src, best_perm)
    for axis, flip in enumerate(best_flips):
        if flip:
            aligned = np.flip(aligned, axis = axis)
    aligned = np.ascontiguousarray(aligned)
    method = f"axis_permutation_{best_perm}_flips_{best_flips}_hu_overlap_selection"
    return aligned, method, float(best_score)

def voxel_surface_area_mm2(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    m = np.asarray(mask).astype(np.uint8)
    sx, sy, sz = map(float, spacing)
    x_faces = int(np.count_nonzero(np.diff(np.pad(m, ((1, 1), (0, 0), (0, 0))), axis = 0)))
    y_faces = int(np.count_nonzero(np.diff(np.pad(m, ((0, 0), (1, 1), (0, 0))), axis = 1)))
    z_faces = int(np.count_nonzero(np.diff(np.pad(m, ((0, 0), (0, 0), (1, 1))), axis = 2)))
    return float(((((x_faces * sy) * sz) + ((y_faces * sx) * sz)) + ((z_faces * sx) * sy)))

STENOSIS_LABEL_PALETTE = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (166, 86, 40),
    (247, 129, 191),
    (153, 153, 51),
    (102, 194, 165),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
    (229, 196, 148),
    (179, 179, 179),
]

def _stenosis_interval_short(ratio: float) -> str:
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
    return "99-100%"

def _draw_side_labels(
    ax,
    pts: List[Tuple[float, float]],
    labels: List[str],
    colors: List[Tuple[int, int, int]],
    img_shape: Tuple[int, int],
) -> None:
    if not pts:
        return
    h, w = img_shape[:2]
    left_idx = [i for i, p in enumerate(pts) if (p[0] < (w / 2.0))]
    right_idx = [i for i, p in enumerate(pts) if (p[0] >= (w / 2.0))]
    for side, idxs in [("left", left_idx), ("right", right_idx)]:
        if not idxs:
            continue
        idxs = sorted(idxs, key = lambda i: pts[i][1])
        ys = np.linspace((0.12 * h), (0.88 * h), len(idxs))
        x_text = ((0.05 * w) if (side == "left") else (0.95 * w))
        ha = ("left" if (side == "left") else "right")
        for j, i in enumerate(idxs):
            x0, y0 = pts[i]
            yt = float(ys[j])
            xt = float(x_text)
            c = (np.asarray(colors[i], dtype = np.float32) / 255.0)
            ax.plot([x0, (0.5 * (x0 + xt)), xt], [y0, yt, yt], color = c, linewidth = 2.0)
            ax.text(
                xt,
                yt,
                labels[i],
                color = c,
                fontsize = 11.5,
                fontweight = "bold",
                va = "center",
                ha = ha,
                bbox = dict(boxstyle = "round,pad=0.22", fc = "white", ec = c, lw = 2.0, alpha = 0.96),
            )

def _content_bbox_from_rgb(rgb: np.ndarray, pad: int = 10) -> Tuple[int, int, int, int]:
    arr = np.asarray(rgb)
    if (arr.ndim == 2):
        fg = (arr > 0)
    else:
        fg = np.any((arr > 0), axis = -1)
    h, w = fg.shape[:2]
    ys, xs = np.where(fg)
    if ((xs.size == 0) or (ys.size == 0)):
        return (0, 0, int(w), int(h))
    x0 = max(0, (int(xs.min()) - int(pad)))
    x1 = min(int(w), ((int(xs.max()) + int(pad)) + 1))
    y0 = max(0, (int(ys.min()) - int(pad)))
    y1 = min(int(h), ((int(ys.max()) + int(pad)) + 1))
    return (x0, y0, x1, y1)

def _crop_array_with_bbox(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if (np.asarray(arr).ndim == 2):
        return np.asarray(arr)[y0:y1, x0:x1]
    return np.asarray(arr)[y0:y1, x0:x1, ...]

def _project_rotated_points(
    points: List[Tuple[int, int, int]], axis: int, base_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    pts2d: List[Tuple[float, float]] = []
    if (points is None):
        return pts2d
    width = (int(base_shape[1]) if (len(base_shape) >= 2) else 0)
    for p in points:
        x, y, z = [int(v) for v in p]
        if (int(axis) == 2):
            r, c = x, y
        elif (int(axis) == 1):
            r, c = x, z
        else:
            r, c = y, z
        pts2d.append((float(r), float(((width - 1) - c))))
    return pts2d

def _snap_point_to_mask(pt: Tuple[float, float], mask: Optional[np.ndarray]) -> Tuple[float, float]:
    if (mask is None):
        return pt
    mask_arr = np.asarray(mask).astype(bool)
    if ((mask_arr.ndim != 2) or (mask_arr.size == 0) or not np.any(mask_arr)):
        return pt
    h, w = mask_arr.shape
    x = float(np.clip(pt[0], 0, max((w - 1), 0)))
    y = float(np.clip(pt[1], 0, max((h - 1), 0)))
    xi = int(round(x))
    yi = int(round(y))
    xi = int(np.clip(xi, 0, max((w - 1), 0)))
    yi = int(np.clip(yi, 0, max((h - 1), 0)))
    if mask_arr[yi, xi]:
        return (float(xi), float(yi))
    ys, xs = np.where(mask_arr)
    if (xs.size == 0):
        return (x, y)
    dist2 = (((xs.astype(np.float32) - x) ** 2) + ((ys.astype(np.float32) - y) ** 2))
    k = int(np.argmin(dist2))
    return (float(xs[k]), float(ys[k]))

def _branch_legend_handles(label_map: Dict[str, int]) -> List[Any]:
    import matplotlib.patches as mpatches

    handles: List[Any] = []
    for branch, value in sorted(label_map.items(), key = lambda kv: int(kv[1])):
        c = (np.asarray(branch_display_color(branch), dtype = np.float32) / 255.0)
        handles.append(mpatches.Patch(color = c, label = f"{value} = {branch}"))
    return handles

def _composite_label_projection_rgb(
    multilabel: np.ndarray, label_map: Dict[str, int], axis: int
) -> np.ndarray:
    arr = np.asarray(multilabel, dtype = np.uint16)
    masks: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    for branch, value in sorted(label_map.items(), key = lambda kv: branch_sort_key(kv[0])):
        pmask = np.max((arr == int(value)).astype(np.uint8), axis = int(axis)).astype(bool)
        if np.any(pmask):
            masks.append(pmask)
            colors.append(np.asarray(branch_display_color(branch), dtype = np.float32))
    if not masks:
        proj_shape = np.max(arr, axis = int(axis)).shape
        return np.zeros((proj_shape[::-1] + (3,)), dtype = np.uint8)
    shape = masks[0].shape
    accum = np.zeros((shape + (3,)), dtype = np.float32)
    count = np.zeros(shape, dtype = np.float32)
    for mask, color in zip(masks, colors):
        accum[mask] += color
        count[mask] += 1.0
    rgb = np.zeros((shape + (3,)), dtype = np.uint8)
    fg = (count > 0)
    rgb[fg] = np.clip((accum[fg] / count[fg, None]), 0, 255).astype(np.uint8)
    return np.rot90(rgb)

def _projection_display_spacing(
    axis: int, spacing: Optional[Tuple[float, float, float]]
) -> Tuple[float, float]:
    if (spacing is None):
        return 1.0, 1.0
    sx, sy, sz = [float(max(v, 1e-6)) for v in spacing]
    if (int(axis) == 2):
        return sx, sy
    if (int(axis) == 1):
        return sx, sz
    return sy, sz

def _physical_resample_rgb_and_masks(
    rgb: np.ndarray,
    points: List[Tuple[float, float]],
    branch_masks: Dict[str, np.ndarray],
    axis: int,
    spacing: Optional[Tuple[float, float, float]],
) -> Tuple[np.ndarray, List[Tuple[float, float]], Dict[str, np.ndarray]]:
    if (spacing is None):
        return rgb, points, branch_masks
    from scipy.ndimage import zoom

    horizontal_mm, vertical_mm = _projection_display_spacing(axis, spacing)
    target = min(horizontal_mm, vertical_mm)
    zoom_x = float(np.clip((horizontal_mm / target), 0.5, 4.0))
    zoom_y = float(np.clip((vertical_mm / target), 0.5, 4.0))
    if ((abs((zoom_x - 1.0)) < 1e-3) and (abs((zoom_y - 1.0)) < 1e-3)):
        return rgb, points, branch_masks
    out_rgb = zoom(np.asarray(rgb), (zoom_y, zoom_x, 1.0), order = 0, mode = "nearest")
    out_pts = [(float((x * zoom_x)), float((y * zoom_y))) for x, y in points]
    out_masks: Dict[str, np.ndarray] = {}
    for name, mask in branch_masks.items():
        out_masks[name] = zoom(
            mask.astype(np.uint8), (zoom_y, zoom_x), order = 0, mode = "nearest"
        ).astype(bool)
    return np.asarray(out_rgb, dtype = np.uint8), out_pts, out_masks

def _multilabel_projection_bundle(
    multilabel: np.ndarray,
    label_map: Dict[str, int],
    axis: int,
    points: Optional[List[Tuple[int, int, int]]] = None,
    point_branches: Optional[List[str]] = None,
    pad: int = 10,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], Dict[str, np.ndarray], Tuple[int, int, int, int]]:
    arr = np.asarray(multilabel, dtype = np.uint16)
    proj = np.max(arr, axis = int(axis))
    base_shape = tuple(int(v) for v in proj.shape)
    rgb_full = _composite_label_projection_rgb(arr, label_map, axis)
    rot_pts = _project_rotated_points((points or []), axis, base_shape)
    branch_masks_full: Dict[str, np.ndarray] = {}
    if point_branches:
        for branch in sorted(set(point_branches), key = branch_sort_key):
            if (branch not in label_map):
                continue
            value = int(label_map[branch])
            pmask = np.max((arr == value).astype(np.uint8), axis = int(axis)).astype(bool)
            branch_masks_full[branch] = np.rot90(pmask)
    rgb_full, rot_pts, branch_masks_full = _physical_resample_rgb_and_masks(
        rgb_full, rot_pts, branch_masks_full, axis, spacing
    )
    bbox = _content_bbox_from_rgb(rgb_full, pad = pad)
    rgb = _crop_array_with_bbox(rgb_full, bbox)
    x0, y0, _, _ = bbox
    crop_pts = [(float((x - x0)), float((y - y0))) for x, y in rot_pts]
    branch_masks = {
        name: _crop_array_with_bbox(mask, bbox).astype(bool)
        for name, mask in branch_masks_full.items()
    }
    return rgb, crop_pts, branch_masks, bbox

def _multilabel_projection_rgb_and_points(
    multilabel: np.ndarray,
    label_map: Dict[str, int],
    axis: int,
    points: Optional[List[Tuple[int, int, int]]] = None,
    pad: int = 10,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    rgb, pts, _, _ = _multilabel_projection_bundle(
        multilabel, label_map, axis, points = points, point_branches = None, pad = pad, spacing = spacing
    )
    return rgb, pts

def _stenosis_palette_without_branch_colors(
    label_map: Optional[Dict[str, int]], count: int
) -> List[Tuple[int, int, int]]:
    branch_cols = []
    if label_map:
        branch_cols = [
            np.asarray(branch_display_color(name), dtype = np.float32) for name in label_map
        ]
    candidates = [np.asarray(c, dtype = np.float32) for c in STENOSIS_LABEL_PALETTE]
    selected: List[Tuple[int, int, int]] = []
    for cand in candidates:
        if (branch_cols and (min(float(np.linalg.norm((cand - bc))) for bc in branch_cols) < 70.0)):
            continue
        selected.append(tuple(int(v) for v in cand))
        if (len(selected) >= count):
            return selected
    import colorsys

    i = 0
    while (len(selected) < count):
        h = ((0.07 + (0.61803398875 * i)) % 1.0)
        rgb = (np.asarray(colorsys.hsv_to_rgb(h, 0.88, 0.92), dtype = np.float32) * 255.0)
        i += 1
        if (branch_cols and (min(float(np.linalg.norm((rgb - bc))) for bc in branch_cols) < 70.0)):
            continue
        selected.append(tuple(int(round(v)) for v in rgb))
    return selected

def save_branch_mask_mip_figure(
    case_id: str,
    multilabel: np.ndarray,
    label_map: Dict[str, int],
    out_dir: Path,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Optional[Path]:
    if not MATPLOTLIB_OK:
        return None
    ensure_dir(out_dir)
    views = [("Axial MIP", 2), ("Coronal MIP", 1), ("Sagittal MIP", 0)]
    panels: List[Tuple[str, np.ndarray]] = []
    ratios: List[float] = []
    for title, axis in views:
        rgb, _ = _multilabel_projection_rgb_and_points(
            multilabel, label_map, axis, points = None, pad = 10, spacing = spacing
        )
        panels.append((title, rgb))
        ratios.append(float(np.clip((rgb.shape[1] / max(rgb.shape[0], 1)), 0.85, 1.55)))
    fig, axes = plt.subplots(
        1,
        4,
        figsize = (17.5, 5.6),
        facecolor = PUBLICATION_BG,
        gridspec_kw = {"width_ratios": (ratios + [0.95])},
    )
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(
        f"{case_id} Multi-label mask MIP",
        color = PUBLICATION_FG,
        fontsize = 17,
        fontweight = "bold",
        y = 0.965,
    )
    for ax, (title, rgb) in zip(axes[:3], panels):
        ax.set_facecolor("black")
        ax.imshow(rgb, interpolation = "nearest")
        ax.set_aspect("equal")
        ax.set_title(title, color = PUBLICATION_FG, pad = 6)
        ax.axis("off")
    legend_ax = axes[3]
    legend_ax.axis("off")
    handles = _branch_legend_handles(label_map)
    if handles:
        legend_ax.legend(
            handles = handles,
            loc = "center",
            fontsize = 10.5,
            framealpha = 0.92,
            borderpad = 0.8,
            labelspacing = 0.65,
        )
    fig.subplots_adjust(left = 0.018, right = 0.99, top = 0.89, bottom = 0.05, wspace = 0.06)
    out_path = (Path(out_dir) / "figure_SF_multilabel_mip.png")
    plt.savefig(out_path, dpi = 180, bbox_inches = "tight", facecolor = PUBLICATION_BG)
    plt.close(fig)
    return out_path

def save_stenosis_debug_all_mips(
    case_id: str,
    image: np.ndarray,
    union_mask: np.ndarray,
    paths_global: Dict[str, np.ndarray],
    branch_profiles: Dict[str, Dict[str, Any]],
    out_dir: Path,
    aligned_label: Optional[np.ndarray] = None,
    label_map: Optional[Dict[str, int]] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> List[Path]:
    if not MATPLOTLIB_OK:
        return []
    lesions: List[Dict[str, Any]] = []
    for branch in sorted(branch_profiles, key = branch_sort_key):
        prof = branch_profiles[branch]
        path = np.asarray(
            paths_global.get(branch, np.zeros((0, 3), dtype = np.int32)), dtype = np.int32
        )
        for cand in prof.get("accepted_candidates", []):
            peak_idx = int(cand.get("peak_idx", -1))
            if (0 <= peak_idx < path.shape[0]):
                lesions.append(
                    {
                        "branch": branch,
                        "coord": tuple(int(v) for v in path[peak_idx]),
                        "label": f"{branch} {cand.get('interval_label', _stenosis_interval_short(float(cand.get('report_ratio', 0.0))))}",
                    }
                )
    projections = [
        ("axial_mip", 2, "Axial MIP"),
        ("coronal_mip", 1, "Coronal MIP"),
        ("sagittal_mip", 0, "Sagittal MIP"),
    ]
    outputs: List[Path] = []
    points3d = [x["coord"] for x in lesions]
    branches = [x["branch"] for x in lesions]
    labels = [x["label"] for x in lesions]
    colours = _stenosis_palette_without_branch_colors(label_map, max(len(lesions), 1))

    for suffix, axis, title in projections:
        if ((aligned_label is not None) and (label_map is not None)):
            img, pts, branch_masks, _ = _multilabel_projection_bundle(
                aligned_label,
                label_map,
                axis,
                points = points3d,
                point_branches = branches,
                pad = 12,
                spacing = spacing,
            )
            pts = [
                _snap_point_to_mask(pt, branch_masks.get(branch))
                for pt, branch in zip(pts, branches)
            ]
        else:
            base = np.max(union_mask.astype(np.uint8), axis = axis).astype(bool)
            rgb0 = np.zeros((base.shape + (3,)), dtype = np.uint8)
            rgb0[base] = (255, 255, 255)
            img = np.rot90(rgb0)
            pts = _project_rotated_points(points3d, axis, base.shape)
            bbox = _content_bbox_from_rgb(img, pad = 12)
            img = _crop_array_with_bbox(img, bbox)
            x0, y0, _, _ = bbox
            pts = [((x - x0), (y - y0)) for x, y in pts]
            fg = np.any((img > 0), axis = -1)
            pts = [_snap_point_to_mask(pt, fg) for pt in pts]

        aspect = (float(img.shape[1]) / max(float(img.shape[0]), 1.0))
        fig_w = float(np.clip((7.2 * aspect), 7.5, 12.5))
        fig, ax = plt.subplots(1, 1, figsize = (fig_w, 7.3), facecolor = PUBLICATION_BG)
        ax.set_facecolor("black")
        ax.imshow(img, interpolation = "nearest")
        ax.set_aspect("equal")
        for i, (x, y) in enumerate(pts):
            c = (np.asarray(colours[i], dtype = np.float32) / 255.0)
            ax.plot(x, y, marker = "x", markersize = 16, markeredgewidth = 3.5, color = c)
        if pts:
            _draw_side_labels(ax, pts, labels, colours[: len(pts)], img.shape)
        else:
            ax.text(
                0.5,
                0.04,
                "No accepted stenosis",
                transform = ax.transAxes,
                ha = "center",
                va = "bottom",
                color = "white",
                fontsize = 11,
                bbox = dict(boxstyle = "round,pad=0.25", fc = "black", ec = "white", alpha = 0.75),
            )
        ax.set_title(
            f"{case_id} stenosis debug {title}",
            fontsize = 18,
            fontweight = "bold",
            color = PUBLICATION_FG,
        )
        ax.axis("off")
        fig.subplots_adjust(left = 0.03, right = 0.98, top = 0.90, bottom = 0.04)
        out_path = (Path(out_dir) / f"stenosis_debug_all_branches_{suffix}.png")
        plt.savefig(out_path, dpi = 180, bbox_inches = "tight", facecolor = PUBLICATION_BG)
        plt.close(fig)
        outputs.append(out_path)
    return outputs
