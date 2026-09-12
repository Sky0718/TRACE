from __future__ import annotations

import gc
import os
import re
import tempfile
from collections.abc import Mapping, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

import cta_legacy_backend_v1_0_15 as legacy
import stenosis_v2_core as core

try:
    import nibabel as nib
except Exception as exc:
    nib = None
    NIBABEL_IMPORT_ERROR = exc
else:
    NIBABEL_IMPORT_ERROR = None

try:
    import pydicom
except Exception as exc:
    pydicom = None
    PYDICOM_IMPORT_ERROR = exc
else:
    PYDICOM_IMPORT_ERROR = None

PIPELINE_VERSION = "2.0.10"

PIPELINE_NAME = "CTA Stenosis v2.0.10"

INPUT_COMPATIBILITY_VERSION = "optional_colortable_v1"

def release_large_arrays() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def ensure_dependencies() -> None:
    if (nib is None):
        raise RuntimeError(f"nibabel is required: {NIBABEL_IMPORT_ERROR}")
    if (pydicom is None):
        raise RuntimeError(f"pydicom is required: {PYDICOM_IMPORT_ERROR}")

def clean_path(text: str) -> Path:
    value = str(text).strip().strip('"').strip("'")
    return Path(os.path.expandvars(os.path.expanduser(value)))

def sanitize_case_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return (text or "case")

def case_key_from_name(name: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"\bcase[\s_-]*(\d+)\s*-\s*(\d+)(?!\d)", str(name), flags = re.IGNORECASE)
    return ((int(match.group(1)), int(match.group(2))) if match else None)

def normalized_case_id(key: Tuple[int, int]) -> str:
    return f"case_{int(key[0])}-{int(key[1])}"

def discover_packaged_cases(parent: Path) -> List[Dict[str, Any]]:
    folder = Path(parent)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    zip_map: Dict[Tuple[int, int], Path] = {}
    color_map: Dict[Tuple[int, int], Path] = {}
    label_map: Dict[Tuple[int, int], Path] = {}
    mrb_map: Dict[Tuple[int, int], Path] = {}
    for fp in sorted(folder.iterdir(), key = lambda p: p.name.lower()):
        if not fp.is_file():
            continue
        key = case_key_from_name(fp.name)
        if (key is None):
            continue
        low = fp.name.lower()
        if (
            low.endswith(".zip")
            and ("segmentation" not in low)
            and re.search(r"\bcta(?:[\s_-]*\d+(?:\.\d+)?)?\.zip$", low)
        ):
            zip_map[key] = fp
        elif (low.endswith(".txt") and re.search(r"segmentation[\s_-]*colou?rtable", low)):
            color_map[key] = fp
        elif ((low.endswith(".nii.gz") or low.endswith(".nii")) and re.search(
            r"segmentation[\s_-]*label", low
        )):
            label_map[key] = fp
        elif low.endswith(".mrb"):
            if (key in mrb_map):
                raise ValueError(
                    f"Multiple MRB files found for {normalized_case_id(key)}; keep only the intended scene."
                )
            mrb_map[key] = fp
    keys = sorted((((set(zip_map) | set(color_map)) | set(label_map)) | set(mrb_map)))
    cases: List[Dict[str, Any]] = []
    for key in keys:
        missing = []
        if (key not in zip_map):
            missing.append("CTA ZIP")
        if ((key not in label_map) and (key not in mrb_map)):
            missing.append("Segmentation-label or MRB")
        if missing:
            continue
        cases.append(
            {
                "case_group": int(key[0]),
                "case_number": int(key[1]),
                "case_id": normalized_case_id(key),
                "dicom_zip": zip_map[key],
                "color_table": (color_map.get(key) if (key in label_map) else None),
                "label_nifti": label_map.get(key),
                "label_mrb": (mrb_map.get(key) if (key not in label_map) else None),
            }
        )
    if not cases:
        raise FileNotFoundError(
            "No complete packaged CTA case was found. Each case requires a CTA ZIP "
            "and Segmentation-label.nii.gz or a Slicer MRB; Segmentation_ColorTable.txt is optional."
        )
    return cases

@contextmanager
def materialize_input(input_path: Path) -> Iterator[Path]:
    source = Path(input_path)
    if source.is_dir():
        yield source
        return
    if (not source.is_file() or (source.suffix.lower() != ".zip")):
        raise FileNotFoundError(f"Input must be a folder or ZIP archive: {source}")
    with tempfile.TemporaryDirectory(prefix = "cta_stenosis_v2_input_") as tmp:
        destination = Path(tmp)
        legacy.safe_extract_zip(source, destination)
        children = [p for p in destination.iterdir() if p.is_dir()]
        files = [p for p in destination.iterdir() if p.is_file()]
        if ((len(children) == 1) and not files):
            yield children[0]
        else:
            yield destination

def parse_case_selection(
    text: Optional[str], available: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if ((text is None) or not str(text).strip() or (str(text).strip().lower() in {"all", "*"})):
        return list(available)
    wanted_ids: set[str] = set()
    wanted_numbers: set[int] = set()
    for token in re.split(r"[,;\s]+", str(text).strip().lower()):
        if not token:
            continue
        token = token.replace("case ", "case_")
        key = case_key_from_name(token)
        if (key is not None):
            wanted_ids.add(normalized_case_id(key))
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            lo, hi = sorted((int(range_match.group(1)), int(range_match.group(2))))
            wanted_numbers.update(range(lo, (hi + 1)))
            continue
        if token.isdigit():
            wanted_numbers.add(int(token))
            continue
        raise ValueError(f"Invalid case token: {token}")
    selected = [
        c
        for c in available
        if ((c["case_id"].lower() in wanted_ids) or (int(c["case_number"]) in wanted_numbers))
    ]
    if not selected:
        raise ValueError("No cases matched --cases selection.")
    return selected

def parse_branch_selection(text: Optional[str], available: Sequence[str]) -> List[str]:
    if ((text is None) or not str(text).strip() or (str(text).strip().lower() in {"all", "*"})):
        return list(available)
    wanted = {core.base_branch_name(v) for v in re.split(r"[,;\s]+", str(text).strip()) if v}
    return [b for b in available if (core.base_branch_name(b) in wanted)]

def _slice_positions(headers: Sequence[Any]) -> Optional[np.ndarray]:
    try:
        iop = np.asarray([float(v) for v in headers[0].ImageOrientationPatient], dtype = np.float64)
        normal = np.cross(iop[:3], iop[3:])
        return np.asarray(
            [
                float(np.dot(np.asarray([float(v) for v in ds.ImagePositionPatient]), normal))
                for ds in headers
            ],
            dtype = np.float64,
        )
    except Exception:
        return None

def load_dicom_volume_preallocated(
    folder: Path,
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    series = legacy.discover_dicom_series(Path(folder))
    selected_uid = max(series, key = lambda uid: len(series[uid]))
    files = list(series[selected_uid])
    headers = [pydicom.dcmread(str(fp), stop_before_pixels = True, force = True) for fp in files]
    positions = _slice_positions(headers)
    if ((positions is not None) and (positions.size == len(files))):
        order = np.argsort(positions)
        sorted_positions = positions[order]
    else:
        order = np.argsort([int(getattr(ds, "InstanceNumber", i)) for i, ds in enumerate(headers)])
        sorted_positions = None
    files = [files[int(i)] for i in order]
    headers = [headers[int(i)] for i in order]

    first_ds = pydicom.dcmread(str(files[0]), force = True)
    first_pixels = first_ds.pixel_array
    if (first_pixels.ndim != 2):
        return legacy.load_dicom_volume(Path(folder), preferred_series_uid = selected_uid)
    rows, cols = first_pixels.shape
    volume = np.empty((rows, cols, len(files)), dtype = np.float32)
    for i, fp in enumerate(files):
        ds = (first_ds if (i == 0) else pydicom.dcmread(str(fp), force = True))
        arr = np.asarray(ds.pixel_array, dtype = np.float32)
        if (arr.shape != (rows, cols)):
            raise ValueError(f"DICOM slice shape changed at {fp}: {arr.shape} vs {(rows, cols)}")
        volume[:, :, i] = ((arr * float(getattr(ds, "RescaleSlope", 1.0))) + float(
            getattr(ds, "RescaleIntercept", 0.0)
        ))
        if (i == 0):
            del first_pixels
    first = headers[0]
    try:
        row_spacing = float(first.PixelSpacing[0])
        col_spacing = float(first.PixelSpacing[1])
    except Exception:
        row_spacing = col_spacing = 1.0
    if ((sorted_positions is not None) and (sorted_positions.size >= 2)):
        diffs = np.diff(sorted_positions)
        diffs = diffs[(np.abs(diffs) > 1e-6)]
        slice_spacing = (
            float(np.median(np.abs(diffs)))
            if diffs.size
            else float(getattr(first, "SliceThickness", 1.0))
        )
    else:
        slice_spacing = float(
            getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
        )
    spacing = (row_spacing, col_spacing, abs(slice_spacing))
    meta = {
        "series_uid": str(selected_uid),
        "series_count": int(len(series)),
        "series_slice_counts": {str(k): int(len(v)) for k, v in series.items()},
        "selected_file_count": int(len(files)),
        "volume_shape": [int(v) for v in volume.shape],
        "spacing_mm": [float(v) for v in spacing],
        "first_file": str(files[0]),
        "last_file": str(files[-1]),
        "loader": "preallocated_single_frame_dicom_loader",
        "kvp": (
            float(getattr(first, "KVP", np.nan))
            if (getattr(first, "KVP", None) is not None)
            else np.nan
        ),
        "convolution_kernel": str(getattr(first, "ConvolutionKernel", "")),
        "slice_thickness_mm": (
            float(getattr(first, "SliceThickness", slice_spacing))
            if (getattr(first, "SliceThickness", None) is not None)
            else float(slice_spacing)
        ),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "model_name": str(getattr(first, "ManufacturerModelName", "")),
    }
    return volume, spacing, meta

def load_label_uint16(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    raw = np.asarray(image.dataobj)
    if np.issubdtype(raw.dtype, np.integer):
        maximum = (int(np.max(raw)) if raw.size else 0)
        dtype = (np.uint8 if (maximum <= 255) else np.uint16)
        labels = np.asarray(raw, dtype = dtype)
    else:
        rounded = np.rint(raw)
        maximum = (int(np.max(rounded)) if rounded.size else 0)
        dtype = (np.uint8 if (maximum <= 255) else np.uint16)
        labels = rounded.astype(dtype)
    spacing = tuple(float(abs(v)) for v in image.header.get_zooms()[:3])
    if hasattr(image, "uncache"):
        image.uncache()
    del image, raw
    gc.collect()
    return labels, spacing

class LabelMaskMapping(Mapping[str, np.ndarray]):

    def __init__(self, labels: np.ndarray, label_map: Dict[str, int]):
        self.labels = labels
        self.label_map = dict(label_map)

    def __getitem__(self, key: str) -> np.ndarray:
        return (self.labels == int(self.label_map[key]))

    def __iter__(self) -> Iterator[str]:
        return iter(self.label_map)

    def __len__(self) -> int:
        return len(self.label_map)

def crop_mask(mask: np.ndarray, pad: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(np.asarray(mask).astype(bool))
    if (coords.shape[0] == 0):
        return np.zeros((0, 0, 0), dtype = bool), np.zeros(3, dtype = np.int32)
    lo = np.maximum((coords.min(axis = 0) - pad), 0)
    hi = np.minimum(((coords.max(axis = 0) + pad) + 1), np.asarray(mask.shape))
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    return np.ascontiguousarray(mask[sl].astype(bool)), lo.astype(np.int32)

def mask_geometry(
    branch_mask: np.ndarray, centerline: core.CenterlineResult, spacing: Sequence[float]
) -> Dict[str, Any]:
    crop, origin = crop_mask(branch_mask, 2)
    voxel_count = int(np.sum(crop))
    voxel_volume = float(np.prod(np.asarray(spacing, dtype = np.float64)))
    if ((crop.size == 0) or not np.any(crop)):
        return {
            "mask_voxel_count": 0,
            "mask_volume_mm3": 0.0,
            "mask_surface_area_mm2": 0.0,
            "mask_diameter_mean_mm": 0.0,
            "mask_diameter_median_mm": 0.0,
            "mask_diameter_min_mm": 0.0,
            "mask_diameter_max_mm": 0.0,
        }
    surface = float(legacy.voxel_surface_area_mm2(crop, tuple(float(v) for v in spacing)))
    radius = distance_transform_edt(crop, sampling = tuple(float(v) for v in spacing)).astype(
        np.float32
    )
    path_local = (centerline.path_voxel - origin.astype(np.float32)[None, :])
    path_int = np.rint(path_local).astype(np.int32)
    valid = np.all(((path_int >= 0) & (path_int < np.asarray(crop.shape)[None, :])), axis = 1)
    samples = (
        (radius[path_int[valid, 0], path_int[valid, 1], path_int[valid, 2]] * 2.0)
        if np.any(valid)
        else np.zeros(0)
    )
    return {
        "mask_voxel_count": voxel_count,
        "mask_volume_mm3": float((voxel_count * voxel_volume)),
        "mask_surface_area_mm2": surface,
        "mask_surface_to_volume_ratio_per_mm": float(
            (surface / max((voxel_count * voxel_volume), 1e-6))
        ),
        "mask_diameter_mean_mm": (float(np.mean(samples)) if samples.size else 0.0),
        "mask_diameter_median_mm": (float(np.median(samples)) if samples.size else 0.0),
        "mask_diameter_min_mm": (float(np.min(samples)) if samples.size else 0.0),
        "mask_diameter_max_mm": (float(np.max(samples)) if samples.size else 0.0),
    }

def infer_branch_labels_without_colortable(
    aligned_label: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    if (aligned_label.ndim != 3):
        raise ValueError("A three-dimensional segmentation label is required.")
    positive_values = [int(v) for v in np.unique(aligned_label) if (v > 0)]
    if not positive_values:
        raise ValueError("No positive labels were found in the segmentation.")
    vessel_union = (aligned_label > 0)
    inferred_masks, inference_info = legacy.separate_branches(vessel_union)
    names = sorted(inferred_masks, key = legacy.branch_sort_key)
    synthetic = np.zeros(aligned_label.shape, dtype = (np.uint8 if (len(names) < 256) else np.uint16))
    branch_specs: List[Dict[str, Any]] = []
    for branch in names:
        mask = np.asarray(inferred_masks[branch], dtype = bool)
        if (mask.shape != aligned_label.shape):
            raise ValueError(f"Inferred mask shape mismatch: {branch}")
        mask = ((mask & vessel_union) & (synthetic == 0))
        if not np.any(mask):
            continue
        value = (len(branch_specs) + 1)
        synthetic[mask] = value
        rgb = legacy.BRANCH_RGB.get(branch, (149, 165, 166))
        branch_specs.append(
            {
                "branch": str(branch),
                "label_value": value,
                "original_name": f"INFERRED_{branch}",
                "rgba": (tuple(int(v) for v in rgb) + (255,)),
            }
        )
    if not branch_specs:
        raise ValueError("No branch navigation regions could be inferred without ColorTable.")
    unassigned = int(np.count_nonzero((vessel_union & (synthetic == 0))))
    provenance = {
        "input_compatibility_version": INPUT_COMPATIBILITY_VERSION,
        "label_mapping_source": "inferred_without_colortable",
        "source_positive_label_values": positive_values,
        "inferred_branches": [spec["branch"] for spec in branch_specs],
        "unassigned_positive_voxel_count": unassigned,
        "branch_inference": inference_info,
        "label_mapping_warning": (
            "ColorTable absent: positive labels were combined and navigation branches "
            "were inferred by the existing legacy geometry separator. Source branch names "
            "and POINT markers cannot be recovered; inferred anatomy requires review. "
            "This is not equivalent to source-named multi-label input."
        ),
    }
    return synthetic, branch_specs, provenance
