from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .models import (
    CenterlineResult,
    LesionResult,
    PlanePatch,
)

def sample_cpr(
    image: np.ndarray,
    centerline: CenterlineResult,
    spacing: Sequence[float],
    direction: str,
    half_width_mm: float = 7.0,
    pixel_mm: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray]:
    if not centerline.ok:
        return np.zeros((0, 0), dtype = np.float32), np.zeros(0, dtype = np.float32)
    offsets = np.arange(
        -half_width_mm, (half_width_mm + (0.5 * pixel_mm)), pixel_mm, dtype = np.float32
    )
    normal = (centerline.normal_u if (direction.lower() == "u") else centerline.normal_v)
    phys = (centerline.path_physical_mm[None, :, :] + (offsets[:, None, None] * normal[None, :, :]))
    vox = (phys / np.asarray(spacing)[None, None, :])
    cpr = map_coordinates(
        np.asarray(image, dtype = np.float32),
        [vox[..., 0], vox[..., 1], vox[..., 2]],
        order = 1,
        mode = "constant",
        cval = -1000.0,
    )
    return np.asarray(cpr, dtype = np.float32), offsets

def save_cpr_figure(
    case_id: str,
    branch: str,
    image: np.ndarray,
    centerline: CenterlineResult,
    profile: pd.DataFrame,
    lesions: Sequence[LesionResult],
    spacing: Sequence[float],
    output_path: Path,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    u, offsets = sample_cpr(image, centerline, spacing, "u")
    v, _ = sample_cpr(image, centerline, spacing, "v")
    if (u.size == 0):
        return None
    accepted = [r for r in lesions if (int(r.accepted) == 1)]
    not_assessable = [r for r in lesions if (r.assessment_status != "assessable")]
    total = float(centerline.cumulative_mm[-1])
    extent = [0.0, total, float(offsets[0]), float(offsets[-1])]
    fig = plt.figure(figsize = (16, 8.5), facecolor = "white")
    gs = fig.add_gridspec(3, 1, height_ratios = [0.55, 0.55, 1.2], hspace = 0.34)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax, cpr, title in (
        (axes[0], u, "Full-vessel CPR plane U"),
        (axes[1], v, "Full-vessel CPR plane V"),
    ):
        ax.imshow(
            cpr, cmap = "gray", origin = "lower", aspect = "auto", extent = extent, vmin = -100, vmax = 800
        )
        ax.set_box_aspect(min(0.22, max(0.10, ((2.0 * abs(offsets[-1])) / max(total, 1.0)))))
        for lesion in accepted:
            ax.axvspan(lesion.start_distance_mm, lesion.end_distance_mm, alpha = 0.18)
            ax.axvline(lesion.peak_distance_mm, linewidth = 1.4)
        for lesion in not_assessable:
            ax.axvspan(lesion.start_distance_mm, lesion.end_distance_mm, alpha = 0.10, color = "gray")
        ax.set_ylabel("Offset (mm)")
        ax.set_title(title)
    dist = pd.to_numeric(profile["distance_mm"], errors = "coerce")
    diam = pd.to_numeric(profile["min_diameter_mm"], errors = "coerce")
    ref_d = pd.to_numeric(
        profile.get("reference_diameter_mm", pd.Series(np.nan, index = profile.index)),
        errors = "coerce",
    )
    valid = (
        pd.to_numeric(profile.get("valid", pd.Series(0, index = profile.index)), errors = "coerce")
        .fillna(0)
        .astype(int)
    )
    axes[2].plot(dist, diam, label = "Tracked CTA minimum diameter", linewidth = 1.3)
    axes[2].plot(dist, ref_d, label = "Local reference diameter", linewidth = 1.2)
    axes[2].scatter(
        dist[(valid == 0)],
        np.zeros(int(np.sum((valid == 0)))),
        s = 9,
        marker = "x",
        color = "gray",
        label = "Invalid/not assessable section",
    )
    for lesion in accepted:
        axes[2].axvspan(lesion.start_distance_mm, lesion.end_distance_mm, alpha = 0.18)
        axes[2].scatter([lesion.peak_distance_mm], [lesion.minimum_lumen_diameter_mm], s = 34)
    axes[2].set_xlabel("Root-to-tip distance (mm)")
    axes[2].set_ylabel("Diameter (mm)")
    axes[2].grid(alpha = 0.25)
    axes[2].legend(loc = "best")
    fig.suptitle(
        f"{case_id} {branch} longitudinally tracked CTA lumen CPR", fontsize = 16, fontweight = "bold"
    )
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(output_path, dpi = 180, bbox_inches = "tight", facecolor = "white")
    plt.close(fig)
    return output_path

def save_lesion_cross_section_figure(
    case_id: str,
    branch: str,
    lesion: LesionResult,
    patches: Mapping[int, PlanePatch],
    output_path: Path,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    indices = [lesion.start_index, lesion.peak_index, lesion.end_index]
    if not all((i in patches) for i in indices):
        return None
    fig, axes = plt.subplots(1, 3, figsize = (13, 4.5), facecolor = "white")
    for ax, idx, title in zip(axes, indices, ("Lesion start", "Minimum lumen", "Lesion end")):
        patch = patches[idx]
        axis = patch.axis_mm
        extent = [float(axis[0]), float(axis[-1]), float(axis[0]), float(axis[-1])]
        ax.imshow(patch.image, cmap = "gray", origin = "lower", extent = extent, vmin = -100, vmax = 800)
        if ((patch.contour_xy_mm.shape[0] >= 3) and (int(patch.measurement.valid) == 1)):
            contour = np.vstack([patch.contour_xy_mm, patch.contour_xy_mm[0]])
            ax.plot(contour[:, 0], contour[:, 1], linewidth = 1.6)
        ax.axhline(0, color = "white", alpha = 0.3, linewidth = 0.6)
        ax.axvline(0, color = "white", alpha = 0.3, linewidth = 0.6)
        status = ("valid" if (int(patch.measurement.valid) == 1) else "not assessable")
        ax.set_title(
            (
                f"{title}\n{status}; Dmin={patch.measurement.min_diameter_mm:.2f} mm"
                if np.isfinite(patch.measurement.min_diameter_mm)
                else f"{title}\n{status}"
            )
        )
        ax.set_xlabel("U (mm)")
        ax.set_ylabel("V (mm)")
    fig.suptitle(
        f"{case_id} {branch} {lesion.segment_span} lesion | {lesion.stenosis_interval} | {lesion.assessment_status}",
        fontsize = 14,
        fontweight = "bold",
    )
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.tight_layout(rect = [0, 0, 1, 0.90])
    plt.savefig(output_path, dpi = 180, bbox_inches = "tight", facecolor = "white")
    plt.close(fig)
    return output_path

def save_section_stack_npz(
    output_path: Path,
    patches: Mapping[int, PlanePatch],
    centerline: CenterlineResult,
) -> Optional[Path]:
    if not patches:
        return None
    indices = sorted(patches)
    images = np.stack([patches[i].image for i in indices]).astype(np.float16)
    masks = np.stack([patches[i].lumen_component for i in indices]).astype(np.uint8)
    valid = np.asarray([patches[i].measurement.valid for i in indices], dtype = np.uint8)
    axis = patches[indices[0]].axis_mm.astype(np.float32)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(
        output_path,
        indices = np.asarray(indices, dtype = np.int32),
        cta_sections = images,
        lumen_masks = masks,
        valid = valid,
        axis_mm = axis,
        distance_mm = centerline.cumulative_mm[np.asarray(indices, dtype = int)].astype(np.float32),
        path_voxel = centerline.path_voxel[np.asarray(indices, dtype = int)].astype(np.float32),
    )
    return output_path
