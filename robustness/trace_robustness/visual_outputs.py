from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from typing import Tuple
import math
import numpy as np
import pandas as pd
from . import (
    runtime,
    models,
    measurements,
    quality_tables,
    reliability,
    storage,
    visual_layers,
)

def save_heatmap_slices_figure(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    z_list = visual_layers.select_slice_indices(
        calc_mask if np.any(calc_mask) else vessel_mask, count = 8
    )
    cols = 4
    rows = int(math.ceil((len(z_list) / cols)))
    (fig, axes) = runtime.plt.subplots(
        rows, cols, figsize = (18, (4.8 * rows)), facecolor = runtime.visual_fig_bg()
    )
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(
        f"{case_id} CTA calcium heatmap slices",
        color = runtime.visual_text_color(),
        fontsize = 18,
        fontweight = "bold",
    )
    for ax, z in zip(axes, z_list):
        ax.set_facecolor(runtime.visual_image_axis_bg())
        ax.imshow(image[:, :, int(z)].T, cmap = "gray", origin = "lower", vmin = -150, vmax = 500)
        if np.any(vessel_mask[:, :, int(z)]):
            ax.contour(
                vessel_mask[:, :, int(z)].T.astype(np.uint8),
                levels = [0.5],
                colors = ["cyan"],
                linewidths = 0.7,
            )
        if np.any(calc_mask[:, :, int(z)]):
            ax.contour(
                calc_mask[:, :, int(z)].T.astype(np.uint8),
                levels = [0.5],
                colors = ["orange"],
                linewidths = 1.0,
            )
        ax.set_title(f"Z={int(z)}", color = runtime.visual_text_color())
        ax.axis("off")
    for ax in axes[len(z_list) :]:
        ax.axis("off")
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.95])
    out_path = (out_dir / "figure_A_heatmap_slices.png")
    runtime.plt.savefig(
        out_path, dpi = 170, bbox_inches = "tight", facecolor = runtime.visual_fig_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_standard_visual_outputs(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Dict[str, str]:
    visual_dir = storage.ensure_dir((out_dir / "visuals"))
    video_dir = storage.ensure_dir((out_dir / "videos"))
    outputs: Dict[str, str] = {}
    p = save_case_qa_figure(case_id, image, vessel_mask, calc_mask, branch_rows, visual_dir)
    if p:
        outputs["overview"] = str(p)
    p = save_heatmap_slices_figure(case_id, image, vessel_mask, calc_mask, visual_dir)
    if p:
        outputs["heatmap_slices"] = str(p)
    p = save_mip_projection_figure(case_id, image, vessel_mask, calc_mask, visual_dir)
    if p:
        outputs["mip_projection"] = str(p)
    p = save_branch_panels_figure(
        case_id, image, calc_mask, branches, branch_rows, spacing, visual_dir
    )
    if p:
        outputs["branch_concepts"] = str(p)
    p = save_centerline_profiles_figure(case_id, branches, calc_mask, spacing, visual_dir)
    if p:
        outputs["centerline_profiles"] = str(p)
    p = save_stenosis_marked_figure(
        case_id, image, vessel_mask, calc_mask, branches, branch_rows, spacing, visual_dir
    )
    if p:
        outputs["stenosis_marked_figure"] = str(p)
    p = visual_layers.save_axial_overlay_video(
        case_id, image, vessel_mask, calc_mask, video_dir
    )
    if p:
        outputs["axial_video"] = str(p)
    p = visual_layers.save_branch_overlay_video(
        case_id, image, branches, calc_mask, video_dir
    )
    if p:
        outputs["branch_video"] = str(p)
    p = visual_layers.save_stenosis_marked_video(
        case_id, image, vessel_mask, calc_mask, branches, branch_rows, spacing, video_dir
    )
    if p:
        outputs["stenosis_marked_video"] = str(p)
    storage.write_json((out_dir / "visual_outputs.json"), outputs)
    return outputs

def _top_narrowing_branch(
    branches: Dict[str, np.ndarray], branch_rows: List[models.BranchConcept]
) -> Optional[Tuple[str, np.ndarray, models.BranchConcept]]:
    if not branch_rows:
        return None
    rows = sorted(
        branch_rows, key = lambda r: float(getattr(r, "stenosis_ratio", 0.0)), reverse = True
    )
    for r in rows:
        if ((str(r.branch) in branches) and (float(r.stenosis_ratio) > 0)):
            return (str(r.branch), branches[str(r.branch)], r)
    for r in rows:
        if (str(r.branch) in branches):
            return (str(r.branch), branches[str(r.branch)], r)
    return None

def save_paper_narrowing_evidence(
    case_id: str,
    branches: Dict[str, np.ndarray],
    calc_mask: np.ndarray,
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if (not runtime.MATPLOTLIB_OK or not branches):
        return None
    top = _top_narrowing_branch(branches, branch_rows)
    if (top is None):
        return None
    (branch, bmask, brow) = top
    prof = measurements.centerline_profile_arrays(
        bmask, spacing, calcium_mask = (bmask & calc_mask)
    )
    if not prof.get("ok"):
        return None
    cum = np.asarray(prof["cum"], dtype = np.float32)
    ratio = np.asarray(prof["ratio"], dtype = np.float32)
    if (cum.size == 0):
        return None
    idx = int(np.argmax(ratio)) if ratio.size else 0
    (fig, axes) = runtime.plt.subplots(
        3, 1, figsize = (13, 10), facecolor = runtime.visual_paper_bg(), sharex = True
    )
    fig.suptitle(
        f"{case_id} | Multi-evidence narrowing profile ({branch})",
        color = runtime.visual_text_color(),
        fontsize = 17,
        fontweight = "bold",
    )
    for ax in axes:
        ax.set_facecolor(runtime.visual_axis_bg())
        ax.tick_params(colors = runtime.visual_text_color())
        for spine in ax.spines.values():
            spine.set_color(runtime.visual_text_color())
        ax.grid(alpha = 0.25)
    axes[0].plot(cum, prof["diameter"], label = "observed diameter")
    axes[0].plot(cum, prof["reference"], label = "reference diameter", linestyle = "--")
    axes[0].axvline(float(cum[idx]), linestyle = ":", linewidth = 1.5)
    axes[0].set_ylabel("Diameter (mm)", color = runtime.visual_text_color())
    axes[0].legend()
    axes[1].plot(cum, ratio, label = "geometric narrowing index")
    if (prof.get("ratio_area") is not None):
        axes[1].plot(cum, prof["ratio_area"], label = "area evidence", alpha = 0.8)
    axes[1].axvline(float(cum[idx]), linestyle = ":", linewidth = 1.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score", color = runtime.visual_text_color())
    axes[1].legend()
    if (prof.get("calcium_penalty") is not None):
        axes[2].plot(cum, prof["calcium_penalty"], label = "peripheral high-HU proximity")
    axes[2].axvline(float(cum[idx]), linestyle = ":", linewidth = 1.5)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("High-HU evidence", color = runtime.visual_text_color())
    axes[2].set_xlabel("Centerline distance (mm)", color = runtime.visual_text_color())
    axes[2].legend()
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.94])
    out_path = (out_dir / "figure_PAPER_narrowing_evidence.png")
    runtime.plt.savefig(
        out_path, dpi = 190, bbox_inches = "tight", facecolor = runtime.visual_paper_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_mip_projection_figure(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    projections = [("Axial MIP", 2), ("Coronal MIP", 1), ("Sagittal MIP", 0)]
    (fig, axes) = runtime.plt.subplots(
        1, 3, figsize = (18, 6), facecolor = runtime.visual_fig_bg()
    )
    fig.suptitle(
        f"{case_id} CTA vessel / high-HU candidate MIP",
        color = runtime.visual_text_color(),
        fontsize = 17,
        fontweight = "bold",
    )
    for ax, (title, axis) in zip(axes, projections):
        ax.imshow(
            visual_layers._full_vessel_projection_rgb(
                image, vessel_mask, calc_mask, axis = axis
            )
        )
        ax.set_title(title, color = runtime.visual_text_color())
        ax.axis("off")
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.92])
    out_path = (out_dir / "figure_A_mip_projection.png")
    runtime.plt.savefig(
        out_path, dpi = 190, bbox_inches = "tight", facecolor = runtime.visual_fig_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_paper_orthogonal_full_tree_map(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    projections = [("Axial", 2), ("Coronal", 1), ("Sagittal", 0)]
    (fig, axes) = runtime.plt.subplots(
        1, 3, figsize = (20, 7), facecolor = runtime.visual_paper_bg()
    )
    fig.suptitle(
        f"{case_id} | Full coronary-tree map from segmentation mask",
        color = runtime.visual_text_color(),
        fontsize = 18,
        fontweight = "bold",
    )
    for ax, (title, axis) in zip(axes, projections):
        ax.imshow(
            visual_layers._full_vessel_projection_rgb(
                image, vessel_mask, calc_mask, axis = axis
            )
        )
        ax.set_title(
            f"{title} full-vessel MIP", color = runtime.visual_text_color(), fontsize = 13
        )
        ax.axis("off")
    try:
        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(
                color = (np.asarray(runtime.VESSEL_OVERLAY_RGB) / 255.0),
                label = "Coronary vessel mask",
            ),
            mpatches.Patch(
                color = (np.asarray(runtime.CALCIUM_OVERLAY_RGB) / 255.0),
                label = "Peripheral high-HU candidate",
            ),
            mpatches.Patch(
                color = (np.asarray(runtime.VESSEL_EDGE_RGB) / 255.0), label = "Mask edge"
            ),
        ]
        fig.legend(
            handles = handles,
            loc = "lower center",
            ncol = 3,
            frameon = False,
            labelcolor = runtime.visual_text_color(),
        )
    except Exception:
        pass
    runtime.plt.tight_layout(rect = [0, 0.06, 1, 0.93])
    out_path = (out_dir / "figure_PAPER_orthogonal_branch_map.png")
    runtime.plt.savefig(
        out_path, dpi = 200, bbox_inches = "tight", facecolor = runtime.visual_paper_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_paper_multipanel_summary(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    (fig, axes) = runtime.plt.subplots(
        2, 3, figsize = (20, 12), facecolor = runtime.visual_paper_bg()
    )
    axes = axes.reshape(2, 3)
    fig.suptitle(
        f"{case_id} | CoroConcept full-tree concept dashboard",
        color = runtime.visual_text_color(),
        fontsize = 19,
        fontweight = "bold",
    )
    raw_mip = np.rot90(np.max(visual_layers.intensity_to_uint8(image), axis = 2).T)
    axes[0, 0].imshow(raw_mip, cmap = "gray")
    axes[0, 0].set_title("Raw axial MIP", color = runtime.visual_text_color())
    axes[0, 0].axis("off")
    axes[0, 1].imshow(
        visual_layers._full_vessel_projection_rgb(image, vessel_mask, calc_mask, axis = 2)
    )
    axes[0, 1].set_title(
        "Full vessel mask, exact projected thickness", color = runtime.visual_text_color()
    )
    axes[0, 1].axis("off")
    top = _top_narrowing_branch(branches, branch_rows)
    if (top is not None):
        (branch, bmask, brow) = top
        prof = measurements.centerline_profile_arrays(
            bmask, spacing, calcium_mask = (bmask & calc_mask)
        )
        if prof.get("ok"):
            cum = prof["cum"]
            axes[0, 2].plot(cum, prof["diameter"], label = "diameter")
            axes[0, 2].plot(cum, prof["reference"], label = "reference", linestyle = "--")
            axes[0, 2].set_title(
                f"Diameter profile: {branch}", color = runtime.visual_text_color()
            )
            axes[0, 2].set_xlabel("Distance (mm)", color = runtime.visual_text_color())
            axes[0, 2].set_ylabel("Diameter (mm)", color = runtime.visual_text_color())
            axes[0, 2].legend()
    else:
        axes[0, 2].text(
            0.5,
            0.5,
            "No valid branch profile",
            color = runtime.visual_text_color(),
            ha = "center",
            va = "center",
        )
    axes[0, 2].set_facecolor(runtime.visual_axis_bg())
    axes[0, 2].tick_params(colors = runtime.visual_text_color())
    for spine in axes[0, 2].spines.values():
        spine.set_color(runtime.visual_text_color())
    axes[0, 2].grid(alpha = 0.25)
    names = [r.branch for r in branch_rows]
    lengths = [float(r.vessel_length_mm) for r in branch_rows]
    axes[1, 0].bar(names, lengths)
    axes[1, 0].set_title("Branch length", color = runtime.visual_text_color())
    axes[1, 0].set_ylabel("mm", color = runtime.visual_text_color())
    torts = [float(r.tortuosity) for r in branch_rows]
    axes[1, 1].bar(names, torts)
    axes[1, 1].axhline(runtime.TORTUOSITY_OUTLIER_THRESHOLD, linestyle = "--", linewidth = 1)
    axes[1, 1].set_title(
        "Tortuosity with review threshold", color = runtime.visual_text_color()
    )
    axes[1, 1].set_ylabel("ratio", color = runtime.visual_text_color())
    nar = [float(r.stenosis_ratio) for r in branch_rows]
    axes[1, 2].bar(names, nar)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Geometric narrowing index", color = runtime.visual_text_color())
    axes[1, 2].set_ylabel("0-1", color = runtime.visual_text_color())
    for ax in axes[1, :]:
        ax.set_facecolor(runtime.visual_axis_bg())
        ax.tick_params(colors = runtime.visual_text_color())
        for spine in ax.spines.values():
            spine.set_color(runtime.visual_text_color())
        ax.grid(alpha = 0.22)
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.94])
    out_path = (out_dir / "figure_PAPER_multipanel_summary.png")
    runtime.plt.savefig(
        out_path, dpi = 200, bbox_inches = "tight", facecolor = runtime.visual_paper_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_enhanced_visual_outputs(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Dict[str, str]:
    outputs = save_standard_visual_outputs(
        case_id, image, vessel_mask, calc_mask, branches, branch_rows, spacing, out_dir
    )
    visual_dir = storage.ensure_dir((out_dir / "visuals"))
    p = save_paper_multipanel_summary(
        case_id, image, vessel_mask, calc_mask, branches, branch_rows, spacing, visual_dir
    )
    if p:
        outputs["paper_multipanel_summary"] = str(p)
    p = save_paper_orthogonal_full_tree_map(
        case_id, image, vessel_mask, calc_mask, visual_dir
    )
    if p:
        outputs["paper_orthogonal_branch_map"] = str(p)
    p = save_paper_narrowing_evidence(
        case_id, branches, calc_mask, branch_rows, spacing, visual_dir
    )
    if p:
        outputs["paper_narrowing_evidence"] = str(p)
    storage.write_json((out_dir / "visual_outputs.json"), outputs)
    return outputs

def _merge_reliability_into_branch_csv(
    case_id: str,
    branch_rows: List[models.BranchConcept],
    reliability_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    branch_path = (Path(out_dir) / "cta_concepts_branch_level.csv")
    if ((not branch_path.exists() or (reliability_df is None)) or reliability_df.empty):
        return
    try:
        branch_df = pd.read_csv(branch_path)
        if ("branch" not in branch_df.columns):
            return
        rel_cols = [c for c in reliability_df.columns if (c not in ("case_id",))]
        merged = branch_df.merge(
            reliability_df[(["branch"] + [c for c in rel_cols if (c != "branch")])],
            on = "branch",
            how = "left",
        )
        merged.to_csv(branch_path, index = False, encoding = "utf-8-sig")
    except Exception:
        pass

def save_case_qa_figure(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branch_rows: List[models.BranchConcept],
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    active_z = np.where(vessel_mask.any(axis = (0, 1)))[0]
    z = int(active_z[(len(active_z) // 2)]) if active_z.size else (image.shape[2] // 2)
    (fig, axes) = runtime.plt.subplots(1, 3, figsize = (15, 5))
    axes[0].imshow(image[:, :, z].T, cmap = "gray", origin = "lower", vmin = -150, vmax = 500)
    axes[0].contour(
        vessel_mask[:, :, z].T.astype(np.uint8), levels = [0.5], colors = "cyan", linewidths = 0.8
    )
    axes[0].set_title(f"{case_id} vessel ROI Z={z}")
    axes[0].axis("off")
    axes[1].imshow(image[:, :, z].T, cmap = "gray", origin = "lower", vmin = -150, vmax = 500)
    if np.any(calc_mask[:, :, z]):
        axes[1].contour(
            calc_mask[:, :, z].T.astype(np.uint8),
            levels = [0.5],
            colors = "orange",
            linewidths = 1.0,
        )
    axes[1].set_title("Peripheral high-HU candidates")
    axes[1].axis("off")
    names = [r.branch for r in branch_rows]
    ratios = [r.stenosis_ratio for r in branch_rows]
    axes[2].bar(names, ratios)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Geometric narrowing index")
    axes[2].set_title("Branch max geometric narrowing estimate")
    runtime.plt.tight_layout()
    out_path = (out_dir / "figure_A_concepts_overview.png")
    runtime.plt.savefig(out_path, dpi = 160, bbox_inches = "tight")
    runtime.plt.close(fig)
    return out_path

def save_branch_panels_figure(
    case_id: str,
    image: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    if not branch_rows:
        return None
    rel_path = (
        (Path(out_dir).parent / "cta_concepts_narrowing_reliability.csv")
        if (Path(out_dir).name == "visuals")
        else (Path(out_dir) / "cta_concepts_narrowing_reliability.csv")
    )
    rel_df = storage.read_csv_safe(rel_path)
    rel_lookup = {}
    if (not rel_df.empty and ("branch" in rel_df.columns)):
        rel_lookup = {str(r["branch"]): r for (_, r) in rel_df.iterrows()}
    rows = len(branch_rows)
    (fig, axes) = runtime.plt.subplots(
        rows, 3, figsize = (18, max(4.0, (4.2 * rows))), facecolor = runtime.visual_fig_bg()
    )
    axes = np.asarray(axes)
    if (rows == 1):
        axes = axes.reshape(1, 3)
    fig.suptitle(
        f"{case_id} branch-level CTA concept QA",
        color = runtime.visual_text_color(),
        fontsize = 18,
        fontweight = "bold",
    )
    for i, row in enumerate(branch_rows):
        bmask = branches.get(row.branch)
        if (bmask is None):
            continue
        per_z = np.sum(bmask, axis = (0, 1))
        z = int(np.argmax(per_z)) if (np.max(per_z) > 0) else (image.shape[2] // 2)
        rgb = visual_layers.gray_to_rgb(visual_layers.intensity_to_uint8(image[:, :, z].T))
        branch_slice = bmask[:, :, z].T.astype(bool)
        calc_slice = (bmask & calc_mask)[:, :, z].T.astype(bool)
        rgb = visual_layers.overlay_rgb(
            rgb, branch_slice, visual_layers.branch_color(row.branch), 0.42
        )
        rgb = visual_layers.overlay_rgb(rgb, calc_slice, runtime.CALCIUM_OVERLAY_RGB, 0.9)
        axes[i, 0].imshow(rgb, origin = "lower")
        axes[i, 0].set_title(
            f"{row.branch} representative Z={z}", color = runtime.visual_text_color()
        )
        axes[i, 0].axis("off")
        prof = measurements.centerline_profile_arrays(
            bmask, spacing, calcium_mask = (bmask & calc_mask)
        )
        if prof.get("ok"):
            x = np.arange(len(prof["ratio"]))
            axes[i, 1].plot(x, prof["ratio"], label = "geometric narrowing")
            axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title(
            f"{row.branch} narrowing profile", color = runtime.visual_text_color()
        )
        axes[i, 1].tick_params(colors = runtime.visual_text_color())
        for spine in axes[i, 1].spines.values():
            spine.set_color(runtime.visual_text_color())
        axes[i, 1].grid(alpha = 0.25)
        axes[i, 2].axis("off")
        rel = rel_lookup.get(str(row.branch), {})
        conf = (
            str(rel.get("candidate_confidence_category", "not_available"))
            if isinstance(rel, pd.Series)
            else str(rel.get("candidate_confidence_category", "not_available"))
        )
        score = (
            storage.to_float(rel.get("candidate_reliability_score", np.nan), np.nan)
            if isinstance(rel, (pd.Series, dict))
            else np.nan
        )
        notes = (
            quality_tables.clean_warning_value(rel.get("reliability_notes", ""))
            if isinstance(rel, (pd.Series, dict))
            else ""
        )
        lines = [
            f"Branch: {row.branch}",
            f"Length: {row.vessel_length_mm:.2f} mm",
            f"Mean diameter: {row.vessel_diameter_mean_mm:.2f} mm",
            f"Narrowing: {reliability.stenosis_interval_label(row.stenosis_ratio)}",
            f"Position: {row.stenosis_position_norm:.3f}",
            (f"Reliability: {conf}" + (f" ({score:.0f}/100)" if np.isfinite(score) else "")),
            f"Tortuosity: {row.tortuosity:.3f}",
            f"High-HU candidate volume: {row.calcium_volume_mm3:.2f} mm³",
            f"High-HU burden: {row.calcium_burden_pct:.2f}%",
        ]
        if (notes and (notes != "no_geometric_narrowing_candidate")):
            lines.append(f"Notes: {notes}")
        axes[i, 2].text(
            0.02,
            0.95,
            "\n".join(lines),
            ha = "left",
            va = "top",
            color = runtime.visual_text_color(),
            fontsize = 12,
            transform = axes[i, 2].transAxes,
        )
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.96])
    out_path = (out_dir / "figure_SF_branch_concepts.png")
    runtime.plt.savefig(
        out_path, dpi = 170, bbox_inches = "tight", facecolor = runtime.visual_fig_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_centerline_profiles_figure(
    case_id: str,
    branches: Dict[str, np.ndarray],
    calc_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if (not runtime.MATPLOTLIB_OK or not branches):
        return None
    rows = len(branches)
    (fig, axes) = runtime.plt.subplots(
        rows, 2, figsize = (16, max(4.0, (3.6 * rows))), facecolor = runtime.visual_fig_bg()
    )
    axes = np.asarray(axes)
    if (rows == 1):
        axes = axes.reshape(1, 2)
    fig.suptitle(
        f"{case_id} centerline diameter and geometric narrowing profiles",
        color = runtime.visual_text_color(),
        fontsize = 18,
        fontweight = "bold",
    )
    for i, (branch, bmask) in enumerate(branches.items()):
        prof = measurements.centerline_profile_arrays(
            bmask, spacing, calcium_mask = (bmask & calc_mask)
        )
        for ax in axes[i]:
            ax.set_facecolor(runtime.visual_axis_bg())
            ax.tick_params(colors = runtime.visual_text_color())
            for spine in ax.spines.values():
                spine.set_color(runtime.visual_text_color())
            ax.grid(alpha = 0.25)
        if prof.get("ok"):
            cum = prof["cum"]
            axes[i, 0].plot(cum, prof["diameter"], label = "diameter")
            axes[i, 0].plot(cum, prof["reference"], label = "reference")
            axes[i, 0].legend()
            axes[i, 1].plot(cum, prof["ratio"], label = "geometric narrowing index")
            axes[i, 1].set_ylim(0, 1)
        axes[i, 0].set_title(
            f"{branch} diameter profile", color = runtime.visual_text_color()
        )
        axes[i, 0].set_xlabel("Centerline distance (mm)", color = runtime.visual_text_color())
        axes[i, 0].set_ylabel("Diameter (mm)", color = runtime.visual_text_color())
        axes[i, 1].set_title(
            f"{branch} geometric narrowing estimate", color = runtime.visual_text_color()
        )
        axes[i, 1].set_xlabel("Centerline distance (mm)", color = runtime.visual_text_color())
        axes[i, 1].set_ylabel("Index", color = runtime.visual_text_color())
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.96])
    out_path = (out_dir / "figure_SF_centerline_profiles.png")
    runtime.plt.savefig(
        out_path, dpi = 170, bbox_inches = "tight", facecolor = runtime.visual_fig_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_stenosis_marked_figure(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.MATPLOTLIB_OK:
        return None
    points = visual_layers.estimate_stenosis_points(
        branches, branch_rows, spacing, calc_mask = calc_mask
    )
    if not points:
        return None
    best = points[0]
    (bx, by, bz) = best["point"]
    z = int(np.clip(bz, 0, (image.shape[2] - 1)))
    base = visual_layers.gray_to_rgb(visual_layers.intensity_to_uint8(image[:, :, z].T))
    base = visual_layers.overlay_rgb(base, vessel_mask[:, :, z].T, (0, 220, 255), 0.18)
    base = visual_layers.overlay_rgb(
        base, calc_mask[:, :, z].T, runtime.CALCIUM_OVERLAY_RGB, 0.7
    )
    for branch, bmask in branches.items():
        if np.any(bmask[:, :, z]):
            base = visual_layers.overlay_rgb(
                base, bmask[:, :, z].T, visual_layers.branch_color(branch), 0.24
            )
    label_text = f"{best['branch']} {reliability.stenosis_interval_label(float(best['stenosis_ratio']))}"
    base = visual_layers.draw_stenosis_marker(
        base, (bx, by, bz), label_text, color = (255, 0, 0)
    )
    mip_with_vessels = visual_layers.gray_to_rgb(
        np.max(visual_layers.intensity_to_uint8(image), axis = 2).T
    )
    vessel_mip = np.max(vessel_mask.astype(np.uint8), axis = 2).T.astype(bool)
    calc_mip = np.max(calc_mask.astype(np.uint8), axis = 2).T.astype(bool)
    mip_with_vessels = visual_layers.overlay_rgb(
        mip_with_vessels, vessel_mip, (0, 220, 255), 0.2
    )
    mip_with_vessels = visual_layers.overlay_rgb(
        mip_with_vessels, calc_mip, runtime.CALCIUM_OVERLAY_RGB, 0.72
    )
    mip_marked = mip_with_vessels.copy()
    mip_label_items: List[Dict[str, Any]] = []
    for pt in points[:6]:
        (px, py, pz) = pt["point"]
        mip_label_items.append(
            {
                "x": int(px),
                "y": int(py),
                "label": f"{pt['branch']} {reliability.stenosis_interval_label(float(pt['stenosis_ratio']))}",
            }
        )
    mip_marked = visual_layers.draw_stenosis_markers_with_auto_labels(
        mip_marked, mip_label_items, color = (255, 0, 0), font_scale = 0.55
    )
    (fig, axes) = runtime.plt.subplots(
        1, 4, figsize = (24, 6), facecolor = runtime.visual_fig_bg()
    )
    fig.suptitle(
        f"{case_id} geometric narrowing QA marker",
        color = runtime.visual_text_color(),
        fontsize = 18,
        fontweight = "bold",
    )
    axes[0].imshow(base)
    axes[0].set_title(f"Marked axial slice Z={z}", color = runtime.visual_text_color())
    axes[0].axis("off")
    axes[1].imshow(mip_with_vessels)
    axes[1].set_title("Axial MIP (vessel overlay)", color = runtime.visual_text_color())
    axes[1].axis("off")
    axes[2].imshow(mip_marked)
    axes[2].set_title(
        "Axial MIP with geometric narrowing candidates", color = runtime.visual_text_color()
    )
    axes[2].axis("off")
    axes[3].axis("off")
    lines = [
        "Estimated geometric narrowing candidates",
        "Source: program-derived centerline diameter profile",
        "Not ImageCAS native annotation",
        "Not clinical stenosis ground truth",
        "",
    ]
    for pt in points[:8]:
        lines.append(
            f"{pt['branch']}: {reliability.stenosis_interval_label(float(pt['stenosis_ratio']))} | pos={float(pt['stenosis_position_norm']):.3f} | {pt['stenosis_segment']}"
        )
    axes[3].text(
        0.02,
        0.96,
        "\n".join(lines),
        ha = "left",
        va = "top",
        color = runtime.visual_text_color(),
        fontsize = 12,
        transform = axes[3].transAxes,
    )
    runtime.plt.tight_layout(rect = [0, 0, 1, 0.92])
    out_path = (out_dir / "figure_SF_stenosis_marked.png")
    runtime.plt.savefig(
        out_path, dpi = 170, bbox_inches = "tight", facecolor = runtime.visual_fig_bg()
    )
    runtime.plt.close(fig)
    return out_path

def save_case_visual_outputs(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Dict[str, str]:
    reliability_df = reliability.compute_case_narrowing_reliability(
        case_id, branches, branch_rows, calc_mask, spacing
    )
    if ((reliability_df is not None) and (not reliability_df.empty)):
        reliability_df.to_csv(
            (Path(out_dir) / "cta_concepts_narrowing_reliability.csv"),
            index = False,
            encoding = "utf-8-sig",
        )
        storage.write_json(
            (Path(out_dir) / "cta_concepts_narrowing_reliability.json"),
            reliability_df.to_dict("records"),
        )
        _merge_reliability_into_branch_csv(
            case_id, branch_rows, reliability_df, Path(out_dir)
        )
    outputs = save_enhanced_visual_outputs(
        case_id, image, vessel_mask, calc_mask, branches, branch_rows, spacing, out_dir
    )
    return outputs
