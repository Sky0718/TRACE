from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from typing import Tuple
import numpy as np
from . import runtime, models, measurements, reliability

def intensity_to_uint8(
    arr: np.ndarray, vmin: float = -150.0, vmax: float = 500.0
) -> np.ndarray:
    x = np.asarray(arr, dtype = np.float32)
    y = np.clip(((x - float(vmin)) / max(float((vmax - vmin)), 1e-06)), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype = np.uint8)
    return np.stack([g, g, g], axis = -1).astype(np.uint8)

def overlay_rgb(
    rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.55
) -> np.ndarray:
    out = np.asarray(rgb, dtype = np.float32).copy()
    m = np.asarray(mask).astype(bool)
    if np.any(m):
        col = np.asarray(color, dtype = np.float32)
        out[m] = ((out[m] * (1.0 - float(alpha))) + (col * float(alpha)))
    return np.clip(out, 0, 255).astype(np.uint8)

def branch_color(branch: str) -> Tuple[int, int, int]:
    b = str(branch)
    if b.startswith("OTHER"):
        return runtime.BRANCH_RGB["OTHER"]
    return runtime.BRANCH_RGB.get(b, runtime.BRANCH_RGB["OTHER"])

def select_slice_indices(mask: np.ndarray, count: int = 8) -> np.ndarray:
    active = np.where(mask.any(axis = (0, 1)))[0]
    if (active.size == 0):
        return np.array([(mask.shape[2] // 2)], dtype = np.int32)
    n = min(int(count), int(active.size))
    return active[np.linspace(0, (active.size - 1), n, dtype = np.int32)]

def branch_row_map(
    branch_rows: List[models.BranchConcept],
) -> Dict[str, models.BranchConcept]:
    return {str(row.branch): row for row in branch_rows}

def estimate_stenosis_points(
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    calc_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    row_lookup = branch_row_map(branch_rows)
    points: List[Dict[str, Any]] = []
    for branch, bmask in branches.items():
        branch_calc = (bmask & calc_mask) if (calc_mask is not None) else None
        prof = measurements.centerline_profile_arrays(
            bmask, spacing, calcium_mask = branch_calc
        )
        if not prof.get("ok"):
            continue
        path = np.asarray(prof.get("path", []), dtype = np.int32)
        cum = np.asarray(prof.get("cum", []), dtype = np.float32)
        ratio_profile = np.asarray(prof.get("ratio", []), dtype = np.float32)
        if (((path.shape[0] == 0) or (cum.size == 0)) or (ratio_profile.size == 0)):
            continue
        row = row_lookup.get(str(branch))
        total = float(cum[-1]) if (cum.size > 0) else 0.0
        if (row is not None):
            if (
                ((float(row.stenosis_ratio) <= 0.0)
                or (float(row.stenosis_position_norm) < 0.0))
                or (total <= 1e-06)
            ):
                continue
            target = (float(row.stenosis_position_norm) * total)
            idx = int(np.argmin(np.abs((cum - target))))
            ratio = float(row.stenosis_ratio)
            position = float(row.stenosis_position_norm)
            segment = str(row.stenosis_segment)
        else:
            idx = int(np.argmax(ratio_profile))
            ratio = float(ratio_profile[idx])
            if (ratio < float(runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD)):
                continue
            position = float((cum[idx] / max(total, 1e-06))) if (total > 1e-06) else -1.0
            segment = measurements.segment_name_from_norm(position)
        idx = int(np.clip(idx, 0, (path.shape[0] - 1)))
        p = path[idx]
        points.append(
            {
                "branch": str(branch),
                "point": (int(p[0]), int(p[1]), int(p[2])),
                "path_index": int(idx),
                "stenosis_ratio": float(np.clip(ratio, 0.0, 1.0)),
                "stenosis_position_norm": float(position),
                "stenosis_segment": str(segment),
            }
        )
    points.sort(
        key = lambda x: (float(x.get("stenosis_ratio", 0.0)), str(x.get("branch", ""))),
        reverse = True,
    )
    return points

def draw_stenosis_marker(
    rgb: np.ndarray,
    point: Tuple[int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    out = np.asarray(rgb, dtype = np.uint8).copy()
    x = int(point[0])
    y = int(point[1])
    (h, w) = out.shape[:2]
    if not ((0 <= x < w) and (0 <= y < h)):
        return out
    runtime.cv2.drawMarker(
        out,
        (x, y),
        tuple((int(v) for v in color)),
        markerType = runtime.cv2.MARKER_TILTED_CROSS,
        markerSize = 28,
        thickness = 4,
        line_type = runtime.cv2.LINE_AA,
    )
    label = str(label).strip()
    if label:
        tx = int(min(max((x + 14), 4), max(4, (w - 220))))
        ty = int(min(max((y - 14), 24), max(24, (h - 8))))
        runtime.cv2.putText(
            out,
            label,
            (tx, ty),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            runtime.cv2.LINE_AA,
        )
        runtime.cv2.putText(
            out,
            label,
            (tx, ty),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            tuple((int(v) for v in color)),
            2,
            runtime.cv2.LINE_AA,
        )
    return out

def _boxes_intersect(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 4
) -> bool:
    (ax1, ay1, ax2, ay2) = a
    (bx1, by1, bx2, by2) = b
    return not (
        ((((ax2 + margin) < bx1)
        or ((bx2 + margin) < ax1))
        or ((ay2 + margin) < by1))
        or ((by2 + margin) < ay1)
    )

def draw_stenosis_markers_with_auto_labels(
    rgb: np.ndarray,
    items: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (255, 0, 0),
    font_scale: float = 0.58,
) -> np.ndarray:
    out = np.asarray(rgb, dtype = np.uint8).copy()
    if not items:
        return out
    (h, w) = out.shape[:2]
    placed: List[Tuple[int, int, int, int]] = []
    for it in items:
        (x, y) = (int(it["x"]), int(it["y"]))
        if ((0 <= x < w) and (0 <= y < h)):
            runtime.cv2.drawMarker(
                out,
                (x, y),
                tuple((int(v) for v in color)),
                markerType = runtime.cv2.MARKER_TILTED_CROSS,
                markerSize = 28,
                thickness = 4,
                line_type = runtime.cv2.LINE_AA,
            )
    order = sorted(
        items,
        key = lambda d: (len(str(d.get("label", ""))), -int(d.get("y", 0))),
        reverse = True,
    )
    for it in order:
        (x, y) = (int(it["x"]), int(it["y"]))
        label = str(it.get("label", "")).strip()
        if (not label or not ((0 <= x < w) and (0 <= y < h))):
            continue
        ((tw, th), baseline) = runtime.cv2.getTextSize(
            label, runtime.cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        box_w = (tw + 14)
        box_h = ((th + baseline) + 10)
        candidates = [
            ((x + 18), ((y - box_h) - 10)),
            ((x + 18), (y + 10)),
            (((x - box_w) - 18), ((y - box_h) - 10)),
            (((x - box_w) - 18), (y + 10)),
            ((x + 22), (y - (box_h // 2))),
            (((x - box_w) - 22), (y - (box_h // 2))),
        ]
        best = None
        best_score = None
        for cx, cy in candidates:
            bx1 = int(np.clip(cx, 4, max(4, ((w - box_w) - 4))))
            by1 = int(np.clip(cy, 24, max(24, ((h - box_h) - 4))))
            bx2 = (bx1 + box_w)
            by2 = (by1 + box_h)
            box = (bx1, by1, bx2, by2)
            overlaps = sum((1 for p in placed if _boxes_intersect(box, p, margin = 6)))
            dist = (abs((((bx1 + bx2) // 2) - x)) + abs((((by1 + by2) // 2) - y)))
            score = ((overlaps * 10000) + dist)
            if ((best_score is None) or (score < best_score)):
                best_score = score
                best = box
                if ((overlaps == 0) and (dist < 160)):
                    break
        if (best is None):
            continue
        (bx1, by1, bx2, by2) = best
        placed.append(best)
        anchor_x = bx1 if (x <= ((bx1 + bx2) // 2)) else bx2
        anchor_y = (by1 + (box_h // 2))
        runtime.cv2.line(
            out,
            (x, y),
            (anchor_x, anchor_y),
            tuple((int(v) for v in color)),
            2,
            runtime.cv2.LINE_AA,
        )
        overlay = out.copy()
        runtime.cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (18, 18, 18), -1)
        out = runtime.cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
        runtime.cv2.rectangle(
            out,
            (bx1, by1),
            (bx2, by2),
            tuple((int(v) for v in color)),
            1,
            runtime.cv2.LINE_AA,
        )
        text_org = ((bx1 + 7), ((by2 - baseline) - 5))
        runtime.cv2.putText(
            out,
            label,
            text_org,
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            runtime.cv2.LINE_AA,
        )
        runtime.cv2.putText(
            out,
            label,
            text_org,
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            tuple((int(v) for v in color)),
            1,
            runtime.cv2.LINE_AA,
        )
    return out

def select_video_slices_with_forced(
    mask: np.ndarray, forced_z: List[int], count: int = runtime.MAX_VIDEO_FRAMES
) -> np.ndarray:
    base = select_slice_indices(mask, count = max(1, int(count)))
    z_values = set([int(x) for x in base.tolist()])
    for z in forced_z:
        zi = int(np.clip(int(z), 0, (mask.shape[2] - 1)))
        z_values.add(zi)
    return np.asarray(sorted(z_values), dtype = np.int32)

def resize_video_frame(rgb: np.ndarray, max_side: int = 960) -> np.ndarray:
    arr = np.asarray(rgb, dtype = np.uint8)
    (h, w) = arr.shape[:2]
    scale = min(1.0, (float(max_side) / max(float(h), float(w), 1.0)))
    if (scale < 0.999):
        arr = runtime.cv2.resize(
            arr,
            (max(2, int(round((w * scale)))), max(2, int(round((h * scale))))),
            interpolation = runtime.cv2.INTER_AREA,
        )
    (h, w) = arr.shape[:2]
    if ((h % 2) == 1):
        arr = arr[:-1, :, :]
    if ((w % 2) == 1):
        arr = arr[:, :-1, :]
    return np.ascontiguousarray(arr)

def add_inset_panel(
    base_rgb: np.ndarray,
    inset_rgb: np.ndarray,
    title: str = "Raw MIP",
    scale: float = 0.3,
    margin: int = 12,
) -> np.ndarray:
    out = np.asarray(base_rgb, dtype = np.uint8).copy()
    inset = np.asarray(inset_rgb, dtype = np.uint8).copy()
    (h, w) = out.shape[:2]
    (ih, iw) = inset.shape[:2]
    if ((((ih < 2) or (iw < 2)) or (h < 40)) or (w < 40)):
        return out
    target_w = max(80, int(round((w * float(scale)))))
    target_h = max(80, int(round((h * float(scale)))))
    resize_scale = min((float(target_w) / float(iw)), (float(target_h) / float(ih)))
    new_w = max(2, int(round((iw * resize_scale))))
    new_h = max(2, int(round((ih * resize_scale))))
    inset = runtime.cv2.resize(inset, (new_w, new_h), interpolation = runtime.cv2.INTER_AREA)
    x2 = max(((margin + new_w) + 6), (w - margin))
    y1 = (margin + 26)
    x1 = max(0, ((x2 - new_w) - 6))
    y2 = min(h, ((y1 + new_h) + 6))
    x2 = min(w, ((x1 + new_w) + 6))
    panel_h = (y2 - y1)
    panel_w = (x2 - x1)
    if ((panel_h < (new_h + 6)) or (panel_w < (new_w + 6))):
        return out
    overlay = out.copy()
    runtime.cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 18, 28), -1)
    out = runtime.cv2.addWeighted(overlay, 0.78, out, 0.22, 0)
    out[(y1 + 3) : ((y1 + 3) + new_h), (x1 + 3) : ((x1 + 3) + new_w)] = inset
    runtime.cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1, runtime.cv2.LINE_AA)
    runtime.cv2.putText(
        out,
        str(title),
        (x1, max(18, (y1 - 8))),
        runtime.cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        runtime.cv2.LINE_AA,
    )
    runtime.cv2.putText(
        out,
        str(title),
        (x1, max(18, (y1 - 8))),
        runtime.cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        runtime.cv2.LINE_AA,
    )
    return out

def write_video(
    path: Path, frames_rgb: List[np.ndarray], fps: float = runtime.VIDEO_FPS
) -> bool:
    if (not runtime.CV2_OK or not frames_rgb):
        return False
    frames = [resize_video_frame(f) for f in frames_rgb]
    (h, w) = frames[0].shape[:2]
    writer = runtime.cv2.VideoWriter(
        str(path), runtime.cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h))
    )
    if not writer.isOpened():
        return False
    for f in frames:
        if (f.shape[:2] != (h, w)):
            f = runtime.cv2.resize(f, (w, h), interpolation = runtime.cv2.INTER_AREA)
        writer.write(runtime.cv2.cvtColor(f, runtime.cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists()

def _draw_binary_edge(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = runtime.VESSEL_EDGE_RGB,
    thickness: int = runtime.EDGE_THICKNESS_PX,
) -> np.ndarray:
    out = np.asarray(rgb, dtype = np.uint8).copy()
    if not runtime.CV2_OK:
        return out
    m = np.asarray(mask).astype(np.uint8)
    if ((m.ndim != 2) or not np.any(m)):
        return out
    kernel = np.ones((3, 3), np.uint8)
    grad = runtime.cv2.morphologyEx(m, runtime.cv2.MORPH_GRADIENT, kernel).astype(bool)
    if (int(thickness) > 1):
        grad = runtime.cv2.dilate(
            grad.astype(np.uint8), np.ones((int(thickness), int(thickness)), np.uint8)
        ).astype(bool)
    out[grad] = np.asarray(color, dtype = np.uint8)
    return out

def overlay_mask_opaque_exact(
    rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], draw_edge: bool = True
) -> np.ndarray:
    out = np.asarray(rgb, dtype = np.uint8).copy()
    m = np.asarray(mask).astype(bool)
    if np.any(m):
        out[m] = np.asarray(color, dtype = np.uint8)
    if draw_edge:
        out = _draw_binary_edge(
            out, m, color = runtime.VESSEL_EDGE_RGB, thickness = runtime.EDGE_THICKNESS_PX
        )
    return out

def _project_mask(mask: np.ndarray, axis: int) -> np.ndarray:
    return np.max(mask.astype(np.uint8), axis = int(axis)).astype(bool)

def _full_vessel_projection_rgb(
    image: np.ndarray, vessel_mask: np.ndarray, calc_mask: Optional[np.ndarray], axis: int
) -> np.ndarray:
    base = np.max(intensity_to_uint8(image), axis = int(axis))
    vessel = _project_mask(vessel_mask, int(axis))
    rgb = gray_to_rgb(base)
    rgb = overlay_mask_opaque_exact(rgb, vessel, runtime.VESSEL_OVERLAY_RGB, draw_edge = True)
    if ((calc_mask is not None) and np.any(calc_mask)):
        calc = _project_mask(calc_mask, int(axis))
        rgb = overlay_mask_opaque_exact(
            rgb, calc, runtime.CALCIUM_OVERLAY_RGB, draw_edge = False
        )
    return np.rot90(rgb)

def save_axial_overlay_video(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if not runtime.CV2_OK:
        return None
    z_list = select_slice_indices(
        vessel_mask,
        count = min(
            runtime.MAX_VIDEO_FRAMES, max(1, int(np.sum(vessel_mask.any(axis = (0, 1)))))
        ),
    )
    frames = []
    for z in z_list:
        rgb = gray_to_rgb(intensity_to_uint8(image[:, :, int(z)].T))
        rgb = overlay_mask_opaque_exact(
            rgb, vessel_mask[:, :, int(z)].T, runtime.VESSEL_OVERLAY_RGB, draw_edge = True
        )
        if np.any(calc_mask[:, :, int(z)]):
            rgb = overlay_mask_opaque_exact(
                rgb, calc_mask[:, :, int(z)].T, runtime.CALCIUM_OVERLAY_RGB, draw_edge = False
            )
        runtime.cv2.putText(
            rgb,
            f"{case_id} | Z={int(z)} | exact vessel mask",
            (12, 28),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            runtime.cv2.LINE_AA,
        )
        frames.append(rgb)
    out_path = (out_dir / "video_A_axial_vessel_calcium.mp4")
    return out_path if write_video(out_path, frames) else None

def save_branch_overlay_video(
    case_id: str,
    image: np.ndarray,
    branches: Dict[str, np.ndarray],
    calc_mask: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if (not runtime.CV2_OK or not branches):
        return None
    union = np.zeros_like(calc_mask, dtype = bool)
    for bmask in branches.values():
        union |= bmask.astype(bool)
    z_list = select_slice_indices(
        union,
        count = min(runtime.MAX_VIDEO_FRAMES, max(1, int(np.sum(union.any(axis = (0, 1)))))),
    )
    frames = []
    for z in z_list:
        rgb = gray_to_rgb(intensity_to_uint8(image[:, :, int(z)].T))
        for branch, bmask in branches.items():
            rgb = overlay_rgb(rgb, bmask[:, :, int(z)].T, branch_color(branch), 0.92)
        if np.any(calc_mask[:, :, int(z)]):
            rgb = overlay_mask_opaque_exact(
                rgb, calc_mask[:, :, int(z)].T, runtime.CALCIUM_OVERLAY_RGB, draw_edge = False
            )
        runtime.cv2.putText(
            rgb,
            f"{case_id} | Z={int(z)} | branch QA masks",
            (12, 28),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            runtime.cv2.LINE_AA,
        )
        frames.append(rgb)
    out_path = (out_dir / "video_SF_branch_masks.mp4")
    return out_path if write_video(out_path, frames) else None

def save_stenosis_marked_video(
    case_id: str,
    image: np.ndarray,
    vessel_mask: np.ndarray,
    calc_mask: np.ndarray,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    spacing: Tuple[float, float, float],
    out_dir: Path,
) -> Optional[Path]:
    if (not runtime.CV2_OK or not branches):
        return None
    points = estimate_stenosis_points(branches, branch_rows, spacing, calc_mask = calc_mask)
    if not points:
        return None
    union = vessel_mask.astype(bool).copy()
    for bmask in branches.values():
        union |= bmask.astype(bool)
    forced_z = [int(pt["point"][2]) for pt in points[:8]]
    z_list = select_video_slices_with_forced(
        union,
        forced_z,
        count = min(runtime.MAX_VIDEO_FRAMES, max(1, int(np.sum(union.any(axis = (0, 1)))))),
    )
    mip_inset = gray_to_rgb(np.max(intensity_to_uint8(image), axis = 2).T)
    vessel_mip = np.max(vessel_mask.astype(np.uint8), axis = 2).T.astype(bool)
    calc_mip = np.max(calc_mask.astype(np.uint8), axis = 2).T.astype(bool)
    mip_inset = overlay_rgb(mip_inset, vessel_mip, (0, 220, 255), 0.2)
    mip_inset = overlay_rgb(mip_inset, calc_mip, runtime.CALCIUM_OVERLAY_RGB, 0.72)
    frames: List[np.ndarray] = []
    for z in z_list:
        zi = int(z)
        rgb = gray_to_rgb(intensity_to_uint8(image[:, :, zi].T))
        rgb = overlay_rgb(rgb, vessel_mask[:, :, zi].T, (0, 220, 255), 0.16)
        for branch, bmask in branches.items():
            if np.any(bmask[:, :, zi]):
                rgb = overlay_rgb(rgb, bmask[:, :, zi].T, branch_color(branch), 0.24)
        rgb = overlay_rgb(rgb, calc_mask[:, :, zi].T, runtime.CALCIUM_OVERLAY_RGB, 0.62)
        current_items: List[Dict[str, Any]] = []
        current_labels: List[str] = []
        for pt in points[:8]:
            (px, py, pz) = pt["point"]
            if (abs((int(pz) - zi)) <= 1):
                label_txt = f"{pt['branch']} {reliability.stenosis_interval_label(float(pt['stenosis_ratio']))}"
                current_items.append({"x": int(px), "y": int(py), "label": label_txt})
                current_labels.append(label_txt)
        if current_items:
            rgb = draw_stenosis_markers_with_auto_labels(
                rgb, current_items, color = (255, 0, 0), font_scale = 0.52
            )
        rgb = add_inset_panel(rgb, mip_inset, title = "MIP + vessels", scale = 0.28, margin = 12)
        top_text = f"{case_id} | Z={zi} | candidates={len(current_items)} | red X=estimated geometric narrowing"
        runtime.cv2.putText(
            rgb,
            top_text,
            (12, 28),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            runtime.cv2.LINE_AA,
        )
        runtime.cv2.putText(
            rgb,
            top_text,
            (12, 28),
            runtime.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            runtime.cv2.LINE_AA,
        )
        if current_labels:
            shown = current_labels[:3]
            extra = (len(current_labels) - len(shown))
            bottom_text = (
                ("Current slice: "
                + "; ".join(shown))
                + (f"; +{extra} more" if (extra > 0) else "")
            )
            (h, w) = rgb.shape[:2]
            runtime.cv2.rectangle(rgb, (8, (h - 34)), ((w - 8), (h - 8)), (12, 18, 28), -1)
            runtime.cv2.putText(
                rgb,
                bottom_text,
                (14, (h - 14)),
                runtime.cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 0, 0),
                3,
                runtime.cv2.LINE_AA,
            )
            runtime.cv2.putText(
                rgb,
                bottom_text,
                (14, (h - 14)),
                runtime.cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                1,
                runtime.cv2.LINE_AA,
            )
        frames.append(rgb)
    out_path = (out_dir / "video_SF_stenosis_marked.mp4")
    return out_path if write_video(out_path, frames) else None
