from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import runtime, quality_tables, reliability, storage

def build_geometry_case_row(
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    qa_row: Dict[str, Any],
    visual_row: Dict[str, Any],
) -> Dict[str, Any]:
    cid = str(case_row.get("case_id", case_dir.name))
    out = dict(case_row)
    out["case_id"] = cid
    out["case_output_dir"] = str(case_dir)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    max_ratio = storage.to_float(
        out.get(
            "max_geometric_narrowing_index",
            out.get("max_stenosis_ratio", out.get("stenosis_ratio", 0.0)),
        ),
        0.0,
    )
    out["geometric_narrowing_index"] = float(max_ratio)
    out["geometric_narrowing_category"] = reliability.stenosis_interval_label(max_ratio)
    out["narrowing_candidate_present"] = int(
        ((max_ratio > 0.0)
        and (storage.to_float(
            out.get("stenosis_position_norm", out.get("stenosis_position", -1.0)), -1.0
        )
        >= 0.0))
    )
    for b in ["LM", "LAD", "LCX", "RCA"]:
        out[f"{b}_detected"] = int(qa_row.get(f"{b}_detected", 0))
        out[f"{b}_length_mm"] = quality_tables.branch_value_from_df(
            branch_df, b, "vessel_length_mm", default = 0.0
        )
        out[f"{b}_mean_diameter_mm"] = quality_tables.branch_value_from_df(
            branch_df, b, "vessel_diameter_mean_mm", default = 0.0
        )
        out[f"{b}_min_diameter_mm"] = quality_tables.branch_value_from_df(
            branch_df, b, "vessel_diameter_min_mm", default = 0.0
        )
        out[f"{b}_max_diameter_mm"] = quality_tables.branch_value_from_df(
            branch_df, b, "vessel_diameter_max_mm", default = 0.0
        )
        out[f"{b}_tortuosity"] = quality_tables.branch_value_from_df(
            branch_df, b, "tortuosity", default = 0.0
        )
        out[f"{b}_geometric_narrowing_index"] = quality_tables.branch_value_from_df(
            branch_df, b, "stenosis_ratio", default = 0.0
        )
        out[f"{b}_narrowing_position_norm"] = quality_tables.branch_value_from_df(
            branch_df, b, "stenosis_position_norm", default = -1.0
        )
        out[f"{b}_calcium_volume_mm3"] = quality_tables.branch_value_from_df(
            branch_df, b, "calcium_volume_mm3", default = 0.0
        )
        out[f"{b}_calcium_burden_pct"] = quality_tables.branch_value_from_df(
            branch_df, b, "calcium_burden_pct", default = 0.0
        )
    for key in [
        "branch_count_from_table",
        "valid_branch_count",
        "branches_detected",
        "missing_major_branches",
        "zero_length_branches",
        "branch_failure_flag",
        "centerline_ok_count",
        "centerline_fallback_count",
        "centerline_failed_count",
        "invalid_narrowing_position_flag",
        "no_narrowing_candidate_flag",
        "manual_review_needed",
        "qa_score",
        "runtime_seconds",
    ]:
        out[key] = qa_row.get(key, "")
    out["visual_existing_count"] = visual_row.get("visual_existing_count", 0)
    out["visual_expected_count"] = visual_row.get(
        "visual_expected_count", len(runtime.EXPECTED_VISUAL_OUTPUTS)
    )
    return out

def geometry_dataset_summary_rows(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(metric: str, value: Any, note: str = "") -> None:
        rows.append({"metric": metric, "value": storage.json_safe(value), "note": note})

    add("case_count", int(len(case_df)))
    add("branch_row_count", int(len(branch_df)))
    add("geometric_narrowing_candidate_count", int(len(sten_df)))
    if not qa_df.empty:
        add(
            "manual_review_needed_cases",
            int(
                pd.to_numeric(qa_df.get("manual_review_needed", 0), errors = "coerce")
                .fillna(0)
                .sum()
            ),
        )
        add(
            "branch_failure_cases",
            int(
                pd.to_numeric(qa_df.get("branch_failure_flag", 0), errors = "coerce")
                .fillna(0)
                .sum()
            ),
        )
        add(
            "invalid_narrowing_position_cases",
            int(
                pd.to_numeric(
                    qa_df.get("invalid_narrowing_position_flag", 0), errors = "coerce"
                )
                .fillna(0)
                .sum()
            ),
        )
        add(
            "mean_qa_score",
            float(pd.to_numeric(qa_df.get("qa_score", np.nan), errors = "coerce").mean()),
        )
    for col in [
        "total_vessel_volume_mm3",
        "calcium_volume_mm3",
        "calcium_burden_pct",
        "geometric_narrowing_index",
        "tortuosity_mean",
        "tortuosity_max",
    ]:
        if (col in case_df.columns):
            s = pd.to_numeric(case_df[col], errors = "coerce")
            add(f"{col}_mean", float(s.mean()))
            add(f"{col}_median", float(s.median()))
            add(f"{col}_min", float(s.min()))
            add(f"{col}_max", float(s.max()))
    for col in [
        "vessel_length_mm",
        "vessel_diameter_mean_mm",
        "vessel_diameter_min_mm",
        "tortuosity",
        "geometric_narrowing_index",
    ]:
        if (col in branch_df.columns):
            s = pd.to_numeric(branch_df[col], errors = "coerce")
            add(f"branch_{col}_mean", float(s.mean()))
            add(f"branch_{col}_median", float(s.median()))
    add(
        "terminology_note",
        "geometric_narrowing_index is the paper-safe alias of the old stenosis_ratio output.",
        "Avoid overclaiming clinical stenosis diagnosis without ground-truth labels.",
    )
    return pd.DataFrame(rows)

def build_topology_case_row(
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    qa_row: Dict[str, Any],
    visual_row: Dict[str, Any],
) -> Dict[str, Any]:
    out = build_geometry_case_row(case_dir, case_row, branch_df, qa_row, visual_row)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    out["semantic_branch_warning"] = (
        "Branch names are algorithmic estimates from binary masks; full-vessel visuals should be used as primary image evidence."
    )
    out["main_visualization_mode"] = "opaque_full_vessel_mask_exact_thickness"
    return out

def topology_dataset_summary_rows(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
) -> pd.DataFrame:
    df = geometry_dataset_summary_rows(case_df, branch_df, qa_df, sten_df)
    rows = df.to_dict("records") if ((df is not None) and (not df.empty)) else []
    rows.append(
        {
            "metric": "v1_3_visual_correction",
            "value": "opaque_full_vessel_mask_exact_thickness",
            "note": "Main MIP figures use the original vessel mask projection to preserve small vessels and avoid unreliable semantic-branch coloring.",
        }
    )
    rows.append(
        {
            "metric": "semantic_branch_caution",
            "value": "LM/LAD/LCX/RCA algorithmic estimates",
            "note": "ImageCAS-like binary labels do not provide native semantic branch labels; LM should not be overclaimed without external validation.",
        }
    )
    return pd.DataFrame(rows)

def build_reliability_case_row(
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    qa_row: Dict[str, Any],
    visual_row: Dict[str, Any],
) -> Dict[str, Any]:
    out = build_topology_case_row(case_dir, case_row, branch_df, qa_row, visual_row)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    for key in [
        "narrowing_candidate_count",
        "high_confidence_narrowing_count",
        "moderate_confidence_narrowing_count",
        "low_confidence_narrowing_count",
        "low_confidence_narrowing_branches",
        "sharp_spike_narrowing_branches",
        "endpoint_narrowing_branches",
        "narrowing_review_needed_flag",
    ]:
        out[key] = qa_row.get(
            key, 0 if (key.endswith("count") or key.endswith("flag")) else ""
        )
    out["terminology_note"] = (
        "All paper-facing labels use geometric narrowing; legacy stenosis_* columns are retained only for backward compatibility."
    )
    return out

def reliability_dataset_summary_rows(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
) -> pd.DataFrame:
    df = topology_dataset_summary_rows(case_df, branch_df, qa_df, sten_df)
    rows = df.to_dict("records") if ((df is not None) and (not df.empty)) else []

    def add(metric: str, value: Any, note: str = "") -> None:
        rows.append({"metric": metric, "value": storage.json_safe(value), "note": note})

    if not qa_df.empty:
        for col in [
            "narrowing_candidate_count",
            "high_confidence_narrowing_count",
            "moderate_confidence_narrowing_count",
            "low_confidence_narrowing_count",
            "narrowing_review_needed_flag",
        ]:
            if (col in qa_df.columns):
                add(
                    f"{col}_total",
                    int(pd.to_numeric(qa_df[col], errors = "coerce").fillna(0).sum()),
                )
    if not branch_df.empty:
        bdf = reliability._ensure_reliability_columns(branch_df)
        for col in [
            "candidate_reliability_score",
            "peak_width_mm",
            "area_evidence_at_peak",
            "high_hu_proximity_at_peak",
        ]:
            if (col in bdf.columns):
                s = pd.to_numeric(bdf[col], errors = "coerce")
                if s.notna().any():
                    add(f"branch_{col}_mean", float(s.mean()))
                    add(f"branch_{col}_median", float(s.median()))
        for col in [
            "sharp_spike_flag",
            "candidate_near_endpoint_flag",
            "candidate_near_bifurcation_proxy_flag",
            "high_hu_overlap_flag",
            "candidate_reliable_for_statistics",
        ]:
            if (col in bdf.columns):
                add(
                    f"branch_{col}_count",
                    int(pd.to_numeric(bdf[col], errors = "coerce").fillna(0).sum()),
                )
    add(
        "v1_4_terminology",
        "geometric narrowing",
        "Paper-facing terms avoid clinical stenosis claims on ImageCAS-style masks.",
    )
    add(
        "v1_4_reliability_flags",
        "enabled",
        "Candidate confidence uses endpoint, sharp-spike, lesion-length, high-HU overlap, area agreement, centerline quality and tortuosity checks.",
    )
    return pd.DataFrame(rows)

def build_master_case_row(
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    qa_row: Dict[str, Any],
    visual_row: Dict[str, Any],
) -> Dict[str, Any]:
    bdf = (
        reliability._add_primary_statistics_flags_to_branch_df(branch_df)
        if ((branch_df is not None) and (not branch_df.empty))
        else branch_df
    )
    out = build_reliability_case_row(case_dir, case_row, bdf, qa_row, visual_row)
    out.update(
        reliability._primary_counts_from_branch_df(
            bdf if isinstance(bdf, pd.DataFrame) else pd.DataFrame()
        )
    )
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    out["primary_statistics_note"] = (
        "Primary narrowing statistics are stricter than broad exploratory candidate statistics."
    )
    return out

def dataset_summary_rows(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
) -> pd.DataFrame:
    bdf = (
        reliability._add_primary_statistics_flags_to_branch_df(branch_df)
        if ((branch_df is not None) and (not branch_df.empty))
        else branch_df
    )
    sdf = (
        reliability._sync_primary_flags_into_narrowing_candidates(
            sten_df.to_dict("records"), bdf
        )
        if ((sten_df is not None) and (not sten_df.empty))
        else []
    )
    sdf_df = pd.DataFrame(sdf) if sdf else sten_df
    df = reliability_dataset_summary_rows(
        case_df, bdf if isinstance(bdf, pd.DataFrame) else branch_df, qa_df, sdf_df
    )
    rows = df.to_dict("records") if ((df is not None) and (not df.empty)) else []

    def add(metric: str, value: Any, note: str = "") -> None:
        rows.append({"metric": metric, "value": storage.json_safe(value), "note": note})

    if (isinstance(bdf, pd.DataFrame) and (not bdf.empty)):
        cand = (
            pd.to_numeric(bdf.get("candidate_present", 0), errors = "coerce")
            .fillna(0)
            .astype(int)
        )
        primary = (
            pd.to_numeric(
                bdf.get("candidate_reliable_for_primary_statistics", 0), errors = "coerce"
            )
            .fillna(0)
            .astype(int)
        )
        broad = (
            pd.to_numeric(
                bdf.get(
                    "candidate_reliable_for_broad_statistics",
                    bdf.get("candidate_reliable_for_statistics", 0),
                ),
                errors = "coerce",
            )
            .fillna(0)
            .astype(int)
        )
        add(
            "primary_reliable_narrowing_candidate_total",
            int(((cand == 1) & (primary == 1)).sum()),
            runtime.PRIMARY_STATISTICS_RULE_TEXT,
        )
        add(
            "primary_excluded_narrowing_candidate_total",
            int(((cand == 1) & (primary == 0)).sum()),
            "Candidates excluded from primary but retained for exploratory analysis.",
        )
        add(
            "broad_reliable_narrowing_candidate_total",
            int(((cand == 1) & (broad == 1)).sum()),
            "Broad/exploratory reliability count.",
        )
        if ("candidate_primary_exclusion_reason" in bdf.columns):
            reason_counts: Dict[str, int] = {}
            excluded = bdf[((cand == 1) & (primary == 0))]
            for text in excluded["candidate_primary_exclusion_reason"].astype(str).tolist():
                for part in text.split("|"):
                    part = part.strip()
                    if (part and (part != "included_primary_statistics")):
                        reason_counts[part] = (reason_counts.get(part, 0) + 1)
            add(
                "primary_exclusion_reason_counts",
                json.dumps(reason_counts, ensure_ascii = False, sort_keys = True),
                "Reason counts among exploratory candidates excluded from primary statistics.",
            )
    add("v1_5_6_primary_statistics_flag", "enabled", runtime.PRIMARY_STATISTICS_RULE_TEXT)
    return pd.DataFrame(rows)
