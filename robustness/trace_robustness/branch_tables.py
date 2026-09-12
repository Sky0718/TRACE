from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from . import runtime, quality_tables, reliability, storage

def build_geometry_branch_table(branch_df: pd.DataFrame) -> pd.DataFrame:
    if ((branch_df is None) or branch_df.empty):
        return pd.DataFrame()
    out = branch_df.copy()
    if ("stenosis_ratio" in out.columns):
        out["geometric_narrowing_index"] = pd.to_numeric(
            out["stenosis_ratio"], errors = "coerce"
        ).fillna(0.0)
        out["geometric_narrowing_category"] = out["geometric_narrowing_index"].apply(
            reliability.stenosis_interval_label
        )
    if ("stenosis_position_norm" in out.columns):
        pos = pd.to_numeric(out["stenosis_position_norm"], errors = "coerce").fillna(-1.0)
        ratio = pd.to_numeric(
            out.get("geometric_narrowing_index", 0.0), errors = "coerce"
        ).fillna(0.0)
        out["narrowing_position_valid"] = (
            (ratio <= 0.0) | ((pos >= 0.0) & (pos <= 1.0))
        ).astype(int)
    if ("vessel_length_mm" in out.columns):
        out["branch_valid"] = (
            pd.to_numeric(out["vessel_length_mm"], errors = "coerce").fillna(0.0) > 0
        ).astype(int)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    return out

def build_topology_branch_table(branch_df: pd.DataFrame) -> pd.DataFrame:
    out = build_geometry_branch_table(branch_df)
    if ((out is None) or out.empty):
        return out
    length = pd.to_numeric(out.get("vessel_length_mm", 0.0), errors = "coerce").fillna(0.0)
    tort = pd.to_numeric(out.get("tortuosity", 0.0), errors = "coerce").fillna(0.0)
    ratio = pd.to_numeric(
        out.get("geometric_narrowing_index", out.get("stenosis_ratio", 0.0)),
        errors = "coerce",
    ).fillna(0.0)
    out["short_branch_flag"] = (
        (length > 0) & (length < runtime.SHORT_BRANCH_NARROWING_LENGTH_MM)
    ).astype(int)
    out["tortuosity_outlier_flag"] = (tort > runtime.TORTUOSITY_OUTLIER_THRESHOLD).astype(
        int
    )
    out["narrowing_candidate_reliable"] = (
        ((ratio > 0)
        & (length >= runtime.SHORT_BRANCH_NARROWING_LENGTH_MM))
        & (tort <= runtime.TORTUOSITY_OUTLIER_THRESHOLD)
    ).astype(int)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    return out

def build_reliability_branch_table(branch_df: pd.DataFrame) -> pd.DataFrame:
    out = build_topology_branch_table(branch_df)
    if ((out is None) or out.empty):
        return out
    out = reliability._ensure_reliability_columns(out)
    ratio = pd.to_numeric(
        out.get("geometric_narrowing_index", out.get("stenosis_ratio", 0.0)),
        errors = "coerce",
    ).fillna(0.0)
    rel_score = pd.to_numeric(
        out.get("candidate_reliability_score", np.nan), errors = "coerce"
    )
    out["candidate_present"] = (
        (ratio > 0.0)
        & (
            pd.to_numeric(out.get("stenosis_position_norm", -1), errors = "coerce").fillna(-1)
            >= 0
        )
    ).astype(int)
    out["narrowing_candidate_reliable"] = (
        (out["candidate_present"].astype(int) == 1)
        & (rel_score.fillna(0.0) >= runtime.NARROWING_MODERATE_CONFIDENCE_SCORE)
    ).astype(int)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    return out

def build_reliable_candidate_rows(branch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if ((branch_df is None) or branch_df.empty):
        return []
    bdf = reliability._ensure_reliability_columns(branch_df)
    rows: List[Dict[str, Any]] = []
    for _, r in bdf.iterrows():
        ratio = storage.to_float(
            r.get("stenosis_ratio", r.get("geometric_narrowing_index", 0.0)), 0.0
        )
        presence = storage.to_int(r.get("stenosis_presence", int((ratio > 0.0))), 0)
        if ((ratio <= 0.0) and (presence <= 0)):
            continue
        pos = storage.to_float(r.get("stenosis_position_norm", -1.0), -1.0)
        rows.append(
            {
                "case_id": str(r.get("case_id", "")),
                "branch": str(r.get("branch", "")),
                "branch_name": str(r.get("branch_name", "")),
                "geometric_narrowing_index": float(ratio),
                "geometric_narrowing_category": reliability.stenosis_interval_label(ratio),
                "narrowing_position_norm": float(pos),
                "narrowing_position_valid": int((0.0 <= pos <= 1.0)),
                "narrowing_segment": str(r.get("stenosis_segment", "")),
                "lesion_length_mm": storage.to_float(r.get("lesion_length_mm", 0.0), 0.0),
                "candidate_reliability_score": storage.to_float(
                    r.get("candidate_reliability_score", np.nan), np.nan
                ),
                "candidate_confidence_category": str(
                    r.get("candidate_confidence_category", "not_available")
                ),
                "candidate_reliable_for_statistics": storage.to_int(
                    r.get("candidate_reliable_for_statistics", 0), 0
                ),
                "candidate_near_endpoint_flag": storage.to_int(
                    r.get("candidate_near_endpoint_flag", 0), 0
                ),
                "candidate_near_bifurcation_proxy_flag": storage.to_int(
                    r.get("candidate_near_bifurcation_proxy_flag", 0), 0
                ),
                "sharp_spike_flag": storage.to_int(r.get("sharp_spike_flag", 0), 0),
                "short_lesion_flag": storage.to_int(r.get("short_lesion_flag", 0), 0),
                "high_hu_overlap_flag": storage.to_int(r.get("high_hu_overlap_flag", 0), 0),
                "area_agreement_flag": storage.to_int(r.get("area_agreement_flag", 0), 0),
                "peak_width_mm": storage.to_float(r.get("peak_width_mm", np.nan), np.nan),
                "area_evidence_at_peak": storage.to_float(
                    r.get("area_evidence_at_peak", np.nan), np.nan
                ),
                "high_hu_proximity_at_peak": storage.to_float(
                    r.get("high_hu_proximity_at_peak", np.nan), np.nan
                ),
                "diameter_flank_drop_at_peak": storage.to_float(
                    r.get("diameter_flank_drop_at_peak", np.nan), np.nan
                ),
                "reliability_notes": str(r.get("reliability_notes", "")),
                "reference_diameter_note": "reference diameter is estimated from a smoothed diameter profile; not clinical QCA ground truth",
                "vessel_length_mm": storage.to_float(r.get("vessel_length_mm", 0.0), 0.0),
                "min_diameter_mm": storage.to_float(
                    r.get("vessel_diameter_min_mm", 0.0), 0.0
                ),
                "mean_diameter_mm": storage.to_float(
                    r.get("vessel_diameter_mean_mm", 0.0), 0.0
                ),
                "high_hu_candidate_volume_mm3": storage.to_float(
                    r.get("calcium_volume_mm3", 0.0), 0.0
                ),
                "high_hu_candidate_burden_pct": storage.to_float(
                    r.get("calcium_burden_pct", 0.0), 0.0
                ),
                "algorithm_version": runtime.ALGORITHM_VERSION,
            }
        )
    return rows

def _branch_assignment_confidence_category(score: float) -> str:
    s = float(score)
    if (s >= 0.8):
        return "high"
    if (s >= 0.55):
        return "moderate"
    if (s > 0.0):
        return "low_review"
    return "unavailable"

def compute_branch_assignment_confidence(
    branch_df: pd.DataFrame, branch_info: Dict[str, Any]
) -> pd.DataFrame:
    if ((branch_df is None) or branch_df.empty):
        return branch_df
    out = branch_df.copy()
    method = str(branch_info.get("method", "")) if isinstance(branch_info, dict) else ""
    warnings_all = []
    if isinstance(branch_info, dict):
        warnings_all.extend(
            [
                quality_tables.clean_warning_value(w)
                for w in branch_info.get("warnings", [])
                if quality_tables.clean_warning_value(w)
            ]
        )
        left_info = (
            branch_info.get("left_split", {})
            if isinstance(branch_info.get("left_split", {}), dict)
            else {}
        )
        if quality_tables.clean_warning_value(left_info.get("warning", "")):
            warnings_all.append(
                quality_tables.clean_warning_value(left_info.get("warning", ""))
            )
        if quality_tables.clean_warning_value(left_info.get("lm_repair", "")):
            warnings_all.append(
                quality_tables.clean_warning_value(left_info.get("lm_repair", ""))
            )
    warning_text = "|".join(sorted(set(warnings_all)))
    (scores, cats, notes) = ([], [], [])
    for _, r in out.iterrows():
        b = str(r.get("branch", ""))
        length = storage.to_float(r.get("vessel_length_mm", 0.0), 0.0)
        quality = quality_tables.classify_centerline_quality(
            r.get("centerline_quality_flag", "")
        )
        score = 0.0
        note = []
        if (b == "RCA"):
            score = 0.92 if (length >= 60) else 0.72 if (length >= 20) else 0.45
        elif (b == "LM"):
            if (3.0 <= length <= 25.0):
                score = 0.72
            elif (0.0 < length < 3.0):
                score = 0.38
                note.append("very_short_LM")
            elif (length > 25.0):
                score = 0.55
                note.append("long_LM_unusual")
            else:
                score = 0.0
                note.append("missing_or_zero_LM")
        elif (b in ("LAD", "LCX")):
            score = 0.86 if (length >= 45) else 0.66 if (length >= 15) else 0.42
        elif b.startswith("OTHER"):
            score = 0.25
            note.append("non_semantic_other_branch")
        else:
            score = 0.35
            note.append("unknown_branch_label")
        if (("fallback" in method.lower()) or ("fallback" in warning_text.lower())):
            score = min(score, 0.55)
            note.append("fallback_assignment")
        if (("rescued" in warning_text.lower()) or ("low_confidence" in warning_text.lower())):
            if (b in ("LAD", "LCX", "LM")):
                score = min(score, 0.6)
                note.append("rescued_or_low_confidence_left_split")
        if (quality == "failed"):
            score = min(score, 0.3)
            note.append("failed_centerline")
        elif (quality == "fallback"):
            score = min(score, 0.62)
            note.append("fallback_centerline")
        scores.append(float(np.clip(score, 0.0, 1.0)))
        cats.append(_branch_assignment_confidence_category(score))
        notes.append("|".join(sorted(set(note))) if note else "")
    out["branch_assignment_score"] = scores
    out["branch_assignment_confidence"] = cats
    out["branch_assignment_warning"] = notes
    out["branch_assignment_global_warning"] = warning_text
    out["branch_assignment_confidence_method"] = (
        "heuristic_semantic_assignment_confidence_v1"
    )
    return out

def enrich_branch_confidence_outputs(case_out: Path, branch_info: Dict[str, Any]) -> None:
    bpath = (case_out / "cta_concepts_branch_level.csv")
    branch_df = storage.read_csv_safe(bpath)
    if branch_df.empty:
        return
    branch_df = compute_branch_assignment_confidence(branch_df, branch_info)
    if ("calcium_volume_mm3" in branch_df.columns):
        branch_df["high_hu_candidate_volume_mm3"] = pd.to_numeric(
            branch_df["calcium_volume_mm3"], errors = "coerce"
        ).fillna(0.0)
    if ("calcium_burden_pct" in branch_df.columns):
        burden = pd.to_numeric(branch_df["calcium_burden_pct"], errors = "coerce").fillna(0.0)
        branch_df["high_hu_candidate_burden_pct"] = burden
        branch_df["contrast_lumen_contamination_risk_flag"] = (burden > 5.0).astype(int)
    branch_df["high_hu_interpretation_note"] = (
        "peripheral high-HU candidate in contrast CCTA; not non-contrast CAC ground truth"
    )
    branch_df["algorithm_version"] = runtime.ALGORITHM_VERSION
    branch_df.to_csv(bpath, index = False, encoding = "utf-8-sig")

def enrich_branch_level_outputs(case_out: Path, branch_info: Dict[str, Any]) -> None:
    enrich_branch_confidence_outputs(case_out, branch_info)
    bpath = (Path(case_out) / "cta_concepts_branch_level.csv")
    branch_df = storage.read_csv_safe(bpath)
    if branch_df.empty:
        return
    branch_df = reliability._add_primary_statistics_flags_to_branch_df(branch_df)
    branch_df["algorithm_version"] = runtime.ALGORITHM_VERSION
    branch_df.to_csv(bpath, index = False, encoding = "utf-8-sig")

def build_branch_master(branch_df: pd.DataFrame) -> pd.DataFrame:
    out = build_reliability_branch_table(branch_df)
    if ((out is None) or out.empty):
        return out
    out = reliability._add_primary_statistics_flags_to_branch_df(out)
    out["algorithm_version"] = runtime.ALGORITHM_VERSION
    return out

def build_stenosis_candidate_rows(branch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    bdf = (
        reliability._add_primary_statistics_flags_to_branch_df(branch_df)
        if ((branch_df is not None) and (not branch_df.empty))
        else branch_df
    )
    rows = build_reliable_candidate_rows(bdf)
    return reliability._sync_primary_flags_into_narrowing_candidates(
        rows, bdf if isinstance(bdf, pd.DataFrame) else pd.DataFrame()
    )
