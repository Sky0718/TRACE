from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import json
import numpy as np
import pandas as pd
from . import runtime, models, measurements, storage

def stenosis_interval_label(ratio: float) -> str:
    pct = float((np.clip(ratio, 0.0, 1.0) * 100.0))
    if (pct < 1.0):
        return "0% narrowing candidate"
    if (pct < 25.0):
        return "1-24% geometric narrowing"
    if (pct < 50.0):
        return "25-49% geometric narrowing"
    if (pct < 70.0):
        return "50-69% geometric narrowing"
    if (pct < 100.0):
        return "70-99% geometric narrowing"
    return "99-100% geometric narrowing"

def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)

def _narrowing_confidence_category(score: float, candidate_present: bool = True) -> str:
    if not candidate_present:
        return "no_candidate"
    if (float(score) >= runtime.NARROWING_HIGH_CONFIDENCE_SCORE):
        return "high"
    if (float(score) >= runtime.NARROWING_MODERATE_CONFIDENCE_SCORE):
        return "moderate"
    return "low_review"

def _profile_peak_width_mm(
    cum: np.ndarray, score_profile: np.ndarray, idx: int, level: float
) -> float:
    if (((cum.size == 0) or (score_profile.size == 0)) or (not (0 <= idx < score_profile.size))):
        return 0.0
    flags = np.asarray((score_profile >= float(level)), dtype = bool)
    a = int(idx)
    b = int(idx)
    while (((a - 1) >= 0) and flags[(a - 1)]):
        a -= 1
    while (((b + 1) < flags.size) and flags[(b + 1)]):
        b += 1
    if ((b >= cum.size) or (a >= cum.size)):
        return 0.0
    return float(max(0.0, (cum[b] - cum[a])))

def _local_flank_drop_score(diameter: np.ndarray, idx: int, window: int = 6) -> float:
    d = np.asarray(diameter, dtype = np.float32)
    if ((d.size == 0) or not (0 <= idx < d.size)):
        return 0.0
    a0 = max(0, (idx - int(window)))
    a1 = max(0, (idx - 2))
    b0 = min(d.size, (idx + 3))
    b1 = min(d.size, ((idx + int(window)) + 1))
    flanks = []
    if (a1 > a0):
        flanks.extend(d[a0:a1].tolist())
    if (b1 > b0):
        flanks.extend(d[b0:b1].tolist())
    if not flanks:
        return 0.0
    flank_ref = float(np.median(flanks))
    return float(np.clip((1.0 - (float(d[idx]) / max(flank_ref, 1e-06))), 0.0, 1.0))

def compute_narrowing_reliability_for_branch(
    branch: str,
    branch_mask: np.ndarray,
    row: Any,
    calc_mask: Optional[np.ndarray],
    spacing: Tuple[float, float, float],
) -> Dict[str, Any]:
    ratio = storage.to_float(_row_get(row, "stenosis_ratio", 0.0), 0.0)
    pos = storage.to_float(_row_get(row, "stenosis_position_norm", -1.0), -1.0)
    length = storage.to_float(_row_get(row, "vessel_length_mm", 0.0), 0.0)
    lesion_len = storage.to_float(_row_get(row, "lesion_length_mm", 0.0), 0.0)
    tort = storage.to_float(_row_get(row, "tortuosity", 0.0), 0.0)
    centerline_q = str(_row_get(row, "centerline_quality_flag", "unknown"))
    burden = storage.to_float(_row_get(row, "calcium_burden_pct", 0.0), 0.0)
    candidate_present = bool(((ratio > 0.0) and (0.0 <= pos <= 1.0)))
    result: Dict[str, Any] = {
        "branch": str(branch),
        "geometric_narrowing_index": float(ratio),
        "narrowing_position_norm": float(pos),
        "candidate_present": int(candidate_present),
        "candidate_near_endpoint_flag": 0,
        "candidate_near_bifurcation_proxy_flag": 0,
        "short_branch_flag": int(
            ((length > 0.0) and (length < runtime.SHORT_BRANCH_NARROWING_LENGTH_MM))
        ),
        "short_lesion_flag": 0,
        "sharp_spike_flag": 0,
        "high_hu_overlap_flag": 0,
        "centerline_quality_penalty_flag": int((centerline_q not in ("ok", "OK"))),
        "tortuosity_penalty_flag": int((tort > runtime.TORTUOSITY_OUTLIER_THRESHOLD)),
        "area_agreement_flag": 0,
        "peak_width_mm": 0.0,
        "area_evidence_at_peak": 0.0,
        "high_hu_proximity_at_peak": 0.0,
        "diameter_flank_drop_at_peak": 0.0,
        "candidate_reliability_score": 0.0,
        "candidate_confidence_category": "no_candidate",
        "candidate_reliable_for_statistics": 0,
        "reliability_notes": "no_geometric_narrowing_candidate",
    }
    if not candidate_present:
        return result
    notes: List[str] = []
    score = 100.0
    if (
        (pos < runtime.NARROWING_ENDPOINT_MARGIN)
        or (pos > (1.0 - runtime.NARROWING_ENDPOINT_MARGIN))
    ):
        result["candidate_near_endpoint_flag"] = 1
        score -= 22.0
        notes.append("near_endpoint")
    if ((0.45 <= pos <= 0.58) and (str(branch).upper() in ("LAD", "LCX", "LM"))):
        result["candidate_near_bifurcation_proxy_flag"] = 1
        score -= 8.0
        notes.append("possible_left_bifurcation_zone")
    if result["short_branch_flag"]:
        score -= 18.0
        notes.append("short_branch")
    if ((lesion_len > 0.0) and (lesion_len < max(
        runtime.NARROWING_MIN_LESION_LENGTH_MM, (0.012 * max(length, 1.0))
    ))):
        result["short_lesion_flag"] = 1
        score -= 16.0
        notes.append("short_lesion")
    if result["centerline_quality_penalty_flag"]:
        score -= 16.0
        notes.append(f"centerline_{centerline_q}")
    if result["tortuosity_penalty_flag"]:
        score -= 15.0
        notes.append("tortuosity_outlier")
    try:
        branch_calc = (branch_mask & calc_mask) if (calc_mask is not None) else None
        prof = measurements.centerline_profile_arrays(
            branch_mask, spacing, calcium_mask = branch_calc
        )
        if prof.get("ok"):
            cum = np.asarray(prof.get("cum", []), dtype = np.float32)
            ratio_profile = np.asarray(prof.get("ratio", []), dtype = np.float32)
            diameter = np.asarray(prof.get("diameter", []), dtype = np.float32)
            area_ratio = np.asarray(
                prof.get("ratio_area", np.zeros_like(ratio_profile)), dtype = np.float32
            )
            high_hu = np.asarray(
                prof.get("calcium_penalty", np.zeros_like(ratio_profile)), dtype = np.float32
            )
            if (cum.size and ratio_profile.size):
                target = (float(pos) * float(max(cum[-1], 1e-06)))
                idx = int(np.argmin(np.abs((cum - target))))
                idx = int(np.clip(idx, 0, (ratio_profile.size - 1)))
                peak_level = max(0.18, (0.6 * float(ratio_profile[idx])))
                peak_width = _profile_peak_width_mm(cum, ratio_profile, idx, peak_level)
                result["peak_width_mm"] = float(peak_width)
                if ((peak_width < runtime.NARROWING_SHARP_SPIKE_WIDTH_MM) and (ratio >= 0.2)):
                    result["sharp_spike_flag"] = 1
                    score -= 22.0
                    notes.append("sharp_spike_profile")
                if (area_ratio.size == ratio_profile.size):
                    area_peak = float(area_ratio[idx])
                    result["area_evidence_at_peak"] = area_peak
                    if (area_peak >= max(0.18, (0.5 * ratio))):
                        result["area_agreement_flag"] = 1
                        score += 8.0
                    else:
                        score -= 7.0
                        notes.append("weak_area_agreement")
                if (high_hu.size == ratio_profile.size):
                    high_peak = float(high_hu[idx])
                    result["high_hu_proximity_at_peak"] = high_peak
                    if (high_peak >= runtime.NARROWING_HIGH_HU_PROXIMITY_THRESHOLD):
                        result["high_hu_overlap_flag"] = 1
                        score -= 7.0
                        notes.append("high_hu_overlap")
                if (diameter.size == ratio_profile.size):
                    drop = _local_flank_drop_score(diameter, idx, window = 7)
                    result["diameter_flank_drop_at_peak"] = float(drop)
                    if ((drop >= 0.45) and result["sharp_spike_flag"]):
                        score -= 6.0
                        notes.append("large_single_point_diameter_drop")
    except Exception as exc:
        score -= 10.0
        notes.append(f"profile_reliability_failed:{type(exc).__name__}")
    if ((burden >= 3.0) and (ratio < 0.5)):
        score -= 4.0
        notes.append("branch_high_hu_burden")
    score = float(np.clip(score, 0.0, 100.0))
    result["candidate_reliability_score"] = score
    result["candidate_confidence_category"] = _narrowing_confidence_category(
        score, candidate_present = True
    )
    result["candidate_reliable_for_statistics"] = int(
        (score >= runtime.NARROWING_MODERATE_CONFIDENCE_SCORE)
    )
    result["reliability_notes"] = (
        "|".join(sorted(set(notes))) if notes else "stable_candidate"
    )
    return result

def compute_case_narrowing_reliability(
    case_id: str,
    branches: Dict[str, np.ndarray],
    branch_rows: List[models.BranchConcept],
    calc_mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in branch_rows:
        branch = str(row.branch)
        bmask = branches.get(branch)
        if (bmask is None):
            continue
        rec = compute_narrowing_reliability_for_branch(
            branch, bmask, row, calc_mask, spacing
        )
        rec["case_id"] = str(case_id)
        records.append(rec)
    if not records:
        return pd.DataFrame(
            columns = [
                "case_id",
                "branch",
                "candidate_reliability_score",
                "candidate_confidence_category",
            ]
        )
    cols = (["case_id", "branch"] + [
        c for c in records[0].keys() if (c not in ("case_id", "branch"))
    ])
    return pd.DataFrame(records)[cols]

def ensure_candidate_reliability_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "candidate_present": 0,
        "candidate_near_endpoint_flag": 0,
        "candidate_near_bifurcation_proxy_flag": 0,
        "short_branch_flag": 0,
        "short_lesion_flag": 0,
        "sharp_spike_flag": 0,
        "high_hu_overlap_flag": 0,
        "centerline_quality_penalty_flag": 0,
        "tortuosity_penalty_flag": 0,
        "area_agreement_flag": 0,
        "peak_width_mm": np.nan,
        "area_evidence_at_peak": np.nan,
        "high_hu_proximity_at_peak": np.nan,
        "diameter_flank_drop_at_peak": np.nan,
        "candidate_reliability_score": np.nan,
        "candidate_confidence_category": "not_available",
        "candidate_reliable_for_statistics": 0,
        "reliability_notes": "not_available",
    }
    for col, val in defaults.items():
        if (col not in out.columns):
            out[col] = val
    return out

def _primary_flag_reason_for_row(row: Any) -> Tuple[int, str]:
    reasons: List[str] = []
    branch = str(_row_get(row, "branch", "")).upper()
    candidate_present = storage.to_int(_row_get(row, "candidate_present", 0), 0)
    ratio = storage.to_float(
        _row_get(row, "geometric_narrowing_index", _row_get(row, "stenosis_ratio", 0.0)),
        0.0,
    )
    if ((candidate_present <= 0) and (ratio <= 0.0)):
        reasons.append("no_candidate")
    if (branch not in runtime.PRIMARY_STATISTICS_BRANCHES):
        reasons.append("non_primary_branch")
    branch_conf = str(
        _row_get(row, "branch_assignment_confidence", "not_available")
    ).lower()
    if (branch_conf != "high"):
        reasons.append("branch_assignment_not_high")
    cand_conf = str(_row_get(row, "candidate_confidence_category", "not_available")).lower()
    if (cand_conf != "high"):
        reasons.append("candidate_confidence_not_high")
    flag_map = [
        ("candidate_near_endpoint_flag", "near_endpoint"),
        ("candidate_near_bifurcation_proxy_flag", "possible_bifurcation_zone"),
        ("high_hu_overlap_flag", "high_hu_overlap"),
        ("sharp_spike_flag", "sharp_spike"),
        ("short_lesion_flag", "short_lesion"),
    ]
    for col, reason in flag_map:
        if (storage.to_int(_row_get(row, col, 0), 0) == 1):
            reasons.append(reason)
    reasons = sorted(set(reasons))
    if reasons:
        return (0, "|".join(reasons))
    return (1, "included_primary_statistics")

def _add_primary_statistics_flags_to_branch_df(df: pd.DataFrame) -> pd.DataFrame:
    if (df is None):
        return df
    out = df.copy()
    out = (
        ensure_candidate_reliability_columns(out)
        if callable(ensure_candidate_reliability_columns)
        else out
    )
    if ("candidate_reliable_for_broad_statistics" not in out.columns):
        out["candidate_reliable_for_broad_statistics"] = (
            pd.to_numeric(out.get("candidate_reliable_for_statistics", 0), errors = "coerce")
            .fillna(0)
            .astype(int)
        )
    flags: List[int] = []
    reasons: List[str] = []
    for _, r in out.iterrows():
        (f, reason) = _primary_flag_reason_for_row(r)
        flags.append(int(f))
        reasons.append(reason)
    out["candidate_reliable_for_primary_statistics"] = flags
    out["candidate_primary_exclusion_reason"] = reasons
    out["candidate_primary_statistics_rule"] = runtime.PRIMARY_STATISTICS_RULE_TEXT
    return out

def _primary_counts_from_branch_df(branch_df: pd.DataFrame) -> Dict[str, Any]:
    if ((branch_df is None) or branch_df.empty):
        return {
            "primary_reliable_narrowing_candidate_count": 0,
            "exploratory_narrowing_candidate_count": 0,
            "primary_excluded_narrowing_candidate_count": 0,
            "primary_excluded_narrowing_reasons": "",
            "primary_narrowing_statistics_rule": runtime.PRIMARY_STATISTICS_RULE_TEXT,
        }
    bdf = _add_primary_statistics_flags_to_branch_df(branch_df)
    cand = (
        pd.to_numeric(bdf.get("candidate_present", 0), errors = "coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    primary = (
        pd.to_numeric(
            bdf.get("candidate_reliable_for_primary_statistics", 0), errors = "coerce"
        )
        .fillna(0)
        .astype(int)
        == 1
    )
    excluded = (cand & ~primary)
    reason_counts: Dict[str, int] = {}
    if ("candidate_primary_exclusion_reason" in bdf.columns):
        for text in (
            bdf.loc[excluded, "candidate_primary_exclusion_reason"].astype(str).tolist()
        ):
            for part in text.split("|"):
                part = part.strip()
                if (part and (part != "included_primary_statistics")):
                    reason_counts[part] = (reason_counts.get(part, 0) + 1)
    return {
        "primary_reliable_narrowing_candidate_count": int((cand & primary).sum()),
        "exploratory_narrowing_candidate_count": int(cand.sum()),
        "primary_excluded_narrowing_candidate_count": int(excluded.sum()),
        "primary_excluded_narrowing_reasons": json.dumps(
            reason_counts, ensure_ascii = False, sort_keys = True
        ),
        "primary_narrowing_statistics_rule": runtime.PRIMARY_STATISTICS_RULE_TEXT,
    }

def _sync_primary_flags_into_narrowing_candidates(
    candidate_rows: List[Dict[str, Any]], branch_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    if not candidate_rows:
        return candidate_rows
    if ((branch_df is None) or branch_df.empty):
        for row in candidate_rows:
            row.setdefault("candidate_reliable_for_primary_statistics", 0)
            row.setdefault("candidate_primary_exclusion_reason", "branch_row_not_available")
            row.setdefault(
                "candidate_reliable_for_broad_statistics",
                row.get("candidate_reliable_for_statistics", 0),
            )
            row.setdefault(
                "candidate_primary_statistics_rule", runtime.PRIMARY_STATISTICS_RULE_TEXT
            )
        return candidate_rows
    bdf = _add_primary_statistics_flags_to_branch_df(branch_df)
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in bdf.iterrows():
        lookup[str(r.get("case_id", "")), str(r.get("branch", ""))] = r.to_dict()
    out_rows: List[Dict[str, Any]] = []
    for row in candidate_rows:
        key = (str(row.get("case_id", "")), str(row.get("branch", "")))
        src = lookup.get(key, {})
        row["candidate_reliable_for_broad_statistics"] = storage.to_int(
            src.get(
                "candidate_reliable_for_broad_statistics",
                row.get("candidate_reliable_for_statistics", 0),
            ),
            0,
        )
        row["candidate_reliable_for_primary_statistics"] = storage.to_int(
            src.get("candidate_reliable_for_primary_statistics", 0), 0
        )
        row["candidate_primary_exclusion_reason"] = str(
            src.get("candidate_primary_exclusion_reason", "branch_row_not_available")
        )
        row["candidate_primary_statistics_rule"] = runtime.PRIMARY_STATISTICS_RULE_TEXT
        out_rows.append(row)
    return out_rows

def _ensure_reliability_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_candidate_reliability_columns(df)
    if (out is None):
        return out
    if ("candidate_reliable_for_broad_statistics" not in out.columns):
        out["candidate_reliable_for_broad_statistics"] = (
            pd.to_numeric(out.get("candidate_reliable_for_statistics", 0), errors = "coerce")
            .fillna(0)
            .astype(int)
        )
    if ("candidate_reliable_for_primary_statistics" not in out.columns):
        out["candidate_reliable_for_primary_statistics"] = 0
    if ("candidate_primary_exclusion_reason" not in out.columns):
        out["candidate_primary_exclusion_reason"] = "not_evaluated"
    if ("candidate_primary_statistics_rule" not in out.columns):
        out["candidate_primary_statistics_rule"] = runtime.PRIMARY_STATISTICS_RULE_TEXT
    return out
