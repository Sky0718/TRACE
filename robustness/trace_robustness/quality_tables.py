from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from . import runtime, models, reliability, storage

def branch_presence_flags(branch_df: pd.DataFrame) -> Dict[str, int]:
    branches = set()
    if (((branch_df is not None) and (not branch_df.empty)) and ("branch" in branch_df.columns)):
        branches = set(branch_df["branch"].astype(str).tolist())
    return {f"{b}_detected": int((b in branches)) for b in ["LM", "LAD", "LCX", "RCA"]}

def branch_value_from_df(
    branch_df: pd.DataFrame, branch: str, column: str, default: float = 0.0
) -> float:
    if (
        (((branch_df is None)
        or branch_df.empty)
        or ("branch" not in branch_df.columns))
        or (column not in branch_df.columns)
    ):
        return float(default)
    sub = branch_df[(branch_df["branch"].astype(str) == str(branch))]
    if sub.empty:
        return float(default)
    return storage.to_float(sub.iloc[0][column], default = default)

def classify_centerline_quality(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if (text == "ok"):
        return "ok"
    if ("fallback" in text):
        return "fallback"
    if ((("too_small" in text) or ("failed" in text)) or ("empty" in text)):
        return "failed"
    return text

def clean_warning_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if (text.lower() in ("", "nan", "none", "null", "na", "n/a")):
        return ""
    return text

def compute_geometry_case_qa(
    case_id: str,
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    branch_info: Dict[str, Any],
    visual_outputs: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    raw_warning = clean_warning_value(case_row.get("warning", ""))
    if raw_warning:
        warnings.extend(
            [
                clean_warning_value(w)
                for w in raw_warning.split(";")
                if clean_warning_value(w)
            ]
        )
    for w in branch_info.get("warnings", []) if isinstance(branch_info, dict) else []:
        ww = clean_warning_value(w)
        if ww:
            warnings.append(ww)
    left_warning = ""
    if isinstance(branch_info, dict):
        left_warning = clean_warning_value(
            branch_info.get("left_split", {}).get("warning", "")
        )
    if left_warning:
        warnings.append(left_warning)
    warnings = sorted(set([w for w in warnings if clean_warning_value(w)]))
    branches_detected = []
    if (((branch_df is not None) and (not branch_df.empty)) and ("branch" in branch_df.columns)):
        branches_detected = sorted(set(branch_df["branch"].astype(str).tolist()))
    flags = branch_presence_flags(branch_df)
    missing_major = [b for b in ["LAD", "LCX", "RCA"] if (flags.get(f"{b}_detected", 0) == 0)]
    zero_length_branches: List[str] = []
    invalid_centerline_branches: List[str] = []
    fallback_centerline_branches: List[str] = []
    ok_centerline_count = 0
    fallback_centerline_count = 0
    failed_centerline_count = 0
    branch_count = 0
    valid_branch_count = 0
    if ((branch_df is not None) and (not branch_df.empty)):
        branch_count = int(len(branch_df))
        for _, r in branch_df.iterrows():
            b = str(r.get("branch", ""))
            length = storage.to_float(r.get("vessel_length_mm", 0.0), 0.0)
            if (length > 0):
                valid_branch_count += 1
            else:
                zero_length_branches.append(b)
            q = classify_centerline_quality(r.get("centerline_quality_flag", "unknown"))
            if (q == "ok"):
                ok_centerline_count += 1
            elif (q == "fallback"):
                fallback_centerline_count += 1
                fallback_centerline_branches.append(b)
            elif (q == "failed"):
                failed_centerline_count += 1
                invalid_centerline_branches.append(b)
    max_ratio = storage.to_float(
        case_row.get(
            "max_geometric_narrowing_index",
            case_row.get("max_stenosis_ratio", case_row.get("stenosis_ratio", 0.0)),
        ),
        0.0,
    )
    pos = storage.to_float(
        case_row.get("stenosis_position_norm", case_row.get("stenosis_position", -1.0)),
        -1.0,
    )
    invalid_stenosis_position = int(((max_ratio > 0.0) and (not (0.0 <= pos <= 1.0))))
    no_narrowing_candidate = int(((max_ratio <= 0.0) or (pos < 0.0)))
    expected_visual_outputs = dict(runtime.EXPECTED_VISUAL_OUTPUTS)
    if no_narrowing_candidate:
        expected_visual_outputs.pop("stenosis_marked_figure", None)
        expected_visual_outputs.pop("stenosis_marked_video", None)
    visual_missing: List[str] = []
    visual_existing = 0
    for key, rel in expected_visual_outputs.items():
        path_from_json = (
            visual_outputs.get(key) if isinstance(visual_outputs, dict) else None
        )
        candidate = Path(path_from_json) if path_from_json else (case_dir / rel)
        if candidate.exists():
            visual_existing += 1
        else:
            visual_missing.append(key)
    case_csv_exists = int(
        ((case_dir / "cta_concepts_case_level.csv").exists()
        or (case_dir / "cta_concepts_case_level.json").exists())
    )
    branch_csv_exists = int((case_dir / "cta_concepts_branch_level.csv").exists())
    visual_json_exists = int((case_dir / "visual_outputs.json").exists())
    branch_failure_flag = int(
        bool(((missing_major or zero_length_branches) or invalid_centerline_branches))
    )
    manual_review_needed = int(
        ((((branch_failure_flag
        or invalid_stenosis_position)
        or (failed_centerline_count > 0))
        or (len(warnings) > 0))
        or (len(visual_missing) > 0))
    )
    score = 100.0
    if not case_csv_exists:
        score -= 25.0
    if not branch_csv_exists:
        score -= 25.0
    score -= (8.0 * len(missing_major))
    score -= (7.0 * len(zero_length_branches))
    score -= (6.0 * failed_centerline_count)
    score -= (3.0 * fallback_centerline_count)
    score -= (6.0 * invalid_stenosis_position)
    score -= (2.0 * min(len(warnings), 8))
    score -= (1.0 * max(0, (len(expected_visual_outputs) - visual_existing)))
    score = float(np.clip(score, 0.0, 100.0))
    return {
        "case_id": str(case_id),
        "case_output_dir": str(case_dir),
        "algorithm_version": runtime.ALGORITHM_VERSION,
        "processing_success": int(
            (str(case_row.get("processing_status", "success")).lower() == "success")
        ),
        "case_concept_file_exists": int(case_csv_exists),
        "branch_concept_file_exists": int(branch_csv_exists),
        "visual_outputs_json_exists": int(visual_json_exists),
        "branch_count_from_table": int(branch_count),
        "valid_branch_count": int(valid_branch_count),
        "branches_detected": "|".join(branches_detected),
        **flags,
        "missing_major_branches": "|".join(missing_major),
        "zero_length_branches": "|".join(zero_length_branches),
        "branch_failure_flag": int(branch_failure_flag),
        "centerline_ok_count": int(ok_centerline_count),
        "centerline_fallback_count": int(fallback_centerline_count),
        "centerline_failed_count": int(failed_centerline_count),
        "centerline_fallback_branches": "|".join(fallback_centerline_branches),
        "centerline_failed_branches": "|".join(invalid_centerline_branches),
        "max_geometric_narrowing_index": float(max_ratio),
        "narrowing_position_norm": float(pos),
        "narrowing_position_valid": int(((0.0 <= pos <= 1.0) or (max_ratio <= 0.0))),
        "invalid_narrowing_position_flag": int(invalid_stenosis_position),
        "no_narrowing_candidate_flag": int(no_narrowing_candidate),
        "visual_expected_count": int(len(expected_visual_outputs)),
        "visual_existing_count": int(visual_existing),
        "missing_visual_outputs": "|".join(visual_missing),
        "warning_count": int(len(warnings)),
        "warnings": "|".join(warnings),
        "manual_review_needed": int(manual_review_needed),
        "qa_score": float(score),
        "runtime_seconds": float(elapsed_sec) if (elapsed_sec is not None) else np.nan,
    }

def build_standard_visual_index(
    case_id: str, case_dir: Path, visual_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case_id": str(case_id), "case_output_dir": str(case_dir)}
    existing_count = 0
    for key, rel in runtime.EXPECTED_VISUAL_OUTPUTS.items():
        raw = visual_outputs.get(key) if isinstance(visual_outputs, dict) else None
        p = Path(raw) if raw else (case_dir / rel)
        row[f"{key}_path"] = str(p)
        exists = int(p.exists())
        row[f"{key}_exists"] = exists
        if exists:
            existing_count += 1
    row["visual_existing_count"] = int(existing_count)
    row["visual_expected_count"] = int(len(runtime.EXPECTED_VISUAL_OUTPUTS))
    return row

def write_case_paperready_metadata(
    case_row: models.CaseConcept,
    branch_rows: List[models.BranchConcept],
    branch_info: Dict[str, Any],
    visual_outputs: Dict[str, Any],
    case_out: Path,
    elapsed_sec: float,
) -> Dict[str, Path]:
    branch_df = pd.DataFrame([asdict(r) for r in branch_rows])
    case_dict = asdict(case_row)
    qa_row = compute_case_qa(
        str(case_row.case_id),
        case_out,
        case_dict,
        branch_df,
        branch_info,
        visual_outputs,
        elapsed_sec = elapsed_sec,
    )
    visual_row = build_visual_index_row(str(case_row.case_id), case_out, visual_outputs)
    qa_path = (case_out / "cta_concepts_qa.csv")
    visual_index_path = (case_out / "visual_index.csv")
    pd.DataFrame([qa_row]).to_csv(qa_path, index = False, encoding = "utf-8-sig")
    pd.DataFrame([visual_row]).to_csv(visual_index_path, index = False, encoding = "utf-8-sig")
    storage.write_json((case_out / "cta_concepts_qa.json"), qa_row)
    return {"qa_csv": qa_path, "visual_index_csv": visual_index_path}

def build_visual_index_row(
    case_id: str, case_dir: Path, visual_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    row = build_standard_visual_index(case_id, case_dir, visual_outputs)
    for key, rel in runtime.OPTIONAL_PAPER_VISUAL_OUTPUTS.items():
        p = Path(str(visual_outputs.get(key, (case_dir / rel))))
        if not p.is_absolute():
            p = (case_dir / rel)
        row[f"{key}_path"] = str(p)
        row[f"{key}_exists"] = int(p.exists())
    row["paper_visual_existing_count"] = int(
        sum((int(row.get(f"{k}_exists", 0)) for k in runtime.OPTIONAL_PAPER_VISUAL_OUTPUTS))
    )
    row["paper_visual_available"] = int((row["paper_visual_existing_count"] > 0))
    return row

def compute_topology_case_qa(
    case_id: str,
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    branch_info: Dict[str, Any],
    visual_outputs: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    row = compute_geometry_case_qa(
        case_id,
        case_dir,
        case_row,
        branch_df,
        branch_info,
        visual_outputs,
        elapsed_sec = elapsed_sec,
    )
    tort_outliers: List[str] = []
    short_narrowing_risk: List[str] = []
    if ((branch_df is not None) and (not branch_df.empty)):
        for _, r in branch_df.iterrows():
            b = str(r.get("branch", ""))
            tort = storage.to_float(r.get("tortuosity", 0.0), 0.0)
            length = storage.to_float(r.get("vessel_length_mm", 0.0), 0.0)
            ratio = storage.to_float(
                r.get("stenosis_ratio", r.get("geometric_narrowing_index", 0.0)), 0.0
            )
            if (tort > runtime.TORTUOSITY_OUTLIER_THRESHOLD):
                tort_outliers.append(b)
            if (
                ((ratio > 0.0)
                and (length > 0.0))
                and (length < runtime.SHORT_BRANCH_NARROWING_LENGTH_MM)
            ):
                short_narrowing_risk.append(b)
    row["tortuosity_outlier_flag"] = int((len(tort_outliers) > 0))
    row["tortuosity_outlier_branches"] = "|".join(sorted(set(tort_outliers)))
    row["short_branch_narrowing_risk_flag"] = int((len(short_narrowing_risk) > 0))
    row["short_branch_narrowing_risk_branches"] = "|".join(
        sorted(set(short_narrowing_risk))
    )
    if (row["tortuosity_outlier_flag"] or row["short_branch_narrowing_risk_flag"]):
        row["manual_review_needed"] = 1
        row["qa_score"] = max(
            0.0,
            ((storage.to_float(row.get("qa_score", 100.0), 100.0)
            - (3.0 * row["tortuosity_outlier_flag"]))
            - (2.0 * row["short_branch_narrowing_risk_flag"])),
        )
    return row

def compute_reliability_case_qa(
    case_id: str,
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    branch_info: Dict[str, Any],
    visual_outputs: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    row = compute_topology_case_qa(
        case_id,
        case_dir,
        case_row,
        branch_df,
        branch_info,
        visual_outputs,
        elapsed_sec = elapsed_sec,
    )
    bdf = reliability._ensure_reliability_columns(
        branch_df if (branch_df is not None) else pd.DataFrame()
    )
    if not bdf.empty:
        candidate = (
            pd.to_numeric(bdf.get("candidate_present", 0), errors = "coerce")
            .fillna(0)
            .astype(int)
            == 1
        )
        conf = (
            bdf.get("candidate_confidence_category", "not_available").astype(str)
            if ("candidate_confidence_category" in bdf.columns)
            else pd.Series([], dtype = str)
        )
        high = int((candidate & (conf == "high")).sum())
        moderate = int((candidate & (conf == "moderate")).sum())
        low = int((candidate & (conf == "low_review")).sum())
        low_branches = (
            sorted(
                set(
                    bdf.loc[(candidate & (conf == "low_review")), "branch"]
                    .astype(str)
                    .tolist()
                )
            )
            if ("branch" in bdf.columns)
            else []
        )
        spike_branches = (
            sorted(
                set(
                    bdf.loc[
                        (pd.to_numeric(bdf.get("sharp_spike_flag", 0), errors = "coerce")
                        .fillna(0)
                        .astype(int)
                        == 1),
                        "branch",
                    ]
                    .astype(str)
                    .tolist()
                )
            )
            if ("branch" in bdf.columns)
            else []
        )
        endpoint_branches = (
            sorted(
                set(
                    bdf.loc[
                        (pd.to_numeric(
                            bdf.get("candidate_near_endpoint_flag", 0), errors = "coerce"
                        )
                        .fillna(0)
                        .astype(int)
                        == 1),
                        "branch",
                    ]
                    .astype(str)
                    .tolist()
                )
            )
            if ("branch" in bdf.columns)
            else []
        )
        row["narrowing_candidate_count"] = int(candidate.sum())
        row["high_confidence_narrowing_count"] = high
        row["moderate_confidence_narrowing_count"] = moderate
        row["low_confidence_narrowing_count"] = low
        row["low_confidence_narrowing_branches"] = "|".join(low_branches)
        row["sharp_spike_narrowing_branches"] = "|".join(spike_branches)
        row["endpoint_narrowing_branches"] = "|".join(endpoint_branches)
        row["narrowing_review_needed_flag"] = int(((low > 0) or (len(spike_branches) > 0)))
        if row["narrowing_review_needed_flag"]:
            row["manual_review_needed"] = 1
            row["qa_score"] = max(
                0.0, (storage.to_float(row.get("qa_score", 100.0), 100.0) - 2.0)
            )
    else:
        row["narrowing_candidate_count"] = 0
        row["high_confidence_narrowing_count"] = 0
        row["moderate_confidence_narrowing_count"] = 0
        row["low_confidence_narrowing_count"] = 0
        row["low_confidence_narrowing_branches"] = ""
        row["sharp_spike_narrowing_branches"] = ""
        row["endpoint_narrowing_branches"] = ""
        row["narrowing_review_needed_flag"] = 0
    return row

def enrich_qa_provenance(case_out: Path, meta: Dict[str, Any]) -> None:
    qpath = (case_out / "cta_concepts_qa.csv")
    df = storage.read_csv_safe(qpath)
    if df.empty:
        return
    for k in ["algorithm_version", "run_datetime_utc", "parameter_hash", "script_filename"]:
        df[k] = meta.get(k, "")
    df.to_csv(qpath, index = False, encoding = "utf-8-sig")
    storage.write_json((case_out / "cta_concepts_qa.json"), df.iloc[0].to_dict())

def enrich_qa_with_metadata(case_out: Path, meta: Dict[str, Any]) -> None:
    enrich_qa_provenance(case_out, meta)
    qpath = (Path(case_out) / "cta_concepts_qa.csv")
    bpath = (Path(case_out) / "cta_concepts_branch_level.csv")
    qdf = storage.read_csv_safe(qpath)
    bdf = storage.read_csv_safe(bpath)
    if qdf.empty:
        return
    counts = reliability._primary_counts_from_branch_df(bdf)
    for k, v in counts.items():
        qdf[k] = v
    qdf["algorithm_version"] = runtime.ALGORITHM_VERSION
    qdf.to_csv(qpath, index = False, encoding = "utf-8-sig")
    try:
        storage.write_json((Path(case_out) / "cta_concepts_qa.json"), qdf.iloc[0].to_dict())
    except Exception:
        pass

def compute_case_qa(
    case_id: str,
    case_dir: Path,
    case_row: Dict[str, Any],
    branch_df: pd.DataFrame,
    branch_info: Dict[str, Any],
    visual_outputs: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    row = compute_reliability_case_qa(
        case_id,
        case_dir,
        case_row,
        branch_df,
        branch_info,
        visual_outputs,
        elapsed_sec = elapsed_sec,
    )
    counts = reliability._primary_counts_from_branch_df(branch_df)
    row.update(counts)
    return row
