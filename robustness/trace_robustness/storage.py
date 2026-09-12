from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from typing import Tuple
import gc
import json
import numpy as np
import os
import pandas as pd
import re
import sqlite3
from . import (
    runtime,
    models,
    branch_tables,
    case_tables,
    dictionaries,
    extraction,
    population_reports,
    quality_tables,
)

def clean_user_path(text: str) -> str:
    value = str(text).strip()
    if value.startswith("&"):
        value = value[1:].strip()
    if (((len(value) >= 2) and (value[0] == value[-1])) and (value[0] in ("'", '"'))):
        value = value[1:-1].strip()
    if (((len(value) >= 2) and (value[0] in ("'", '"'))) and (value[-1] not in ("'", '"'))):
        value = value[1:].strip()
    if (((len(value) >= 2) and (value[-1] in ("'", '"'))) and (value[0] not in ("'", '"'))):
        value = value[:-1].strip()
    return os.path.expandvars(os.path.expanduser(value))

def sanitize_case_id(text: str) -> str:
    keep = []
    for ch in str(text):
        if (ch.isalnum() or (ch in ("-", "_"))):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out if out else "case"

def ensure_dir(path: (Path | str)) -> Path:
    p = Path(path)
    p.mkdir(parents = True, exist_ok = True)
    return p

def resolve_runtime_path(path: (Path | str), base_dir: Path = runtime.SCRIPT_DIR) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p)

def extract_number(p: Path) -> int:
    nums = re.findall("\\d+", p.name)
    return int(nums[0]) if nums else 0

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii = False, indent = 2), encoding = "utf-8"
    )

def case_sort_value(case_id: str) -> Tuple[int, str]:
    nums = re.findall("\\d+", str(case_id))
    if nums:
        return (int(nums[0]), str(case_id))
    return ((10**12), str(case_id))

def update_concept_database(
    case_row: models.CaseConcept, db_root: Path = runtime.DATABASE_ROOT
) -> Dict[str, str]:
    db_root = ensure_dir(db_root)
    legacy_csv = (db_root / "cta_concept_database.csv")
    legacy_xlsx = (db_root / "cta_concept_database.xlsx")
    legacy_sqlite = (db_root / "cta_concept_database.sqlite")
    new_row = database_row_from_case(case_row)
    if legacy_csv.exists():
        try:
            df = pd.read_csv(legacy_csv)
        except Exception:
            df = pd.DataFrame(columns = runtime.DATABASE_COLUMNS)
    else:
        df = pd.DataFrame(columns = runtime.DATABASE_COLUMNS)
    for col in runtime.DATABASE_COLUMNS:
        if (col not in df.columns):
            df[col] = np.nan
    df = df[runtime.DATABASE_COLUMNS].copy()
    if not df.empty:
        df = df[(df["case_id"].astype(str) != str(new_row["case_id"]))]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
    df["_sort_num"] = df["case_id"].apply(lambda x: case_sort_value(str(x))[0])
    df["_sort_str"] = df["case_id"].astype(str)
    df = df.sort_values(["_sort_num", "_sort_str"], kind = "mergesort").drop(
        columns = ["_sort_num", "_sort_str"]
    )
    for col in runtime.DATABASE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors = "coerce")
    df.to_csv(legacy_csv, index = False, encoding = "utf-8-sig")
    try:
        with pd.ExcelWriter(legacy_xlsx, engine = "openpyxl") as writer:
            df.to_excel(writer, sheet_name = "cta_concept_database", index = False)
    except Exception:
        pass
    try:
        with sqlite3.connect(legacy_sqlite) as conn:
            df.to_sql("cta_concepts", conn, if_exists = "replace", index = False)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_cta_concepts_case_id ON cta_concepts(case_id)"
            )
    except Exception:
        pass
    return {"csv": str(legacy_csv), "xlsx": str(legacy_xlsx), "sqlite": str(legacy_sqlite)}

def master_output_paths(db_root: Path = runtime.DATABASE_ROOT) -> Dict[str, Path]:
    root = ensure_dir(db_root)
    return {
        "case_level_csv": (root / runtime.MASTER_CASE_LEVEL_CSV_NAME),
        "branch_level_csv": (root / runtime.MASTER_BRANCH_LEVEL_CSV_NAME),
        "qa_csv": (root / runtime.MASTER_QA_CSV_NAME),
        "visual_index_csv": (root / runtime.MASTER_VISUAL_INDEX_CSV_NAME),
        "stenosis_csv": (root / runtime.MASTER_STENOSIS_CSV_NAME),
        "dataset_summary_csv": (root / runtime.MASTER_DATASET_SUMMARY_CSV_NAME),
        "data_dictionary_csv": (root / runtime.MASTER_DATA_DICTIONARY_CSV_NAME),
        "manifest_csv": (root / runtime.MASTER_MANIFEST_CSV_NAME),
        "xlsx": (root / runtime.MASTER_XLSX_NAME),
        "sqlite": (root / runtime.MASTER_SQLITE_NAME),
    }

def infer_database_root_from_cases_root(cases_root: Path) -> Path:
    cases_root = Path(cases_root)
    if (cases_root.name.lower() in ("cases", "case_outputs", "per_case")):
        return cases_root.parent
    return cases_root

def read_json_safe(path: Path, default: Any = None) -> Any:
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding = "utf-8"))
    except Exception:
        pass
    return default

def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if (value is None):
            return float(default)
        v = float(value)
        if (np.isnan(v) or np.isinf(v)):
            return float(default)
        return v
    except Exception:
        return float(default)

def to_int(value: Any, default: int = 0) -> int:
    try:
        if (value is None):
            return int(default)
        v = float(value)
        if (np.isnan(v) or np.isinf(v)):
            return int(default)
        return int(round(v))
    except Exception:
        return int(default)

def first_row_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if ((df is None) or df.empty):
        return {}
    return {str(k): json_safe(v) for (k, v) in df.iloc[0].to_dict().items()}

def case_output_directories(cases_root: Path) -> List[Path]:
    root = Path(cases_root)
    if not root.exists():
        return []
    dirs: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        expected = [
            (p / "cta_concepts_case_level.csv"),
            (p / "cta_concepts_case_level.json"),
            (p / "database_row.json"),
            (p / "cta_concepts_branch_level.csv"),
        ]
        if any((x.exists() for x in expected)):
            dirs.append(p)
    dirs.sort(key = lambda x: case_sort_value(x.name))
    return dirs

def load_case_level_from_dir(case_dir: Path) -> Dict[str, Any]:
    df = read_csv_safe((case_dir / "cta_concepts_case_level.csv"))
    row = first_row_dict(df)
    if not row:
        row = (read_json_safe((case_dir / "cta_concepts_case_level.json"), default = {}) or {})
    if not row:
        row = (read_json_safe((case_dir / "database_row.json"), default = {}) or {})
    if (("case_id" not in row) or (str(row.get("case_id", "")).strip() == "")):
        row["case_id"] = case_dir.name
    return row

def make_ablation_table_from_master(
    branch_df: pd.DataFrame, db_root: Path
) -> Optional[Path]:
    if ((branch_df is None) or branch_df.empty):
        return None
    rows: List[Dict[str, Any]] = []
    for _, r in branch_df.iterrows():
        case_id = str(r.get("case_id", ""))
        branch = str(r.get("branch", ""))
        gni = to_float(
            r.get("geometric_narrowing_index", r.get("stenosis_ratio", 0.0)), 0.0
        )
        area = to_float(r.get("area_evidence_at_peak", np.nan), np.nan)
        high_hu = to_float(r.get("high_hu_proximity_at_peak", np.nan), np.nan)
        rel_score = to_float(r.get("candidate_reliability_score", np.nan), np.nan)
        reliable = to_int(
            r.get(
                "candidate_reliable_for_statistics",
                r.get("narrowing_candidate_reliable", 0),
            ),
            0,
        )
        conf = str(r.get("candidate_confidence_category", ""))
        if pd.isna(area):
            area = 0.0
        if pd.isna(high_hu):
            high_hu = 0.0
        if pd.isna(rel_score):
            rel_score = 0.0
        variants = {
            "baseline_diameter_drop_only": gni,
            "diameter_plus_area_evidence": float(np.clip(((0.7 * gni) + (0.3 * area)), 0.0, 1.0)),
            "diameter_plus_highHU_penalty": float(
                np.clip((gni * (1.0 - (0.25 * high_hu))), 0.0, 1.0)
            ),
            "full_multievidence_score": gni,
            "reliability_filtered_score": gni if reliable else 0.0,
        }
        for variant, score in variants.items():
            rows.append(
                {
                    "case_id": case_id,
                    "branch": branch,
                    "ablation_variant": variant,
                    "geometric_narrowing_score": float(score),
                    "full_score": float(gni),
                    "area_evidence_at_peak": float(area),
                    "high_hu_proximity_at_peak": float(high_hu),
                    "candidate_reliability_score": float(rel_score),
                    "candidate_confidence_category": conf,
                    "candidate_reliable_for_statistics": int(reliable),
                    "algorithm_version": runtime.ALGORITHM_VERSION,
                    "note": "component-wise scoring ablation derived from stored multi-evidence candidate features",
                }
            )
    out = pd.DataFrame(rows)
    path = (Path(db_root) / "cta_master_ablation.csv")
    out.to_csv(path, index = False, encoding = "utf-8-sig")
    if not out.empty:
        summary = (
            out.groupby("ablation_variant")["geometric_narrowing_score"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        summary.to_csv(
            (Path(db_root) / "cta_master_ablation_summary.csv"),
            index = False,
            encoding = "utf-8-sig",
        )
    return path

def rebuild_raw_master_database(
    cases_root: Path = runtime.DATABASE_CASES_ROOT, db_root: Optional[Path] = None
) -> Dict[str, str]:
    cases_root = Path(cases_root)
    if (db_root is None):
        db_root = infer_database_root_from_cases_root(cases_root)
    paths = master_output_paths(db_root)
    case_dirs = case_output_directories(cases_root)
    case_master_rows: List[Dict[str, Any]] = []
    branch_master_frames: List[pd.DataFrame] = []
    qa_rows: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []
    stenosis_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for cdir in case_dirs:
        case_row = load_case_level_from_dir(cdir)
        cid = str(case_row.get("case_id", cdir.name))
        branch_df = read_csv_safe((cdir / "cta_concepts_branch_level.csv"))
        if branch_df.empty:
            branch_df = pd.DataFrame()
        branch_info = (read_json_safe((cdir / "branch_separation_info.json"), default = {}) or {})
        visual_outputs = (read_json_safe((cdir / "visual_outputs.json"), default = {}) or {})
        saved_qa = first_row_dict(read_csv_safe((cdir / "cta_concepts_qa.csv")))
        qa_row = quality_tables.compute_case_qa(
            cid,
            cdir,
            case_row,
            branch_df,
            branch_info,
            visual_outputs,
            elapsed_sec = saved_qa.get("runtime_seconds") if saved_qa else None,
        )
        visual_row = quality_tables.build_visual_index_row(cid, cdir, visual_outputs)
        master_case = case_tables.build_master_case_row(
            cdir, case_row, branch_df, qa_row, visual_row
        )
        case_master_rows.append(master_case)
        qa_rows.append(qa_row)
        visual_rows.append(visual_row)
        bmaster = branch_tables.build_branch_master(branch_df)
        if not bmaster.empty:
            branch_master_frames.append(bmaster)
            stenosis_rows.extend(branch_tables.build_stenosis_candidate_rows(bmaster))
        manifest_rows.append(
            {
                "case_id": cid,
                "case_output_dir": str(cdir),
                "has_case_level": int(
                    ((cdir / "cta_concepts_case_level.csv").exists()
                    or (cdir / "cta_concepts_case_level.json").exists())
                ),
                "has_branch_level": int((cdir / "cta_concepts_branch_level.csv").exists()),
                "has_branch_info_json": int(
                    (cdir / "branch_separation_info.json").exists()
                ),
                "has_visual_outputs_json": int((cdir / "visual_outputs.json").exists()),
                "qa_generated_in_master": 1,
            }
        )
    case_df = pd.DataFrame(case_master_rows)
    branch_df_all = (
        pd.concat(branch_master_frames, ignore_index = True)
        if branch_master_frames
        else pd.DataFrame()
    )
    qa_df = pd.DataFrame(qa_rows)
    visual_df = pd.DataFrame(visual_rows)
    sten_df = pd.DataFrame(stenosis_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    if case_df.empty:
        case_df = pd.DataFrame(columns = ["case_id", "case_output_dir", "algorithm_version"])
    if branch_df_all.empty:
        branch_df_all = pd.DataFrame(columns = ["case_id", "branch", "algorithm_version"])
    if qa_df.empty:
        qa_df = pd.DataFrame(columns = ["case_id", "qa_score", "manual_review_needed"])
    if visual_df.empty:
        visual_df = pd.DataFrame(
            columns = ["case_id", "visual_existing_count", "visual_expected_count"]
        )
    if sten_df.empty:
        sten_df = pd.DataFrame(
            columns = [
                "case_id",
                "branch",
                "geometric_narrowing_index",
                "geometric_narrowing_category",
            ]
        )
    if manifest_df.empty:
        manifest_df = pd.DataFrame(
            columns = ["case_id", "case_output_dir", "has_case_level", "has_branch_level"]
        )
    summary_df = case_tables.dataset_summary_rows(case_df, branch_df_all, qa_df, sten_df)
    dictionary_df = dictionaries.data_dictionary_rows()
    for df in [case_df, qa_df, visual_df, manifest_df, sten_df]:
        if (not df.empty and ("case_id" in df.columns)):
            df["_sort_num"] = df["case_id"].apply(lambda x: case_sort_value(str(x))[0])
            df["_sort_str"] = df["case_id"].astype(str)
            sort_cols = ["_sort_num", "_sort_str"]
            if ("branch" in df.columns):
                sort_cols.append("branch")
            df.sort_values(sort_cols, kind = "mergesort", inplace = True)
            df.drop(columns = ["_sort_num", "_sort_str"], inplace = True)
    if (not branch_df_all.empty and ("case_id" in branch_df_all.columns)):
        branch_df_all["_sort_num"] = branch_df_all["case_id"].apply(
            lambda x: case_sort_value(str(x))[0]
        )
        branch_df_all["_sort_str"] = branch_df_all["case_id"].astype(str)
        sort_cols = (["_sort_num", "_sort_str"] + (
            ["branch"] if ("branch" in branch_df_all.columns) else []
        ))
        branch_df_all.sort_values(sort_cols, kind = "mergesort", inplace = True)
        branch_df_all.drop(columns = ["_sort_num", "_sort_str"], inplace = True)
    case_df.to_csv(paths["case_level_csv"], index = False, encoding = "utf-8-sig")
    branch_df_all.to_csv(paths["branch_level_csv"], index = False, encoding = "utf-8-sig")
    qa_df.to_csv(paths["qa_csv"], index = False, encoding = "utf-8-sig")
    visual_df.to_csv(paths["visual_index_csv"], index = False, encoding = "utf-8-sig")
    sten_df.to_csv(paths["stenosis_csv"], index = False, encoding = "utf-8-sig")
    summary_df.to_csv(paths["dataset_summary_csv"], index = False, encoding = "utf-8-sig")
    dictionary_df.to_csv(paths["data_dictionary_csv"], index = False, encoding = "utf-8-sig")
    manifest_df.to_csv(paths["manifest_csv"], index = False, encoding = "utf-8-sig")
    try:
        with pd.ExcelWriter(paths["xlsx"], engine = "openpyxl") as writer:
            case_df.to_excel(writer, sheet_name = "case_level_master", index = False)
            branch_df_all.to_excel(writer, sheet_name = "branch_level_master", index = False)
            qa_df.to_excel(writer, sheet_name = "qa_master", index = False)
            visual_df.to_excel(writer, sheet_name = "visual_index", index = False)
            sten_df.to_excel(writer, sheet_name = "narrowing_candidates", index = False)
            summary_df.to_excel(writer, sheet_name = "dataset_summary", index = False)
            dictionary_df.to_excel(writer, sheet_name = "data_dictionary", index = False)
            manifest_df.to_excel(writer, sheet_name = "case_manifest", index = False)
            wb = writer.book
            for ws in wb.worksheets:
                ws.freeze_panes = "A2"
                for cell in ws[1]:
                    cell.font = cell.font.copy(bold = True)
                for col in ws.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col:
                        value = "" if (cell.value is None) else str(cell.value)
                        max_len = max(max_len, min(len(value), 65))
                    ws.column_dimensions[col_letter].width = max(10, min((max_len + 2), 42))
    except Exception:
        pass
    try:
        with sqlite3.connect(paths["sqlite"]) as conn:
            case_df.to_sql("case_level_master", conn, if_exists = "replace", index = False)
            branch_df_all.to_sql(
                "branch_level_master", conn, if_exists = "replace", index = False
            )
            qa_df.to_sql("qa_master", conn, if_exists = "replace", index = False)
            visual_df.to_sql("visual_index", conn, if_exists = "replace", index = False)
            sten_df.to_sql("narrowing_candidates", conn, if_exists = "replace", index = False)
            summary_df.to_sql("dataset_summary", conn, if_exists = "replace", index = False)
            dictionary_df.to_sql("data_dictionary", conn, if_exists = "replace", index = False)
            manifest_df.to_sql("case_manifest", conn, if_exists = "replace", index = False)
            if (not case_df.empty and ("case_id" in case_df.columns)):
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_case_level_master_case_id ON case_level_master(case_id)"
                )
            if (not qa_df.empty and ("case_id" in qa_df.columns)):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_qa_master_case_id ON qa_master(case_id)"
                )
            if (not branch_df_all.empty and ("case_id" in branch_df_all.columns)):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_branch_level_master_case_id ON branch_level_master(case_id)"
                )
    except Exception:
        pass
    return {key: str(value) for (key, value) in paths.items()}

def find_pairs(
    input_dir: Path, start: Optional[int] = None, end: Optional[int] = None
) -> List[Tuple[str, Path, Path]]:
    images = sorted(input_dir.glob("*.img.nii.gz"), key = extract_number)
    label_dict = {
        lb.name.replace(".label.nii.gz", ""): lb for lb in input_dir.glob("*.label.nii.gz")
    }
    pairs: List[Tuple[str, Path, Path]] = []
    for img in images:
        pid = img.name.replace(".img.nii.gz", "")
        if (pid in label_dict):
            pairs.append((pid, img, label_dict[pid]))
        else:
            pass
    if ((start is not None) or (end is not None)):
        s = max(0, (int(start) - 1)) if (start is not None) else 0
        e = min(len(pairs), int(end)) if (end is not None) else len(pairs)
        pairs = pairs[s:e]
    return pairs

def load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not runtime.NIBABEL_OK:
        raise RuntimeError("nibabel is required. Install with: pip install nibabel")
    img = runtime.nib.load(str(path))
    data = np.array(img.get_fdata(dtype = np.float32), copy = True)
    spacing = tuple((float(abs(v)) for v in img.header.get_zooms()[:3]))
    if hasattr(img, "uncache"):
        img.uncache()
    del img
    gc.collect()
    return (data, spacing)

def ensure_same_shape(image: np.ndarray, label: np.ndarray) -> None:
    if (image.shape != label.shape):
        raise ValueError(
            f"Image/label shape mismatch: image={image.shape}, label={label.shape}"
        )

def export_legacy_database_from_master(case_df: pd.DataFrame, db_root: Path) -> None:
    if ((case_df is None) or case_df.empty):
        return
    cols = [
        "case_id",
        "LAD_length",
        "RCA_length",
        "calcium_volume",
        "calcium_burden",
        "stenosis_ratio",
        "stenosis_position",
        "tortuosity",
    ]
    out = pd.DataFrame()
    out["case_id"] = case_df.get("case_id", "").astype(str)
    out["LAD_length"] = pd.to_numeric(
        case_df.get("LAD_length_mm", case_df.get("LAD_length", np.nan)), errors = "coerce"
    )
    out["RCA_length"] = pd.to_numeric(
        case_df.get("RCA_length_mm", case_df.get("RCA_length", np.nan)), errors = "coerce"
    )
    out["calcium_volume"] = pd.to_numeric(
        case_df.get(
            "high_hu_candidate_volume_mm3", case_df.get("calcium_volume_mm3", np.nan)
        ),
        errors = "coerce",
    )
    out["calcium_burden"] = pd.to_numeric(
        case_df.get(
            "high_hu_candidate_burden_pct", case_df.get("calcium_burden_pct", np.nan)
        ),
        errors = "coerce",
    )
    out["stenosis_ratio"] = pd.to_numeric(
        case_df.get(
            "geometric_narrowing_index",
            case_df.get("max_geometric_narrowing_index", np.nan),
        ),
        errors = "coerce",
    )
    out["stenosis_position"] = pd.to_numeric(
        case_df.get("stenosis_position_norm", np.nan), errors = "coerce"
    )
    out["tortuosity"] = pd.to_numeric(
        case_df.get("tortuosity_mean", np.nan), errors = "coerce"
    )
    out["_sort_num"] = out["case_id"].apply(lambda x: case_sort_value(str(x))[0])
    out["_sort_str"] = out["case_id"].astype(str)
    out = out.sort_values(["_sort_num", "_sort_str"]).drop(
        columns = ["_sort_num", "_sort_str"]
    )
    out[cols].to_csv(
        (Path(db_root) / "cta_concept_database_from_master.csv"),
        index = False,
        encoding = "utf-8-sig",
    )

def postprocess_master_outputs(db_root: Path) -> Dict[str, Any]:
    root = Path(db_root)
    case_df = read_csv_safe((root / runtime.MASTER_CASE_LEVEL_CSV_NAME))
    branch_df = read_csv_safe((root / runtime.MASTER_BRANCH_LEVEL_CSV_NAME))
    qa_df = read_csv_safe((root / runtime.MASTER_QA_CSV_NAME))
    visual_df = read_csv_safe((root / runtime.MASTER_VISUAL_INDEX_CSV_NAME))
    sten_df = read_csv_safe((root / runtime.MASTER_STENOSIS_CSV_NAME))
    outputs: Dict[str, Any] = {}
    population_reports.make_version_audit(case_df, root)
    outputs["version_audit"] = str((root / "cta_master_version_audit.csv"))
    outputs.update(
        {
            k: str(v)
            for (k, v) in population_reports.make_paper_summary_tables(
                case_df, branch_df, qa_df, visual_df, sten_df, root
            ).items()
        }
    )
    ablation_path = make_ablation_table_from_master(branch_df, root)
    if (ablation_path is not None):
        outputs["ablation"] = str(ablation_path)
        outputs["ablation_summary"] = str((root / "cta_master_ablation_summary.csv"))
    population_reports.choose_representative_cases(
        case_df, branch_df, qa_df, sten_df, visual_df, root
    )
    outputs["representative_cases"] = str((root / "cta_master_representative_cases.csv"))
    outputs.update(
        population_reports.create_master_population_figures(
            case_df, branch_df, qa_df, sten_df, root
        )
    )
    export_legacy_database_from_master(case_df, root)
    outputs["legacy_database_from_master"] = str(
        (root / "cta_concept_database_from_master.csv")
    )
    manifest = {
        "algorithm_version": runtime.ALGORITHM_VERSION,
        "postprocess_datetime_utc": extraction.now_utc_iso(),
        "case_count": int(len(case_df)) if (case_df is not None) else 0,
        "branch_row_count": int(len(branch_df)) if (branch_df is not None) else 0,
        "narrowing_candidate_count": int(len(sten_df)) if (sten_df is not None) else 0,
        "outputs": outputs,
    }
    write_json((root / "cta_master_paper_system_manifest.json"), manifest)
    outputs["paper_system_manifest"] = str((root / "cta_master_paper_system_manifest.json"))
    return outputs

def rebuild_master_database(
    cases_root: Path = runtime.DATABASE_CASES_ROOT, db_root: Optional[Path] = None
) -> Dict[str, str]:
    paths = rebuild_raw_master_database(cases_root, db_root = db_root)
    root = (
        Path(db_root)
        if (db_root is not None)
        else infer_database_root_from_cases_root(Path(cases_root))
    )
    extra = postprocess_master_outputs(root)
    paths.update({k: str(v) for (k, v) in extra.items()})
    return paths

def safe_case_max_narrowing(case_row: Any, default: float = 0.0) -> float:
    try:
        if isinstance(case_row, dict):
            for key in (
                "max_geometric_narrowing_index",
                "geometric_narrowing_index",
                "max_stenosis_ratio",
                "stenosis_ratio",
            ):
                if (key in case_row):
                    return to_float(case_row.get(key), default)
            return float(default)
        if hasattr(case_row, "max_geometric_narrowing_index"):
            return to_float(getattr(case_row, "max_geometric_narrowing_index"), default)
        if hasattr(case_row, "geometric_narrowing_index"):
            return to_float(getattr(case_row, "geometric_narrowing_index"), default)
        if hasattr(case_row, "max_stenosis_ratio"):
            return to_float(getattr(case_row, "max_stenosis_ratio"), default)
        if hasattr(case_row, "stenosis_ratio"):
            return to_float(getattr(case_row, "stenosis_ratio"), default)
    except Exception:
        pass
    return float(default)

def database_row_from_case(case_row: models.CaseConcept) -> Dict[str, Any]:
    return {
        "case_id": str(getattr(case_row, "case_id", "")),
        "LAD_length": float(getattr(case_row, "LAD_length_mm", 0.0)),
        "RCA_length": float(getattr(case_row, "RCA_length_mm", 0.0)),
        "calcium_volume": float(getattr(case_row, "calcium_volume_mm3", 0.0)),
        "calcium_burden": float(getattr(case_row, "calcium_burden_pct", 0.0)),
        "stenosis_ratio": float(safe_case_max_narrowing(case_row, 0.0)),
        "stenosis_position": float(getattr(case_row, "stenosis_position_norm", -1.0)),
        "tortuosity": float(getattr(case_row, "tortuosity_mean", 0.0)),
    }
