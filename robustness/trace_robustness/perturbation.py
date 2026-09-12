from __future__ import annotations
from typing import Any
from typing import Dict
from concurrent.futures import FIRST_COMPLETED
from typing import List
from typing import Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gc
import numpy as np
import os
import pandas as pd
from datetime import timezone
import traceback
from concurrent.futures import wait
from . import (
    runtime,
    dictionaries,
    perturbation_features,
    population_reports,
    reliability,
    storage,
)

def _robustness_one_case_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(payload.get("case_id", ""))
    lbl_path = Path(str(payload.get("source_label", "")))
    img_path = Path(str(payload.get("source_img", "")))
    min_branch_voxels = int(
        payload.get("min_branch_voxels", runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS)
    )
    stenosis_threshold = float(
        payload.get("stenosis_threshold", runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD)
    )
    threshold_hu = float(payload.get("threshold_hu", runtime.DEFAULT_CALCIUM_THRESHOLD_HU))
    min_calc_vol_mm3 = float(
        payload.get("min_calc_vol_mm3", runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3)
    )
    case_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    delta_case_rows: List[Dict[str, Any]] = []
    delta_branch_rows: List[Dict[str, Any]] = []
    instability_rows: List[Dict[str, Any]] = []
    label_vol = None
    image = None
    try:
        if not lbl_path.exists():
            raise FileNotFoundError(f"label not found: {lbl_path}")
        (label_vol, spacing) = storage.load_nifti(lbl_path)
        base_mask = (label_vol > 0)
        if img_path.exists():
            try:
                (image, img_spacing) = storage.load_nifti(img_path)
                if (np.shape(image) != np.shape(base_mask)):
                    image = None
            except Exception:
                image = None
        from scipy.ndimage import binary_erosion, binary_dilation, binary_opening

        structure = np.ones((3, 3, 3), dtype = bool)
        variants = (
            ("original", base_mask),
            ("erode_1voxel", binary_erosion(base_mask, structure = structure, iterations = 1)),
            (
                "dilate_1voxel",
                binary_dilation(base_mask, structure = structure, iterations = 1),
            ),
            ("open_1voxel", binary_opening(base_mask, structure = structure, iterations = 1)),
        )
        for vname, mask in variants:
            if not np.any(mask):
                case_rows.append(
                    {
                        "case_id": cid,
                        "variant": vname,
                        "status": "empty_mask",
                        "algorithm_version": runtime.ALGORITHM_VERSION,
                        "robustness_analysis_version": runtime.ROBUSTNESS_ANALYSIS_VERSION,
                    }
                )
                continue
            (c, b, cand) = perturbation_features._robustness_variant_tables(
                cid,
                vname,
                image,
                mask.astype(bool),
                spacing,
                threshold_hu,
                min_calc_vol_mm3,
                min_branch_voxels,
                stenosis_threshold,
            )
            c["source_label"] = str(lbl_path)
            c["source_img"] = str(img_path) if img_path.exists() else ""
            c["spacing_x_mm"] = float(spacing[0])
            c["spacing_y_mm"] = float(spacing[1])
            c["spacing_z_mm"] = float(spacing[2])
            case_rows.append(c)
            branch_rows.extend(b)
            candidate_rows.extend(cand)
        base_case = next(
            (
                r
                for r in case_rows
                if ((r.get("variant") == "original") and (r.get("status") == "success"))
            ),
            None,
        )
        if (base_case is not None):
            case_metrics = [
                "mask_voxel_count",
                "mask_volume_mm3",
                "surface_voxel_count",
                "surface_volume_ratio",
                "branch_count",
                "valid_branch_count",
                "total_branch_length_mm",
                "mean_branch_length_mm",
                "median_branch_length_mm",
                "mean_branch_diameter_mm",
                "median_branch_diameter_mm",
                "mean_tortuosity",
                "max_tortuosity",
                "max_geometric_narrowing_index",
                "candidate_count",
                "broad_reliable_candidate_count",
                "primary_reliable_candidate_count",
                "primary_excluded_candidate_count",
                "calcium_volume_mm3",
                "calcium_burden_pct",
            ]
            for pert in [
                r
                for r in case_rows
                if ((r.get("variant") not in ("original",)) and (r.get("status") == "success"))
            ]:
                drec = perturbation_features._robustness_delta_record(
                    base_case, pert, ["case_id"], case_metrics
                )
                orig_cat = str(
                    base_case.get(
                        "max_geometric_narrowing_category",
                        reliability.stenosis_interval_label(0),
                    )
                )
                pert_cat = str(
                    pert.get(
                        "max_geometric_narrowing_category",
                        reliability.stenosis_interval_label(0),
                    )
                )
                drec.update(
                    {
                        "original_max_category": orig_cat,
                        "perturbed_max_category": pert_cat,
                        "max_category_changed": int((orig_cat != pert_cat)),
                        "branch_count_changed": int(
                            (storage.to_int(base_case.get("branch_count", 0), 0)
                            != storage.to_int(pert.get("branch_count", 0), 0))
                        ),
                        "global_length_unstable_5pct": int(
                            (storage.to_float(
                                drec.get("abs_relative_delta_total_branch_length_mm", 0), 0
                            )
                            > 0.05)
                        ),
                        "mean_tortuosity_unstable_10pct": int(
                            (storage.to_float(
                                drec.get("abs_relative_delta_mean_tortuosity", 0), 0
                            )
                            > 0.1)
                        ),
                        "max_narrowing_unstable_abs_0p05": int(
                            (storage.to_float(
                                drec.get("abs_delta_max_geometric_narrowing_index", 0), 0
                            )
                            > 0.05)
                        ),
                        "max_narrowing_unstable_abs_0p10": int(
                            (storage.to_float(
                                drec.get("abs_delta_max_geometric_narrowing_index", 0), 0
                            )
                            > 0.1)
                        ),
                        "primary_candidate_count_changed": int(
                            (storage.to_int(
                                base_case.get("primary_reliable_candidate_count", 0), 0
                            )
                            != storage.to_int(
                                pert.get("primary_reliable_candidate_count", 0), 0
                            ))
                        ),
                    }
                )
                instability_rows.append(
                    {
                        **drec,
                        "baseline_mask_volume_mm3": base_case.get(
                            "mask_volume_mm3", np.nan
                        ),
                        "baseline_surface_volume_ratio": base_case.get(
                            "surface_volume_ratio", np.nan
                        ),
                        "baseline_branch_count": base_case.get("branch_count", np.nan),
                        "baseline_total_branch_length_mm": base_case.get(
                            "total_branch_length_mm", np.nan
                        ),
                        "baseline_mean_tortuosity": base_case.get(
                            "mean_tortuosity", np.nan
                        ),
                        "baseline_max_tortuosity": base_case.get("max_tortuosity", np.nan),
                        "baseline_max_geometric_narrowing_index": base_case.get(
                            "max_geometric_narrowing_index", np.nan
                        ),
                        "baseline_primary_reliable_candidate_count": base_case.get(
                            "primary_reliable_candidate_count", np.nan
                        ),
                        "baseline_calcium_burden_pct": base_case.get(
                            "calcium_burden_pct", np.nan
                        ),
                        "any_high_instability_flag": int(
                            ((((drec.get("branch_count_changed", 0) == 1)
                            or (drec.get("max_category_changed", 0) == 1))
                            or (drec.get("max_narrowing_unstable_abs_0p10", 0) == 1))
                            or (drec.get("global_length_unstable_5pct", 0) == 1))
                        ),
                    }
                )
                delta_case_rows.append(drec)
        base_branch = {
            str(r.get("branch", "")): r
            for r in branch_rows
            if (r.get("variant") == "original")
        }
        branch_metrics = [
            "branch_voxels",
            "vessel_volume_mm3",
            "vessel_length_mm",
            "vessel_diameter_mean_mm",
            "vessel_diameter_median_mm",
            "vessel_diameter_min_mm",
            "vessel_diameter_max_mm",
            "tortuosity",
            "geometric_narrowing_index",
            "stenosis_position_norm",
            "lesion_length_mm",
            "candidate_reliability_score",
            "peak_width_mm",
            "area_evidence_at_peak",
            "high_hu_proximity_at_peak",
            "profile_length_mm",
            "ratio_auc_mm",
            "length_ratio_ge_25_mm",
            "length_ratio_ge_50_mm",
            "diameter_gradient_max_abs_mm_per_mm",
            "calcium_volume_mm3",
            "calcium_burden_pct",
        ]
        for r in [x for x in branch_rows if (x.get("variant") != "original")]:
            bname = str(r.get("branch", ""))
            base = base_branch.get(bname)
            if (base is None):
                rec = {
                    "case_id": cid,
                    "branch": bname,
                    "perturbation_variant": str(r.get("variant", "")),
                    "branch_missing_in_original": 1,
                    "branch_missing_in_perturbed": 0,
                }
            else:
                rec = perturbation_features._robustness_delta_record(
                    base, r, ["case_id", "branch"], branch_metrics
                )
                rec["branch_missing_in_original"] = 0
                rec["branch_missing_in_perturbed"] = 0
                orig_cat = str(
                    base.get(
                        "geometric_narrowing_category",
                        reliability.stenosis_interval_label(0),
                    )
                )
                pert_cat = str(
                    r.get(
                        "geometric_narrowing_category",
                        reliability.stenosis_interval_label(0),
                    )
                )
                rec["original_category"] = orig_cat
                rec["perturbed_category"] = pert_cat
                rec["category_changed"] = int((orig_cat != pert_cat))
                rec["candidate_primary_changed"] = int(
                    (storage.to_int(
                        base.get("candidate_reliable_for_primary_statistics", 0), 0
                    )
                    != storage.to_int(
                        r.get("candidate_reliable_for_primary_statistics", 0), 0
                    ))
                )
                rec["candidate_confidence_changed"] = int(
                    (str(base.get("candidate_confidence_category", ""))
                    != str(r.get("candidate_confidence_category", "")))
                )
            delta_branch_rows.append(rec)
        return {
            "case_id": cid,
            "case_rows": case_rows,
            "branch_rows": branch_rows,
            "candidate_rows": candidate_rows,
            "delta_case_rows": delta_case_rows,
            "delta_branch_rows": delta_branch_rows,
            "instability_rows": instability_rows,
            "error": "",
        }
    except Exception as exc:
        return {
            "case_id": cid,
            "case_rows": [
                {
                    "case_id": cid,
                    "variant": "all",
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "algorithm_version": runtime.ALGORITHM_VERSION,
                    "robustness_analysis_version": runtime.ROBUSTNESS_ANALYSIS_VERSION,
                }
            ],
            "branch_rows": [],
            "candidate_rows": [],
            "delta_case_rows": [],
            "delta_branch_rows": [],
            "instability_rows": [],
            "error": str(exc),
        }
    finally:
        try:
            del label_vol
        except Exception:
            pass
        try:
            del image
        except Exception:
            pass
        gc.collect()

def _robustness_paths(root: Path) -> Dict[str, Path]:
    return {
        "case": (root / runtime.ROBUSTNESS_CASE_CSV_NAME),
        "branch": (root / runtime.ROBUSTNESS_BRANCH_CSV_NAME),
        "candidate": (root / runtime.ROBUSTNESS_CANDIDATE_CSV_NAME),
        "delta_case": (root / runtime.ROBUSTNESS_DELTA_CASE_CSV_NAME),
        "delta_branch": (root / runtime.ROBUSTNESS_DELTA_BRANCH_CSV_NAME),
        "instability": (root / runtime.ROBUSTNESS_INSTABILITY_CSV_NAME),
        "summary": (root / "cta_master_robustness_summary.csv"),
        "transitions": (root / runtime.ROBUSTNESS_TRANSITION_CSV_NAME),
        "dictionary": (root / runtime.ROBUSTNESS_DICTIONARY_CSV_NAME),
        "run_log": (root / "cta_robustness_run_log.csv"),
    }

def _drop_case_rows(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if (((df is None) or df.empty) or ("case_id" not in df.columns)):
        return pd.DataFrame()
    return df[(df["case_id"].astype(str) != str(case_id))].copy()

def _write_df_sorted(df: pd.DataFrame, path: Path, variant_col: str = "variant") -> None:
    out = df.copy() if (df is not None) else pd.DataFrame()
    if (not out.empty and ("case_id" in out.columns)):
        out["_sort_num"] = (
            out["case_id"].astype(str).map(lambda x: storage.case_sort_value(str(x))[0])
        )
        out["_sort_str"] = out["case_id"].astype(str)
        sort_cols = ["_sort_num", "_sort_str"]
        if (variant_col in out.columns):
            variant_order = {
                name: idx for (idx, name) in enumerate(runtime.ROBUSTNESS_VARIANT_NAMES)
            }
            out["_variant_order"] = (
                out[variant_col].astype(str).map(lambda x: variant_order.get(x, 99))
            )
            sort_cols.append("_variant_order")
        if ("branch" in out.columns):
            sort_cols.append("branch")
        out.sort_values(sort_cols, inplace = True, kind = "mergesort")
        out.drop(
            columns = [
                c for c in ["_sort_num", "_sort_str", "_variant_order"] if (c in out.columns)
            ],
            inplace = True,
        )
    out.to_csv(path, index = False, encoding = "utf-8-sig")

def _write_robustness_all_outputs(
    root: Path,
    case_rows: List[Dict[str, Any]],
    branch_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    delta_case_rows: List[Dict[str, Any]],
    delta_branch_rows: List[Dict[str, Any]],
    instability_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    paths = _robustness_paths(root)
    case_df = pd.DataFrame(case_rows)
    branch_df = pd.DataFrame(branch_rows)
    cand_df = pd.DataFrame(candidate_rows)
    dc_df = pd.DataFrame(delta_case_rows)
    db_df = pd.DataFrame(delta_branch_rows)
    inst_df = pd.DataFrame(instability_rows)
    _write_df_sorted(case_df, paths["case"], variant_col = "variant")
    _write_df_sorted(branch_df, paths["branch"], variant_col = "variant")
    _write_df_sorted(cand_df, paths["candidate"], variant_col = "variant")
    _write_df_sorted(dc_df, paths["delta_case"], variant_col = "perturbation_variant")
    _write_df_sorted(db_df, paths["delta_branch"], variant_col = "perturbation_variant")
    _write_df_sorted(inst_df, paths["instability"], variant_col = "perturbation_variant")
    summary_rows: List[Dict[str, Any]] = []
    for table_name, df, group_col in [
        ("case", case_df, "variant"),
        ("branch", branch_df, "variant"),
        ("candidate", cand_df, "variant"),
        ("delta_case", dc_df, "perturbation_variant"),
        ("delta_branch", db_df, "perturbation_variant"),
    ]:
        if ((df is None) or df.empty):
            continue
        numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors = "coerce"))
        ]
        useful_cols = [
            c
            for c in numeric_cols
            if any(
                (
                    (s in c)
                    for s in [
                        "delta",
                        "relative",
                        "length",
                        "volume",
                        "tortuosity",
                        "narrowing",
                        "diameter",
                        "candidate",
                        "branch_count",
                        "surface",
                    ]
                )
            )
        ]
        if (group_col in df.columns):
            for group, sub in df.groupby(group_col, dropna = False):
                for col in useful_cols:
                    stats = population_reports._summary_stats(
                        pd.to_numeric(sub[col], errors = "coerce")
                    )
                    stats.update(
                        {"table": table_name, "group": str(group), "metric": str(col)}
                    )
                    summary_rows.append(stats)
        else:
            for col in useful_cols:
                stats = population_reports._summary_stats(
                    pd.to_numeric(df[col], errors = "coerce")
                )
                stats.update({"table": table_name, "group": "all", "metric": str(col)})
                summary_rows.append(stats)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(paths["summary"], index = False, encoding = "utf-8-sig")
    trans_rows: List[Dict[str, Any]] = []
    if (
        ((dc_df is not None)
        and (not dc_df.empty))
        and ("original_max_category" in dc_df.columns)
    ):
        for (variant, oc, pc), sub in dc_df.groupby(
            ["perturbation_variant", "original_max_category", "perturbed_max_category"],
            dropna = False,
        ):
            trans_rows.append(
                {
                    "level": "case_max",
                    "perturbation_variant": variant,
                    "original_category": oc,
                    "perturbed_category": pc,
                    "count": int(len(sub)),
                }
            )
    if (((db_df is not None) and (not db_df.empty)) and ("original_category" in db_df.columns)):
        for (variant, oc, pc), sub in db_df.groupby(
            ["perturbation_variant", "original_category", "perturbed_category"],
            dropna = False,
        ):
            trans_rows.append(
                {
                    "level": "branch",
                    "perturbation_variant": variant,
                    "original_category": oc,
                    "perturbed_category": pc,
                    "count": int(len(sub)),
                }
            )
    pd.DataFrame(trans_rows).to_csv(paths["transitions"], index = False, encoding = "utf-8-sig")
    dictionaries._robustness_data_dictionary().to_csv(
        paths["dictionary"], index = False, encoding = "utf-8-sig"
    )
    outputs = {k: str(v) for (k, v) in paths.items() if v.exists()}
    return outputs

def _robustness_completed_case_ids(existing: pd.DataFrame) -> set:
    completed: set = set()
    if (
        (((existing is None)
        or existing.empty)
        or ("case_id" not in existing.columns))
        or ("variant" not in existing.columns)
    ):
        return completed
    tmp = existing.copy()
    tmp["case_id"] = tmp["case_id"].astype(str)
    for cid, sub in tmp.groupby("case_id"):
        successful = set(
            sub.loc[(sub.get("status", "") == "success"), "variant"].astype(str).tolist()
        )
        if set(runtime.ROBUSTNESS_VARIANT_NAMES).issubset(successful):
            completed.add(str(cid))
    return completed

def run_robustness_analysis(
    cases_root: Path,
    db_root: Optional[Path] = None,
    sample_n: int = 100,
    seed: int = 2026,
    threshold_hu: float = runtime.DEFAULT_CALCIUM_THRESHOLD_HU,
    min_calc_vol_mm3: float = runtime.DEFAULT_MIN_CALCIUM_VOLUME_MM3,
    min_branch_voxels: int = runtime.DEFAULT_MIN_BRANCH_COMPONENT_VOXELS,
    stenosis_threshold: float = runtime.DEFAULT_STENOSIS_REPORT_THRESHOLD,
    workers: int = runtime.DEFAULT_PARALLEL_WORKERS,
    robustness_case_start: Optional[int] = None,
    robustness_case_end: Optional[int] = None,
    robustness_overwrite: bool = False,
) -> Dict[str, str]:
    if (db_root is None):
        db_root = storage.infer_database_root_from_cases_root(Path(cases_root))
    root = storage.ensure_dir(Path(db_root))
    case_df = storage.read_csv_safe((root / runtime.MASTER_CASE_LEVEL_CSV_NAME))
    if case_df.empty:
        return {}
    if ("source_label" not in case_df.columns):
        return {}
    sample_n = int(sample_n)
    if (sample_n <= 0):
        return {}
    workers = max(1, int((workers or 1)))
    if ((os.name == "nt") and (workers > 60)):
        workers = 60
    rng = np.random.default_rng(int(seed))
    df = case_df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["_case_numeric_id"] = df["case_id"].map(lambda x: storage.case_sort_value(str(x))[0])
    explicit_range = ((robustness_case_start is not None) or (robustness_case_end is not None))
    if explicit_range:
        start_num = (
            int(robustness_case_start)
            if (robustness_case_start is not None)
            else int(df["_case_numeric_id"].min())
        )
        end_num = (
            int(robustness_case_end)
            if (robustness_case_end is not None)
            else int(df["_case_numeric_id"].max())
        )
        if (start_num > end_num):
            (start_num, end_num) = (end_num, start_num)
        pool = df[
            ((df["_case_numeric_id"] >= start_num) & (df["_case_numeric_id"] <= end_num))
        ].copy()
    else:
        start_num = None
        end_num = None
        if (sample_n >= len(df)):
            pool = df.copy()
        else:
            if ("manual_review_needed" in df.columns):
                qa_clean = df[
                    (pd.to_numeric(df["manual_review_needed"], errors = "coerce")
                    .fillna(0)
                    .astype(int)
                    == 0)
                ]
                pool = qa_clean.copy() if (len(qa_clean) >= sample_n) else df.copy()
            else:
                pool = df.copy()
            if (len(pool) > sample_n):
                indices = rng.choice(pool.index.to_numpy(), size = sample_n, replace = False)
                pool = pool.loc[indices].copy()
    pool = pool.sort_values(
        "case_id", key = lambda s: s.map(lambda x: storage.case_sort_value(str(x))[0])
    ).copy()
    pool.drop(columns = ["_case_numeric_id"], inplace = True, errors = "ignore")
    paths = _robustness_paths(root)
    existing_case = storage.read_csv_safe(paths["case"])
    existing_branch = storage.read_csv_safe(paths["branch"])
    existing_cand = storage.read_csv_safe(paths["candidate"])
    existing_dc = storage.read_csv_safe(paths["delta_case"])
    existing_db = storage.read_csv_safe(paths["delta_branch"])
    existing_inst = storage.read_csv_safe(paths["instability"])
    selected_ids = set(pool["case_id"].astype(str).tolist()) if not pool.empty else set()

    def _drop_case_id_set(df_in: pd.DataFrame, ids: set) -> pd.DataFrame:
        if ((((df_in is None) or df_in.empty) or ("case_id" not in df_in.columns)) or (not ids)):
            return pd.DataFrame() if (df_in is None) else df_in
        out = df_in.copy()
        return out[~out["case_id"].astype(str).isin(ids)].copy()

    if (robustness_overwrite and selected_ids):
        existing_case = _drop_case_id_set(existing_case, selected_ids)
        existing_branch = _drop_case_id_set(existing_branch, selected_ids)
        existing_cand = _drop_case_id_set(existing_cand, selected_ids)
        existing_dc = _drop_case_id_set(existing_dc, selected_ids)
        existing_db = _drop_case_id_set(existing_db, selected_ids)
        existing_inst = _drop_case_id_set(existing_inst, selected_ids)
    completed_ids = _robustness_completed_case_ids(existing_case)
    pending_rows = pool[~pool["case_id"].isin(completed_ids)].copy()
    case_rows = (
        existing_case.to_dict("records")
        if ((existing_case is not None) and (not existing_case.empty))
        else []
    )
    branch_rows = (
        existing_branch.to_dict("records")
        if ((existing_branch is not None) and (not existing_branch.empty))
        else []
    )
    cand_rows = (
        existing_cand.to_dict("records")
        if ((existing_cand is not None) and (not existing_cand.empty))
        else []
    )
    dc_rows = (
        existing_dc.to_dict("records")
        if ((existing_dc is not None) and (not existing_dc.empty))
        else []
    )
    db_rows = (
        existing_db.to_dict("records")
        if ((existing_db is not None) and (not existing_db.empty))
        else []
    )
    inst_rows = (
        existing_inst.to_dict("records")
        if ((existing_inst is not None) and (not existing_inst.empty))
        else []
    )
    if pending_rows.empty:
        outputs = _write_robustness_all_outputs(
            root, case_rows, branch_rows, cand_rows, dc_rows, db_rows, inst_rows
        )
        return outputs
    payloads: List[Dict[str, Any]] = []
    for _, row in pending_rows.iterrows():
        payloads.append(
            {
                "case_id": str(row.get("case_id", "")),
                "source_label": str(row.get("source_label", "")),
                "source_img": str(row.get("source_img", "")),
                "min_branch_voxels": int(min_branch_voxels),
                "stenosis_threshold": float(stenosis_threshold),
                "threshold_hu": float(threshold_hu),
                "min_calc_vol_mm3": float(min_calc_vol_mm3),
            }
        )
    run_log_df = (
        storage.read_csv_safe(paths["run_log"])
        if paths["run_log"].exists()
        else pd.DataFrame()
    )
    if (
        ((robustness_overwrite
        and selected_ids)
        and (not run_log_df.empty))
        and ("case_id" in run_log_df.columns)
    ):
        run_log_df = run_log_df[
            ~run_log_df["case_id"].astype(str).isin(selected_ids)
        ].copy()
    run_log_rows = (
        run_log_df.to_dict("records")
        if ((run_log_df is not None) and (not run_log_df.empty))
        else []
    )

    def accept_result(result: Dict[str, Any]) -> None:
        nonlocal case_rows, branch_rows, cand_rows, dc_rows, db_rows, inst_rows
        cid = str(result.get("case_id", ""))
        case_rows = _drop_case_rows(pd.DataFrame(case_rows), cid).to_dict("records")
        branch_rows = _drop_case_rows(pd.DataFrame(branch_rows), cid).to_dict("records")
        cand_rows = _drop_case_rows(pd.DataFrame(cand_rows), cid).to_dict("records")
        dc_rows = _drop_case_rows(pd.DataFrame(dc_rows), cid).to_dict("records")
        db_rows = _drop_case_rows(pd.DataFrame(db_rows), cid).to_dict("records")
        inst_rows = _drop_case_rows(pd.DataFrame(inst_rows), cid).to_dict("records")
        case_rows.extend((result.get("case_rows", []) or []))
        branch_rows.extend((result.get("branch_rows", []) or []))
        cand_rows.extend((result.get("candidate_rows", []) or []))
        dc_rows.extend((result.get("delta_case_rows", []) or []))
        db_rows.extend((result.get("delta_branch_rows", []) or []))
        inst_rows.extend((result.get("instability_rows", []) or []))
        _write_robustness_all_outputs(
            root, case_rows, branch_rows, cand_rows, dc_rows, db_rows, inst_rows
        )
        error_text = str((result.get("error", "") or ""))
        run_log_rows.append(
            {
                "case_id": cid,
                "status": "error" if error_text else "success",
                "error": error_text,
                "completed_datetime_utc": datetime.now(timezone.utc).isoformat(
                    timespec = "seconds"
                ),
                "robustness_analysis_version": runtime.ROBUSTNESS_ANALYSIS_VERSION,
            }
        )
        pd.DataFrame(run_log_rows).to_csv(
            paths["run_log"], index = False, encoding = "utf-8-sig"
        )

    if (workers == 1):
        for payload in payloads:
            accept_result(_robustness_one_case_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers = workers) as executor:
            future_map = {
                executor.submit(_robustness_one_case_worker, payload): payload
                for payload in payloads
            }
            pending = set(future_map)
            while pending:
                (done, pending) = wait(pending, timeout = 1.0, return_when = FIRST_COMPLETED)
                for future in done:
                    payload = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        cid = str(payload.get("case_id", ""))
                        result = {
                            "case_id": cid,
                            "case_rows": [
                                {
                                    "case_id": cid,
                                    "variant": "all",
                                    "status": "error",
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                    "algorithm_version": runtime.ALGORITHM_VERSION,
                                    "robustness_analysis_version": runtime.ROBUSTNESS_ANALYSIS_VERSION,
                                }
                            ],
                            "branch_rows": [],
                            "candidate_rows": [],
                            "delta_case_rows": [],
                            "delta_branch_rows": [],
                            "instability_rows": [],
                            "error": str(exc),
                        }
                    accept_result(result)
    outputs = _write_robustness_all_outputs(
        root, case_rows, branch_rows, cand_rows, dc_rows, db_rows, inst_rows
    )
    return outputs
