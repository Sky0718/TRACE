from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import runtime, reliability, storage

def make_version_audit(case_df: pd.DataFrame, db_root: Path) -> pd.DataFrame:
    if ((case_df is None) or case_df.empty):
        audit = pd.DataFrame([{"metric": "case_count", "value": 0, "note": "no cases"}])
    else:
        versions = (
            case_df.get("algorithm_version", pd.Series(([""] * len(case_df))))
            .astype(str)
            .replace("nan", "")
        )
        hashes = (
            case_df.get("parameter_hash", pd.Series(([""] * len(case_df))))
            .astype(str)
            .replace("nan", "")
        )
        version_counts = versions.value_counts(dropna = False).to_dict()
        hash_counts = hashes.value_counts(dropna = False).to_dict()
        audit_rows = [
            {"metric": "case_count", "value": int(len(case_df)), "note": ""},
            {
                "metric": "unique_algorithm_version_count",
                "value": int(len(version_counts)),
                "note": json.dumps(version_counts, ensure_ascii = False),
            },
            {
                "metric": "unique_parameter_hash_count",
                "value": int(len(hash_counts)),
                "note": json.dumps(hash_counts, ensure_ascii = False),
            },
            {
                "metric": "mixed_algorithm_version_flag",
                "value": int((len([v for v in version_counts if v]) > 1)),
                "note": "1 means master table mixes different algorithm versions",
            },
            {
                "metric": "mixed_parameter_hash_flag",
                "value": int((len([h for h in hash_counts if h]) > 1)),
                "note": "1 means master table mixes different parameter settings",
            },
        ]
        audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        (Path(db_root) / "cta_master_version_audit.csv"), index = False, encoding = "utf-8-sig"
    )
    return audit

def _summary_stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors = "coerce").dropna()
    if s.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof = 1)) if (s.size > 1) else 0.0,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def make_population_summary_tables(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    db_root: Path,
) -> Dict[str, Path]:
    root = Path(db_root)
    paths: Dict[str, Path] = {}
    case_metrics = [
        c
        for c in [
            "total_vessel_volume_mm3",
            "high_hu_candidate_volume_mm3",
            "high_hu_candidate_burden_pct",
            "geometric_narrowing_index",
            "tortuosity_mean",
            "tortuosity_max",
            "branch_count",
            "qa_score",
        ]
        if (c in case_df.columns)
    ]
    rows = []
    for c in case_metrics:
        d = _summary_stats(case_df[c])
        d.update({"metric": c})
        rows.append(d)
    t1 = pd.DataFrame(rows)
    paths["table1_case_summary"] = (root / "cta_master_table1_case_summary.csv")
    t1.to_csv(paths["table1_case_summary"], index = False, encoding = "utf-8-sig")
    rows = []
    if (not branch_df.empty and ("branch" in branch_df.columns)):
        for b, sub in branch_df.groupby("branch", dropna = False):
            for c in [
                "vessel_length_mm",
                "vessel_diameter_mean_mm",
                "vessel_diameter_min_mm",
                "tortuosity",
                "geometric_narrowing_index",
                "branch_assignment_score",
                "high_hu_candidate_burden_pct",
            ]:
                if (c in sub.columns):
                    d = _summary_stats(sub[c])
                    d.update({"branch": b, "metric": c})
                    rows.append(d)
            if ("branch_assignment_confidence" in sub.columns):
                counts = (
                    sub["branch_assignment_confidence"].astype(str).value_counts().to_dict()
                )
                rows.append(
                    {
                        "branch": b,
                        "metric": "branch_assignment_confidence_counts",
                        "n": int(len(sub)),
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "note": json.dumps(counts, ensure_ascii = False),
                    }
                )
    t2 = pd.DataFrame(rows)
    paths["table2_branch_summary"] = (root / "cta_master_table2_branch_summary.csv")
    t2.to_csv(paths["table2_branch_summary"], index = False, encoding = "utf-8-sig")
    rows = []
    for c in [
        "manual_review_needed",
        "branch_failure_flag",
        "tortuosity_outlier_flag",
        "short_branch_narrowing_risk_flag",
        "invalid_narrowing_position_flag",
        "contrast_lumen_contamination_risk_flag",
    ]:
        if (c in qa_df.columns):
            vals = pd.to_numeric(qa_df[c], errors = "coerce").fillna(0).astype(int)
            rows.append(
                {
                    "qc_flag": c,
                    "case_count": int(vals.sum()),
                    "case_pct": float((vals.mean() * 100.0)) if len(vals) else np.nan,
                }
            )
        elif (c in case_df.columns):
            vals = pd.to_numeric(case_df[c], errors = "coerce").fillna(0).astype(int)
            rows.append(
                {
                    "qc_flag": c,
                    "case_count": int(vals.sum()),
                    "case_pct": float((vals.mean() * 100.0)) if len(vals) else np.nan,
                }
            )
    t3 = pd.DataFrame(rows)
    paths["table3_qc_failure_summary"] = (root / "cta_master_table3_qc_failure_summary.csv")
    t3.to_csv(paths["table3_qc_failure_summary"], index = False, encoding = "utf-8-sig")
    rows = []
    if not sten_df.empty:
        if ("candidate_confidence_category" in sten_df.columns):
            for cat, sub in sten_df.groupby("candidate_confidence_category", dropna = False):
                rows.append(
                    {
                        "group": f"confidence={cat}",
                        "candidate_count": int(len(sub)),
                        **_summary_stats(
                            sub.get("geometric_narrowing_index", pd.Series(dtype = float))
                        ),
                    }
                )
        if ("branch" in sten_df.columns):
            for b, sub in sten_df.groupby("branch", dropna = False):
                rows.append(
                    {
                        "group": f"branch={b}",
                        "candidate_count": int(len(sub)),
                        **_summary_stats(
                            sub.get("geometric_narrowing_index", pd.Series(dtype = float))
                        ),
                    }
                )
    t4 = pd.DataFrame(rows)
    paths["table4_narrowing_candidate_summary"] = (
        root / "cta_master_table4_narrowing_candidate_summary.csv"
    )
    t4.to_csv(
        paths["table4_narrowing_candidate_summary"], index = False, encoding = "utf-8-sig"
    )
    rows = []
    if not visual_df.empty:
        for c in visual_df.columns:
            if (c.endswith("_exists") or c.endswith("_available")):
                vals = pd.to_numeric(visual_df[c], errors = "coerce").fillna(0).astype(int)
                rows.append(
                    {
                        "visual_field": c,
                        "available_count": int(vals.sum()),
                        "available_pct": (
                            float((vals.mean() * 100.0)) if len(vals) else np.nan
                        ),
                    }
                )
    t5 = pd.DataFrame(rows)
    paths["table5_visual_availability_summary"] = (
        root / "cta_master_table5_visual_availability_summary.csv"
    )
    t5.to_csv(
        paths["table5_visual_availability_summary"], index = False, encoding = "utf-8-sig"
    )
    return paths

def choose_reliable_examples(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    db_root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if ((case_df is None) or case_df.empty):
        out = pd.DataFrame(columns = ["role", "case_id", "reason", "score"])
        out.to_csv(
            (Path(db_root) / "cta_master_representative_cases.csv"),
            index = False,
            encoding = "utf-8-sig",
        )
        return out
    df = case_df.copy()
    df["case_id"] = df["case_id"].astype(str)

    def add(role: str, sub: pd.DataFrame, sort_col: str, ascending: bool, reason: str):
        if (((sub is None) or sub.empty) or (sort_col not in sub.columns)):
            return
        s = sub.copy()
        s[sort_col] = pd.to_numeric(s[sort_col], errors = "coerce")
        s = s.dropna(subset = [sort_col]).sort_values(sort_col, ascending = ascending)
        if s.empty:
            return
        r = s.iloc[0]
        rows.append(
            {
                "role": role,
                "case_id": str(r.get("case_id", "")),
                "reason": reason,
                "score": float(r.get(sort_col, np.nan)),
                "case_output_dir": str(r.get("case_output_dir", "")),
            }
        )

    qa_pass = (
        df[
            (pd.to_numeric(df.get("manual_review_needed", 0), errors = "coerce")
            .fillna(0)
            .astype(int)
            == 0)
        ]
        if ("manual_review_needed" in df.columns)
        else df
    )
    add(
        "best_overall_QA_case",
        qa_pass,
        "qa_score",
        False,
        "highest QA score among non-review cases",
    )
    add(
        "highest_reliable_narrowing_case",
        df,
        "geometric_narrowing_index",
        False,
        "largest case-level geometric narrowing index",
    )
    if ("geometric_narrowing_index" in df.columns):
        low = df[
            (pd.to_numeric(df["geometric_narrowing_index"], errors = "coerce").fillna(0)
            <= 0.05)
        ]
        add(
            "normal_low_narrowing_case",
            low if not low.empty else df,
            "qa_score",
            False,
            "low/no narrowing with good QA",
        )
    add("highest_tortuosity_case", df, "tortuosity_max", False, "largest tortuosity_max")
    if ("manual_review_needed" in df.columns):
        review = df[
            (pd.to_numeric(df["manual_review_needed"], errors = "coerce").fillna(0).astype(int)
            == 1)
        ]
        add(
            "review_needed_case",
            review,
            "qa_score",
            True,
            "lowest QA among review-needed cases",
        )
    if ("high_hu_candidate_burden_pct" in df.columns):
        add(
            "highest_high_HU_candidate_case",
            df,
            "high_hu_candidate_burden_pct",
            False,
            "largest peripheral high-HU candidate burden",
        )
    out = pd.DataFrame(rows).drop_duplicates(subset = ["role"], keep = "first")
    out.to_csv(
        (Path(db_root) / "cta_master_representative_cases.csv"),
        index = False,
        encoding = "utf-8-sig",
    )
    return out

def create_population_figures(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    db_root: Path,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if not runtime.MATPLOTLIB_OK:
        return outputs
    root = Path(db_root)
    try:
        metrics = [
            c
            for c in [
                "geometric_narrowing_index",
                "tortuosity_mean",
                "tortuosity_max",
                "high_hu_candidate_burden_pct",
            ]
            if (c in case_df.columns)
        ]
        if metrics:
            (fig, axes) = runtime.plt.subplots(
                1,
                len(metrics),
                figsize = ((5 * len(metrics)), 4),
                facecolor = runtime.visual_text_color(),
            )
            axes = np.asarray(axes).reshape(-1)
            for ax, c in zip(axes, metrics):
                s = pd.to_numeric(case_df[c], errors = "coerce").dropna()
                ax.hist(s, bins = 30)
                ax.set_title(c)
                ax.grid(alpha = 0.25)
            runtime.plt.tight_layout()
            path = (root / "figure_MASTER_population_summary.png")
            runtime.plt.savefig(path, dpi = 180, bbox_inches = "tight")
            runtime.plt.close(fig)
            outputs["population_summary"] = str(path)
        flags = []
        for c in [
            "manual_review_needed",
            "branch_failure_flag",
            "tortuosity_outlier_flag",
            "short_branch_narrowing_risk_flag",
        ]:
            if (c in qa_df.columns):
                flags.append(
                    (c, int(pd.to_numeric(qa_df[c], errors = "coerce").fillna(0).sum()))
                )
        if flags:
            (fig, ax) = runtime.plt.subplots(
                figsize = (9, 4), facecolor = runtime.visual_text_color()
            )
            ax.bar([x[0] for x in flags], [x[1] for x in flags])
            ax.set_ylabel("Case count")
            ax.set_title("QC / failure mode summary")
            ax.tick_params(axis = "x", rotation = 25)
            ax.grid(axis = "y", alpha = 0.25)
            runtime.plt.tight_layout()
            path = (root / "figure_MASTER_qc_summary.png")
            runtime.plt.savefig(path, dpi = 180, bbox_inches = "tight")
            runtime.plt.close(fig)
            outputs["qc_summary"] = str(path)
        if (not sten_df.empty and ("candidate_confidence_category" in sten_df.columns)):
            counts = sten_df["candidate_confidence_category"].astype(str).value_counts()
            (fig, ax) = runtime.plt.subplots(
                figsize = (6, 4), facecolor = runtime.visual_text_color()
            )
            ax.bar(counts.index.tolist(), counts.values.tolist())
            ax.set_ylabel("Candidate count")
            ax.set_title("Geometric narrowing candidate confidence")
            ax.grid(axis = "y", alpha = 0.25)
            runtime.plt.tight_layout()
            path = (root / "figure_MASTER_candidate_confidence.png")
            runtime.plt.savefig(path, dpi = 180, bbox_inches = "tight")
            runtime.plt.close(fig)
            outputs["candidate_confidence"] = str(path)
    except Exception:
        pass
    return outputs

def make_paper_summary_tables(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    db_root: Path,
) -> Dict[str, Path]:
    bdf = (
        reliability._add_primary_statistics_flags_to_branch_df(branch_df)
        if ((branch_df is not None) and (not branch_df.empty))
        else branch_df
    )
    if ((sten_df is not None) and (not sten_df.empty)):
        sten_rows = reliability._sync_primary_flags_into_narrowing_candidates(
            sten_df.to_dict("records"),
            bdf if isinstance(bdf, pd.DataFrame) else pd.DataFrame(),
        )
        sdf = pd.DataFrame(sten_rows)
    else:
        sdf = sten_df
    paths = make_population_summary_tables(
        case_df,
        bdf if isinstance(bdf, pd.DataFrame) else branch_df,
        qa_df,
        visual_df,
        sdf,
        db_root,
    )
    try:
        t4_path = Path(
            paths.get(
                "table4_narrowing_candidate_summary",
                (Path(db_root) / "cta_master_table4_narrowing_candidate_summary.csv"),
            )
        )
        t4 = storage.read_csv_safe(t4_path)
        rows = t4.to_dict("records") if ((t4 is not None) and (not t4.empty)) else []
        if ((sdf is not None) and (not sdf.empty)):
            if ("candidate_reliable_for_primary_statistics" in sdf.columns):
                for flag, sub in sdf.groupby(
                    "candidate_reliable_for_primary_statistics", dropna = False
                ):
                    rows.append(
                        {
                            "group": f"primary_statistics={int(storage.to_int(flag, 0))}",
                            "candidate_count": int(len(sub)),
                            **_summary_stats(
                                sub.get("geometric_narrowing_index", pd.Series(dtype = float))
                            ),
                        }
                    )
            if ("candidate_primary_exclusion_reason" in sdf.columns):
                excluded = sdf[
                    (pd.to_numeric(
                        sdf.get("candidate_reliable_for_primary_statistics", 0),
                        errors = "coerce",
                    )
                    .fillna(0)
                    .astype(int)
                    == 0)
                ]
                reason_counts: Dict[str, int] = {}
                for text in (
                    excluded["candidate_primary_exclusion_reason"].astype(str).tolist()
                ):
                    for part in text.split("|"):
                        part = part.strip()
                        if (part and (part != "included_primary_statistics")):
                            reason_counts[part] = (reason_counts.get(part, 0) + 1)
                rows.append(
                    {
                        "group": "primary_exclusion_reason_counts",
                        "candidate_count": int(len(excluded)),
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "note": json.dumps(
                            reason_counts, ensure_ascii = False, sort_keys = True
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(t4_path, index = False, encoding = "utf-8-sig")
    except Exception:
        pass
    return paths

def choose_representative_cases(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    db_root: Path,
) -> pd.DataFrame:
    out = choose_reliable_examples(case_df, branch_df, qa_df, sten_df, visual_df, db_root)
    try:
        if ((sten_df is not None) and (not sten_df.empty)):
            bdf = (
                reliability._add_primary_statistics_flags_to_branch_df(branch_df)
                if ((branch_df is not None) and (not branch_df.empty))
                else branch_df
            )
            sdf = pd.DataFrame(
                reliability._sync_primary_flags_into_narrowing_candidates(
                    sten_df.to_dict("records"),
                    bdf if isinstance(bdf, pd.DataFrame) else pd.DataFrame(),
                )
            )
            prim = sdf[
                (pd.to_numeric(
                    sdf.get("candidate_reliable_for_primary_statistics", 0), errors = "coerce"
                )
                .fillna(0)
                .astype(int)
                == 1)
            ]
            if (not prim.empty and ("geometric_narrowing_index" in prim.columns)):
                prim = prim.copy()
                prim["geometric_narrowing_index"] = pd.to_numeric(
                    prim["geometric_narrowing_index"], errors = "coerce"
                )
                r = prim.sort_values("geometric_narrowing_index", ascending = False).iloc[0]
                row = {
                    "role": "highest_primary_reliable_narrowing_case",
                    "case_id": str(r.get("case_id", "")),
                    "reason": "largest geometric narrowing index among primary-statistics candidates",
                    "score": float(r.get("geometric_narrowing_index", np.nan)),
                    "case_output_dir": "",
                }
                out = pd.concat(
                    [out, pd.DataFrame([row])], ignore_index = True
                ).drop_duplicates(subset = ["role"], keep = "last")
                out.to_csv(
                    (Path(db_root) / "cta_master_representative_cases.csv"),
                    index = False,
                    encoding = "utf-8-sig",
                )
    except Exception:
        pass
    return out

def create_master_population_figures(
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    sten_df: pd.DataFrame,
    db_root: Path,
) -> Dict[str, str]:
    outputs = create_population_figures(case_df, branch_df, qa_df, sten_df, db_root)
    if not runtime.MATPLOTLIB_OK:
        return outputs
    try:
        root = Path(db_root)
        bdf = (
            reliability._add_primary_statistics_flags_to_branch_df(branch_df)
            if ((branch_df is not None) and (not branch_df.empty))
            else branch_df
        )
        if ((bdf is not None) and (not bdf.empty)):
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
                == 1
            )
            labels = [
                "all candidates",
                "broad reliable",
                "primary reliable",
                "primary excluded",
            ]
            values = [
                int(cand.sum()),
                int((cand & broad).sum()),
                int((cand & primary).sum()),
                int((cand & ~primary).sum()),
            ]
            (fig, ax) = runtime.plt.subplots(
                figsize = (8, 4), facecolor = runtime.visual_text_color()
            )
            ax.bar(labels, values)
            ax.set_ylabel("Candidate count")
            ax.set_title("Primary vs exploratory geometric narrowing candidates")
            ax.tick_params(axis = "x", rotation = 20)
            ax.grid(axis = "y", alpha = 0.25)
            runtime.plt.tight_layout()
            path = (root / "figure_MASTER_primary_narrowing_summary.png")
            runtime.plt.savefig(path, dpi = 180, bbox_inches = "tight")
            runtime.plt.close(fig)
            outputs["primary_narrowing_summary"] = str(path)
    except Exception:
        pass
    return outputs
