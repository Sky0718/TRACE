from __future__ import annotations
import pandas as pd

def geometry_dictionary_rows() -> pd.DataFrame:
    rows = [
        (
            "case_level",
            "geometric_narrowing_index",
            "Maximum branch-level diameter-drop-derived narrowing estimate.",
            "0-1",
        ),
        (
            "case_level",
            "geometric_narrowing_category",
            "Interval label derived from geometric_narrowing_index.",
            "text",
        ),
        (
            "case_level",
            "qa_score",
            "Rule-based processing quality score for rapid failure analysis.",
            "0-100",
        ),
        (
            "case_level",
            "manual_review_needed",
            "Flag indicating branch/centerline/visualization warnings needing inspection.",
            "0/1",
        ),
        (
            "case_level",
            "branch_failure_flag",
            "Flag for missing major branch, zero-length branch, or failed centerline.",
            "0/1",
        ),
        ("branch_level", "vessel_length_mm", "Branch centerline path length.", "mm"),
        (
            "branch_level",
            "vessel_diameter_mean_mm",
            "Mean diameter sampled from distance transform along centerline.",
            "mm",
        ),
        (
            "branch_level",
            "geometric_narrowing_index",
            "Branch-level diameter-drop-derived narrowing estimate.",
            "0-1",
        ),
        (
            "qa",
            "missing_major_branches",
            "Major branches missing from the branch table; expected LAD, LCX, RCA.",
            "pipe-separated text",
        ),
        (
            "visual_index",
            "visual_existing_count",
            "Number of expected QA figure/video files found on disk.",
            "count",
        ),
        (
            "narrowing_candidates",
            "narrowing_position_norm",
            "Normalized centerline position of the strongest geometric narrowing candidate.",
            "0-1 or -1",
        ),
    ]
    return pd.DataFrame(rows, columns = ["table", "field", "description", "unit"])

def topology_dictionary_rows() -> pd.DataFrame:
    base = geometry_dictionary_rows()
    extra = pd.DataFrame(
        [
            (
                "case_level",
                "main_visualization_mode",
                "Primary paper visual mode; v1.3.0 uses opaque exact full-vessel mask projection.",
                "text",
            ),
            (
                "case_level",
                "semantic_branch_warning",
                "Reminder that branch names are algorithmic estimates from binary vessel masks.",
                "text",
            ),
            (
                "visual_index",
                "paper_orthogonal_branch_map_path",
                "Compatibility path; in v1.3.0 this is a full coronary-tree map, not a semantic branch-color proof.",
                "path",
            ),
        ],
        columns = ["table", "field", "description", "unit"],
    )
    return pd.concat([base, extra], ignore_index = True)

def reliability_dictionary_rows() -> pd.DataFrame:
    base = topology_dictionary_rows()
    extra = pd.DataFrame(
        [
            (
                "branch_level",
                "candidate_present",
                "Whether a branch has a nonzero geometric narrowing candidate with valid position.",
                "0/1",
            ),
            (
                "branch_level",
                "candidate_reliability_score",
                "Rule-based reliability score for the geometric narrowing candidate.",
                "0-100",
            ),
            (
                "branch_level",
                "candidate_confidence_category",
                "Candidate confidence category: high, moderate, low_review, or no_candidate.",
                "text",
            ),
            (
                "branch_level",
                "candidate_reliable_for_statistics",
                "Whether the candidate is reliable enough for aggregate statistics.",
                "0/1",
            ),
            (
                "branch_level",
                "candidate_near_endpoint_flag",
                "Candidate is near the first or last 8% of the centerline.",
                "0/1",
            ),
            (
                "branch_level",
                "candidate_near_bifurcation_proxy_flag",
                "Candidate is in a left-branch mid-zone where bifurcation artifacts are more likely.",
                "0/1",
            ),
            (
                "branch_level",
                "sharp_spike_flag",
                "Candidate peak is narrow/spike-like by centerline profile width.",
                "0/1",
            ),
            (
                "branch_level",
                "short_lesion_flag",
                "Estimated lesion support length is very short.",
                "0/1",
            ),
            (
                "branch_level",
                "high_hu_overlap_flag",
                "Candidate overlaps peripheral high-HU proximity; possible plaque/calcification or blooming artifact.",
                "0/1",
            ),
            (
                "branch_level",
                "area_agreement_flag",
                "Cross-sectional area evidence supports the diameter-profile candidate.",
                "0/1",
            ),
            (
                "branch_level",
                "peak_width_mm",
                "Approximate width of the geometric narrowing peak at 60% of peak score.",
                "mm",
            ),
            (
                "branch_level",
                "reliability_notes",
                "Pipe-separated reasons affecting candidate reliability.",
                "text",
            ),
            (
                "qa",
                "narrowing_review_needed_flag",
                "At least one narrowing candidate is low-confidence or spike-like.",
                "0/1",
            ),
            (
                "qa",
                "low_confidence_narrowing_branches",
                "Branches with low-review geometric narrowing candidates.",
                "pipe-separated text",
            ),
            (
                "narrowing_candidates",
                "candidate_reliability_score",
                "Reliability score carried into the candidate table.",
                "0-100",
            ),
            (
                "narrowing_candidates",
                "candidate_confidence_category",
                "Confidence category carried into the candidate table.",
                "text",
            ),
        ],
        columns = ["table", "field", "description", "unit"],
    )
    return pd.concat([base, extra], ignore_index = True).drop_duplicates(
        subset = ["table", "field"], keep = "last"
    )

def data_dictionary_rows() -> pd.DataFrame:
    base = reliability_dictionary_rows()
    extra = pd.DataFrame(
        [
            (
                "branch_level",
                "candidate_reliable_for_broad_statistics",
                "Broad/exploratory reliability flag; equivalent to candidate_reliable_for_statistics.",
                "0/1",
            ),
            (
                "branch_level",
                "candidate_reliable_for_primary_statistics",
                "Strict primary-statistics flag for manuscript main narrowing analysis.",
                "0/1",
            ),
            (
                "branch_level",
                "candidate_primary_exclusion_reason",
                "Pipe-separated reasons excluding a candidate from primary statistics.",
                "text",
            ),
            (
                "branch_level",
                "candidate_primary_statistics_rule",
                "Rule used to define the primary-statistics inclusion flag.",
                "text",
            ),
            (
                "narrowing_candidates",
                "candidate_reliable_for_broad_statistics",
                "Broad/exploratory reliability flag carried into candidate table.",
                "0/1",
            ),
            (
                "narrowing_candidates",
                "candidate_reliable_for_primary_statistics",
                "Strict primary-statistics flag carried into candidate table.",
                "0/1",
            ),
            (
                "narrowing_candidates",
                "candidate_primary_exclusion_reason",
                "Reason(s) why a candidate is excluded from primary statistics.",
                "text",
            ),
            (
                "narrowing_candidates",
                "candidate_primary_statistics_rule",
                "Rule used to define primary-statistics inclusion.",
                "text",
            ),
            (
                "case_level",
                "primary_reliable_narrowing_candidate_count",
                "Number of narrowing candidates included in primary statistics for this case.",
                "count",
            ),
            (
                "case_level",
                "exploratory_narrowing_candidate_count",
                "Number of all computational narrowing candidates for this case.",
                "count",
            ),
            (
                "case_level",
                "primary_excluded_narrowing_candidate_count",
                "Number of computational candidates excluded from primary statistics.",
                "count",
            ),
            (
                "case_level",
                "primary_excluded_narrowing_reasons",
                "JSON dictionary of exclusion reasons for this case.",
                "JSON text",
            ),
            (
                "qa",
                "primary_reliable_narrowing_candidate_count",
                "Number of primary-statistics narrowing candidates for this case.",
                "count",
            ),
            (
                "qa",
                "primary_excluded_narrowing_candidate_count",
                "Number of candidate excluded from primary statistics for this case.",
                "count",
            ),
        ],
        columns = ["table", "field", "description", "unit"],
    )
    return pd.concat([base, extra], ignore_index = True).drop_duplicates(
        subset = ["table", "field"], keep = "last"
    )

def _robustness_data_dictionary() -> pd.DataFrame:
    rows = [
        (
            "case",
            "variant",
            "Mask perturbation variant: original, erosion, dilation, or opening.",
            "text",
        ),
        ("case", "mask_volume_mm3", "Total perturbed coronary mask volume.", "mm^3"),
        (
            "case",
            "surface_volume_ratio",
            "Surface voxel count divided by total mask voxels; proxy for boundary complexity.",
            "ratio",
        ),
        (
            "case",
            "branch_count",
            "Number of branch masks generated after branch assignment.",
            "count",
        ),
        (
            "case",
            "total_branch_length_mm",
            "Sum of extracted branch centreline lengths.",
            "mm",
        ),
        (
            "case",
            "max_geometric_narrowing_index",
            "Maximum branch-level geometric narrowing index in the case.",
            "0-1",
        ),
        (
            "case",
            "primary_reliable_candidate_count",
            "Number of candidates meeting strict primary reliability criteria.",
            "count",
        ),
        (
            "branch",
            "profile_length_mm",
            "Length of profile used for dense diameter/narrowing analysis.",
            "mm",
        ),
        (
            "branch",
            "ratio_auc_mm",
            "Area under narrowing-index profile along centreline distance.",
            "mm",
        ),
        (
            "branch",
            "length_ratio_ge_25_mm",
            "Centreline length where narrowing profile is at least 0.25.",
            "mm",
        ),
        (
            "branch",
            "diameter_gradient_max_abs_mm_per_mm",
            "Maximum absolute diameter gradient along centreline.",
            "mm/mm",
        ),
        (
            "branch",
            "candidate_reliability_score",
            "Rule-based candidate reliability score.",
            "0-100",
        ),
        (
            "delta_case",
            "abs_delta_max_geometric_narrowing_index",
            "Absolute change in maximum case narrowing after perturbation.",
            "0-1",
        ),
        (
            "delta_case",
            "max_category_changed",
            "Whether the maximum case narrowing category changed after perturbation.",
            "0/1",
        ),
        (
            "delta_branch",
            "category_changed",
            "Whether branch-level narrowing category changed after perturbation.",
            "0/1",
        ),
        (
            "instability",
            "any_high_instability_flag",
            "Composite flag for major perturbation sensitivity.",
            "0/1",
        ),
    ]
    return pd.DataFrame(rows, columns = ["table", "field", "description", "unit"])
