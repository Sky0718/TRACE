from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Tuple
import numpy as np
import pandas as pd
import root_topology_backend_v2 as ref

ALGORITHM_VERSION = "cta_lumen_local_segment_additional_review_v2_0_10"
DEFAULT_CENTERLINE_STEP_MM = 0.55
DEFAULT_CROSS_SECTION_PIXEL_MM = 0.18
DEFAULT_PATCH_HALF_WIDTH_MM = 5.5
DEFAULT_MIN_REPORTABLE_RATIO = 0.10
RootAssignment = ref.RootAssignment
CenterlineResult = ref.CenterlineResult
base_branch_name = ref.base_branch_name
severity_interval = ref.severity_interval
sample_plane_batch = ref.sample_plane_batch
estimate_branch_lumen_reference = ref.estimate_branch_lumen_reference
parallel_transport_frames = ref.parallel_transport_frames
PARENT_BRANCHES = ref.PARENT_BRANCHES

@dataclass
class CrossSectionMeasurement:
    index: int
    distance_mm: float
    center_voxel_x: float
    center_voxel_y: float
    center_voxel_z: float
    lumen_area_mm2: float
    equivalent_diameter_mm: float
    min_diameter_mm: float
    max_diameter_mm: float
    lumen_mean_hu: float
    background_mean_hu: float
    lumen_noise_hu: float
    contrast_to_noise: float
    boundary_support: float
    contour_component_agreement: float
    calcification_fraction: float
    center_shift_mm: float
    quality_score: float
    valid: int
    quality_flags: str
    refined_center_u_mm: float = 0.0
    refined_center_v_mm: float = 0.0
    component_area_mm2: float = float("nan")
    tracking_overlap: float = float("nan")
    tracking_area_ratio: float = float("nan")
    tracking_center_delta_mm: float = float("nan")
    tracking_score: float = 0.0
    tracking_status: str = "independent"
    outer_mask_fraction: float = 0.0
    center_hu: float = float("nan")
    unresolved_lumen: int = 0
    interpolated_for_detection: int = 0
    gradient_boundary_support: float = 0.0
    core_overlap_fraction: float = 0.0
    center_inside_component: int = 0
    radial_cv: float = float("nan")
    eccentricity: float = float("nan")
    lumen_identity_score: float = 0.0

@dataclass
class LesionResult:
    case_id: str
    branch: str
    lesion_index: int
    start_index: int
    peak_index: int
    end_index: int
    start_distance_mm: float
    peak_distance_mm: float
    end_distance_mm: float
    lesion_length_mm: float
    position_norm: float
    segment: str
    stenosis_ratio: float
    stenosis_interval: str
    area_stenosis_ratio: float
    minimum_lumen_diameter_mm: float
    reference_diameter_mm: float
    minimum_lumen_area_mm2: float
    reference_area_mm2: float
    proximal_reference_diameter_mm: float
    distal_reference_diameter_mm: float
    proximal_recovery: float
    distal_recovery: float
    local_prominence: float
    candidate_signal: float
    boundary_support: float
    contour_quality: float
    calcification_fraction: float
    evidence_score: float
    accepted: int
    rejection_reasons: str
    warning_flags: str
    root_source: str
    centerline_quality: str
    assessment_status: str = "assessable"
    not_assessable_reason: str = ""
    start_position_norm: float = -1.0
    peak_position_norm: float = -1.0
    end_position_norm: float = -1.0
    start_segment: str = "unknown"
    peak_segment: str = "unknown"
    end_segment: str = "unknown"
    segment_span: str = "unknown"
    segment_method: str = "normalized_length_fallback"
    segment_confidence: float = 0.0
    start_voxel_x: float = float("nan")
    start_voxel_y: float = float("nan")
    start_voxel_z: float = float("nan")
    peak_voxel_x: float = float("nan")
    peak_voxel_y: float = float("nan")
    peak_voxel_z: float = float("nan")
    end_voxel_x: float = float("nan")
    end_voxel_y: float = float("nan")
    end_voxel_z: float = float("nan")
    start_x_mm: float = float("nan")
    start_y_mm: float = float("nan")
    start_z_mm: float = float("nan")
    peak_x_mm: float = float("nan")
    peak_y_mm: float = float("nan")
    peak_z_mm: float = float("nan")
    end_x_mm: float = float("nan")
    end_y_mm: float = float("nan")
    end_z_mm: float = float("nan")
    peak_valid_run_count: int = 0
    peak_valid_run_mm: float = 0.0
    peak_section_valid: int = 0
    reference_valid_count: int = 0
    reference_valid_fraction: float = 0.0
    suspected_near_occlusion: int = 0
    recovery_between_adjacent_lesions: float = float("nan")
    pre_verifier_accepted: int = 0
    lumen_identity_score: float = 0.0
    gradient_boundary_support: float = 0.0
    profile_noise_ratio: float = 0.0
    profile_signal_z: float = 0.0
    diameter_area_agreement: float = 0.0
    development_verifier_probability: float = float("nan")
    verifier_version: str = ""
    stenosis_ratio_uncorrected: float = float("nan")
    partial_volume_correction_mm: float = 0.0
    severity_method: str = "direct_valid_cross_sections"
    navigation_mask_stenosis_ratio: float = float("nan")
    cta_mask_concordance: float = float("nan")
    tracking_overlap_median: float = float("nan")
    tracking_supported_fraction: float = float("nan")
    center_inside_fraction: float = float("nan")
    radial_cv_median: float = float("nan")
    tracking_score_median: float = float("nan")
    center_shift_median_mm: float = float("nan")
    topology_risk_flag: str = ""
    stenosis_ratio_lower: float = float("nan")
    stenosis_ratio_upper: float = float("nan")
    stenosis_interval_lower: str = ""
    stenosis_interval_upper: str = ""
    peak_relative_lumen_hu: float = float("nan")
    peak_radial_cv: float = float("nan")
    peak_eccentricity: float = float("nan")
    identity_conflict_flag: int = 0
    identity_conflict_reason: str = ""
    repair_attempted: int = 0
    repair_succeeded: int = 0
    repair_method: str = ""
    recovery_plateau_fraction: float = float("nan")
    recovery_plateau_mm: float = 0.0
    consolidation_action: str = ""
    final_decision: str = ""
    measurement_reliability: str = ""
    resolution_limit_flag: int = 0
    formal_ratio_source: str = ""
    v2_0_5_rule: str = ""
    clinical_output_status: str = "formal"
    clinical_review_reason: str = ""
    local_coverage_override: int = 0
    calcification_navigation_conflict: int = 0
    secondary_low_confidence_flag: int = 0
    coverage_scope: str = "whole_branch"
    final_state_version: str = ""
    parent_reference_diameter_mm: float = float("nan")
    parent_reference_stenosis_ratio: float = float("nan")
    parent_reference_used: int = 0
    short_branch_measurement_rule: str = ""
    multilesion_independence_score: float = float("nan")
    multilesion_recovery_plateau_mm: float = 0.0
    anatomical_landmark_evidence: str = ""
    status_consistency_checked: int = 0
    v2_0_7_rule: str = ""
    short_branch_independent_support_score: float = float("nan")
    short_branch_audit_reason: str = ""
    parent_reference_advisory_only: int = 0
    deduplication_action_v2_0_7: str = ""
    duplicate_group_id: str = ""
    independence_from_all_neighbours: int = 0
    segment_review_required: int = 0
    v2_0_8_rule: str = ""
    severity_uncertainty_weight: float = 0.0
    severity_uncertainty_reason: str = ""
    competing_segment_candidates: str = ""
    segment_uncertainty_span: str = ""
    segment_alternative: str = ""
    clinical_review_priority: str = "routine"
    diffuse_multilesion_recovery: int = 0
    diffuse_multilesion_recovery_median: float = float("nan")
    diffuse_multilesion_recovery_plateau_mm: float = 0.0
    diffuse_multilesion_recovery_fraction: float = float("nan")
    v2_0_9_rule: str = ""
    stenosis_ratio_direct_measured: float = float("nan")
    stenosis_ratio_resolution_upper: float = float("nan")
    severity_interval_consistency_checked: int = 0
    blooming_overestimate_flag: int = 0
    blooming_adjusted_ratio: float = float("nan")
    blooming_adjustment_reason: str = ""
    resolution_uncertainty_applied: int = 0
    resolution_uncertainty_mm: float = 0.0
    borderline_review_candidate: int = 0
    borderline_review_reason: str = ""
    reported_segment: str = ""
    reported_segment_span: str = ""
    reported_segment_status: str = ""
    v2_0_10_rule: str = ""
    local_segment_competitor_count: int = 0
    local_segment_competitors: str = ""
    distant_competing_candidate_count: int = 0
    distant_competing_segments: str = ""
    additional_lesion_review_required: int = 0
    additional_lesion_review_candidate: int = 0
    additional_lesion_review_reason: str = ""
    adjacent_severe_shoulder_flag: int = 0
    adjacent_severe_shoulder_reason: str = ""
    area_equivalent_diameter_stenosis_ratio: float = float("nan")
    localization_confidence: float = 0.0
    localization_audit_status: str = ""

@dataclass
class BranchResult:
    case_id: str
    branch: str
    label_value: int
    centerline_length_mm: float
    endpoint_distance_mm: float
    tortuosity: float
    root_source: str
    centerline_method: str
    centerline_quality: str
    cross_section_count: int
    valid_cross_section_count: int
    valid_cross_section_fraction: float
    lumen_diameter_mean_mm: float
    lumen_diameter_median_mm: float
    lumen_diameter_min_mm: float
    lumen_area_mean_mm2: float
    lesion_count: int
    maximum_stenosis_ratio: float
    maximum_stenosis_interval: str
    maximum_stenosis_segment: str
    maximum_stenosis_position_norm: float
    branch_assessability: str
    algorithm_version: str = "cta_lumen_evidence_concordant_short_branch_v2_0_7"
    longest_valid_run_mm: float = 0.0
    cta_centerline_support_fraction: float = 0.0
    cta_recentered_mean_shift_mm: float = 0.0
    cta_recentered_max_shift_mm: float = 0.0
    cta_trimmed_tip_mm: float = 0.0
    branch_assessability_detail: str = ""
    segment_method_default: str = "normalized_length_fallback"
    coverage_scope: str = "whole_branch"
    formal_lesion_count: int = 0
    local_only_lesion_count: int = 0
    whole_branch_negative_interpretation_valid: int = 1
    final_state_version: str = ""
    review_required_lesion_count: int = 0
    severity_uncertainty_lesion_count: int = 0
    diffuse_secondary_lesion_count: int = 0
    borderline_review_candidate_count: int = 0
    formal_segment_review_count: int = 0
    formal_severity_uncertainty_count: int = 0
    additional_lesion_review_count: int = 0
    adjacent_shoulder_review_count: int = 0
    maximum_stenosis_segment_span: str = ""
    maximum_stenosis_segment_status: str = ""
    additional_high_signal_candidate_count: int = 0
    localization_review_lesion_count: int = 0

@dataclass
class PlanePatch:
    image: np.ndarray
    outer_mask: np.ndarray
    calc_mask: np.ndarray
    axis_mm: np.ndarray
    center_pixel: Tuple[float, float]
    refined_center_pixel: Tuple[float, float]
    lumen_component: np.ndarray
    contour_xy_mm: np.ndarray
    radii_mm: np.ndarray
    measurement: CrossSectionMeasurement

def dataframe_from_dataclasses(items: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in items])
