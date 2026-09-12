from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CenterlineConcept:
    method: str
    quality_flag: str
    point_count: int
    length_mm: float
    endpoint_distance_mm: float
    tortuosity: float
    mean_diameter_mm: float
    median_diameter_mm: float
    min_diameter_mm: float
    max_diameter_mm: float
    stenosis_presence: int
    stenosis_ratio: float
    stenosis_position_norm: float
    stenosis_segment: str
    stenosis_length_mm: float
    reference_diameter_mm: float
    min_lumen_diameter_mm: float

@dataclass
class BranchConcept:
    case_id: str
    branch: str
    branch_name: str
    branch_assignment_method: str
    branch_voxels: int
    vessel_volume_mm3: float
    vessel_length_mm: float
    vessel_length_method: str
    vessel_diameter_mean_mm: float
    vessel_diameter_median_mm: float
    vessel_diameter_min_mm: float
    vessel_diameter_max_mm: float
    stenosis_presence: int
    stenosis_ratio: float
    stenosis_position_norm: float
    stenosis_segment: str
    lesion_length_mm: float
    tortuosity: float
    calcium_volume_mm3: float
    calcium_burden_pct: float
    calcium_lesion_count_3d: int
    calcium_mean_hu: float
    calcium_max_hu: float
    centerline_point_count: int
    centerline_quality_flag: str

@dataclass
class CaseConcept:
    case_id: str
    source_img: str
    source_label: str
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float
    total_vessel_volume_mm3: float
    calcium_volume_mm3: float
    calcium_burden_pct: float
    calcium_lesion_count_3d: int
    calcium_mean_hu: float
    calcium_max_hu: float
    agatston_score: float
    agatston_grade: str
    risk_level: str
    branch_count: int
    LAD_length_mm: float
    RCA_length_mm: float
    LAD_mean_diameter_mm: float
    RCA_mean_diameter_mm: float
    max_geometric_narrowing_index: float
    max_stenosis_branch: str
    stenosis_position_norm: float
    stenosis_segment: str
    tortuosity_mean: float
    tortuosity_max: float
    processing_status: str
    warning: str = ""
