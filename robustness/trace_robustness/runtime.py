from __future__ import annotations
from typing import Optional
from pathlib import Path

try:
    import nibabel as nib

    NIBABEL_OK = True
except Exception:
    NIBABEL_OK = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False
try:
    import cv2

    CV2_OK = True
except Exception:
    CV2_OK = False
try:
    from skimage.morphology import skeletonize

    SKIMAGE_SKELETON_OK = True
except Exception:
    SKIMAGE_SKELETON_OK = False
__all__ = ["nib", "plt", "cv2", "skeletonize"]

DEFAULT_CALCIUM_THRESHOLD_HU = 130.0
DEFAULT_MIN_CALCIUM_VOLUME_MM3 = 1.0
DEFAULT_MIN_BRANCH_COMPONENT_VOXELS = 1500
DEFAULT_LEFT_BIFURCATION_MIN_AREA_PX = 20
DEFAULT_STENOSIS_REPORT_THRESHOLD = 0.25
DEFAULT_QA_FIGURE = True
NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (((dx == 0) and (dy == 0)) and (dz == 0))
]
BRANCH_LONG_NAMES = {
    "RCA": "right_coronary_artery",
    "LM": "left_main",
    "LAD": "left_anterior_descending",
    "LCX": "left_circumflex",
}
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATABASE_ROOT = (SCRIPT_DIR / "cta_concept_results")
DATABASE_CASES_ROOT = (DATABASE_ROOT / "cases")
DATABASE_COLUMNS = [
    "case_id",
    "LAD_length",
    "RCA_length",
    "calcium_volume",
    "calcium_burden",
    "stenosis_ratio",
    "stenosis_position",
    "tortuosity",
]
MASTER_CASE_LEVEL_CSV_NAME = "cta_master_case_level.csv"
MASTER_BRANCH_LEVEL_CSV_NAME = "cta_master_branch_level.csv"
MASTER_QA_CSV_NAME = "cta_master_qa.csv"
MASTER_VISUAL_INDEX_CSV_NAME = "cta_master_visual_index.csv"
MASTER_STENOSIS_CSV_NAME = "cta_master_geometric_narrowing_candidates.csv"
MASTER_DATASET_SUMMARY_CSV_NAME = "cta_master_dataset_summary.csv"
MASTER_DATA_DICTIONARY_CSV_NAME = "cta_master_data_dictionary.csv"
MASTER_MANIFEST_CSV_NAME = "cta_master_case_manifest.csv"
MASTER_XLSX_NAME = "cta_master_paper_database.xlsx"
MASTER_SQLITE_NAME = "cta_master_paper_database.sqlite"
MAX_VIDEO_FRAMES = 120
VIDEO_FPS = 8.0
BRANCH_RGB = {
    "RCA": (231, 76, 60),
    "LM": (241, 196, 15),
    "LAD": (46, 204, 113),
    "LCX": (52, 152, 219),
    "OTHER": (149, 165, 166),
}
DEFAULT_VISUAL_BACKGROUND_MODE = "dark"
VISUAL_BACKGROUND_MODE = DEFAULT_VISUAL_BACKGROUND_MODE
EXPECTED_VISUAL_OUTPUTS = {
    "overview": "visuals/figure_A_concepts_overview.png",
    "heatmap_slices": "visuals/figure_A_heatmap_slices.png",
    "mip_projection": "visuals/figure_A_mip_projection.png",
    "branch_concepts": "visuals/figure_SF_branch_concepts.png",
    "centerline_profiles": "visuals/figure_SF_centerline_profiles.png",
    "stenosis_marked_figure": "visuals/figure_SF_stenosis_marked.png",
    "axial_video": "videos/video_A_axial_vessel_calcium.mp4",
    "branch_video": "videos/video_SF_branch_masks.mp4",
    "stenosis_marked_video": "videos/video_SF_stenosis_marked.mp4",
}
SHORT_BRANCH_NARROWING_LENGTH_MM = 10.0
TORTUOSITY_OUTLIER_THRESHOLD = 10.0
LM_REPAIR_MIN_VOXELS = 40
CALCIFICATION_MIN_ADAPTIVE_HU = 350.0
CALCIFICATION_CORE_MIN_DISTANCE_MM = 0.75
CALCIFICATION_PERIPHERAL_BAND_MM = 1.35
OPTIONAL_PAPER_VISUAL_OUTPUTS = {
    "paper_multipanel_summary": "visuals/figure_PAPER_multipanel_summary.png",
    "paper_orthogonal_branch_map": "visuals/figure_PAPER_orthogonal_branch_map.png",
    "paper_narrowing_evidence": "visuals/figure_PAPER_narrowing_evidence.png",
}
VESSEL_OVERLAY_RGB = (245, 98, 28)
VESSEL_EDGE_RGB = (0, 220, 255)
CALCIUM_OVERLAY_RGB = (255, 218, 65)
EDGE_THICKNESS_PX = 1
NARROWING_ENDPOINT_MARGIN = 0.08
NARROWING_SHARP_SPIKE_WIDTH_MM = 2.0
NARROWING_MIN_LESION_LENGTH_MM = 1.5
NARROWING_HIGH_HU_PROXIMITY_THRESHOLD = 0.55
NARROWING_HIGH_CONFIDENCE_SCORE = 75.0
NARROWING_MODERATE_CONFIDENCE_SCORE = 50.0
DEFAULT_PARALLEL_WORKERS = 5
DEFAULT_SKIP_EXISTING = True
PARAMETER_HASH_KEYS = [
    "algorithm_version",
    "threshold_hu",
    "min_calc_vol_mm3",
    "min_branch_voxels",
    "geometric_narrowing_threshold",
    "save_qa",
    "visual_background_mode",
    "high_hu_mode",
    "narrowing_reliability_mode",
    "branch_confidence_mode",
]
ALGORITHM_VERSION = "concept_extraction_CTA_v1_5_7_visual_background"
PRIMARY_STATISTICS_BRANCHES = ("LAD", "LCX", "RCA")
PRIMARY_STATISTICS_RULE_TEXT = "Primary narrowing statistics include only LAD/LCX/RCA candidates with high semantic branch confidence, high candidate confidence, and no endpoint, left-bifurcation-proxy, high-HU-overlap, sharp-spike, or short-lesion flags."
ROBUSTNESS_ANALYSIS_VERSION = "robustness_v2_0_full_raw_concept_uncertainty"
ROBUSTNESS_VARIANT_NAMES = ("original", "erode_1voxel", "dilate_1voxel", "open_1voxel")
ROBUSTNESS_CASE_CSV_NAME = "cta_master_robustness.csv"
ROBUSTNESS_BRANCH_CSV_NAME = "cta_master_robustness_branch_level.csv"
ROBUSTNESS_CANDIDATE_CSV_NAME = "cta_master_robustness_candidate_level.csv"
ROBUSTNESS_DELTA_CASE_CSV_NAME = "cta_master_robustness_delta_case.csv"
ROBUSTNESS_DELTA_BRANCH_CSV_NAME = "cta_master_robustness_delta_branch.csv"
ROBUSTNESS_INSTABILITY_CSV_NAME = "cta_master_robustness_instability_features.csv"
ROBUSTNESS_TRANSITION_CSV_NAME = "cta_master_robustness_category_transitions.csv"
ROBUSTNESS_DICTIONARY_CSV_NAME = "cta_master_robustness_data_dictionary.csv"
ENTRY_SCRIPT = (SCRIPT_DIR / "concept_extraction_CTA.py")

def set_visual_background_mode(mode: Optional[str] = None) -> str:
    global VISUAL_BACKGROUND_MODE
    value = str((mode or DEFAULT_VISUAL_BACKGROUND_MODE)).strip().lower()
    aliases = {
        "": "dark",
        "default": "dark",
        "d": "dark",
        "dark": "dark",
        "black": "dark",
        "original": "dark",
        "w": "white",
        "white": "white",
        "pure_white": "white",
        "pure-white": "white",
        "paper": "white",
    }
    value = aliases.get(value, value)
    if (value not in ("dark", "white")):
        value = "dark"
    VISUAL_BACKGROUND_MODE = value
    return VISUAL_BACKGROUND_MODE

def visual_background_mode() -> str:
    return str((VISUAL_BACKGROUND_MODE or DEFAULT_VISUAL_BACKGROUND_MODE))

def visual_fig_bg() -> str:
    return "#101820" if (visual_background_mode() == "dark") else "#ffffff"

def visual_paper_bg() -> str:
    return "#0b1020" if (visual_background_mode() == "dark") else "#ffffff"

def visual_axis_bg() -> str:
    return "#17212b" if (visual_background_mode() == "dark") else "#ffffff"

def visual_image_axis_bg() -> str:
    return "black" if (visual_background_mode() == "dark") else "#ffffff"

def visual_text_color() -> str:
    return "white" if (visual_background_mode() == "dark") else "black"
