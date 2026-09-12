# TRACE: **T**raceable **R**epresentation of **A**natomy and **C**oronary **E**vidence

TRACE links coronary anatomy, centreline geometry, lumen measurements, lesions and supporting image evidence in structured records. This repository contains the extraction and analysis code accompanying **TRACE: Evidence-Linked Coronary Quantification from CCTA to Clinical Reports**.

The release contains three workflows:

1. Extract branch-level and lesion-level records from CCTA images and coronary segmentations.
2. Measure binary-mask geometry and its response to boundary perturbations in ImageCAS.
3. Calculate clinical agreement statistics from a separately prepared endpoint ledger.

The clinical and binary-mask workflows use different inputs and measurement routes. Their narrowing outputs are not interchangeable. Segmentation generation and hospital-report interpretation are outside the extraction pipeline.

## Repository layout

```text
.
├── clinical/
│   ├── concept_extraction_CTA_stenosis_v2_0_10.py
│   └── lumen/
├── robustness/
│   ├── concept_extraction_CTA.py
│   └── trace_robustness/
├── analysis/
│   └── reproduce_clinical_statistics.py
├── requirements.txt
└── CITATION.cff
```

The three scripts shown above are the command-line entry points. Supporting modules remain beside them; they do not need to be run individually.

## Installation

The code has been tested with 64-bit Python 3.10. Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

Compressed DICOM inputs require a decoder for their transfer syntax. For JPEG 2000 inputs, also install `pylibjpeg-openjpeg` with `python -m pip install pylibjpeg-openjpeg`.

The pipelines use CPU processing and system memory. A dedicated graphics card and a CUDA installation are not required. Commands below work from the repository root; replace the example input and output paths with local paths. Keep clinical inputs and generated results outside the repository.

## Clinical extraction

Place each examination's paired files directly inside one input directory. The following names illustrate the required pattern, using a fictional examination identifier:

```text
case_0-1 CTA.zip
case_0-1 Segmentation-label.nii.gz
case_0-1 Segmentation_ColorTable.txt
```

The CTA ZIP contains the DICOM series. The segmentation accepts `.nii` or `.nii.gz`, and the colour table is optional. File names must share a `case_<group>-<number>` identifier; keep the space before `CTA.zip`. A colour-table row has the form `label name R G B A`.

Without a colour table, navigation branches are inferred geometrically from the combined positive-label mask. This does not recover the original named-branch mapping or point markers. For anatomical label fidelity, provide the matching colour table or segment names in a supported MRB scene.

A Slicer `.mrb` file can replace the separate segmentation and colour table, for example `case_0-1 Segmentation.mrb`. It must contain the segmentation, segment metadata and compatible reference-volume geometry, without unapplied parent transforms. The CTA DICOM ZIP is still required. If both a NIfTI label and an MRB are present, the NIfTI label takes precedence. An outer ZIP containing the paired files is also accepted as `--input`.

```bash
python clinical/concept_extraction_CTA_stenosis_v2_0_10.py --input "../private-data/clinical" --output "../trace-results/clinical"
```

By default, all complete examinations and all available branches are processed, with quality-assurance outputs enabled. Incomplete file sets are skipped; a directory with no complete examination raises an error. Input discovery does not recursively search nested case folders.

Optional selection:

```bash
python clinical/concept_extraction_CTA_stenosis_v2_0_10.py --input "../private-data/clinical" --output "../trace-results/clinical-selected" --cases "case_0-1" --branches "LM,LAD,LCX,RCA"
```

`--cases "1-30"` selects examination numbers 1 through 30, including matching numbers from different groups. In a mixed-group directory, use comma-separated full identifiers instead. `--no-qa` disables QA figures and the additional CPR/section-stack array exports. Run the entry point with `--help` for the remaining parameters.

Each examination produces branch measurements, accepted lesions, candidate states, centreline/profile records and provenance files. Important outputs include:

| Output | Contents |
|---|---|
| `stenosis_lesions.csv` | Formally accepted lesion records |
| `branch_mask_dataset.csv` | Branch-level measurements and output states |
| `stenosis_candidates_all.csv` | Candidate records, including records not accepted as formal lesions |
| `stenosis_review_candidates.csv` | Candidates retained for review |
| `case_summary.json` and `run_manifest.json` | Examination summary and execution provenance |
| `centerlines/`, `profiles/` and `visuals/` | Spatial records, lumen profiles and QA images, where generated |

Batch-level tables and `batch_summary.json` are written to the output root. Failed examinations are recorded in `stenosis_v2_errors.csv`; batch input errors are recorded in `batch_error.json`. Normal execution is silent, so inspect these files and the process exit status. Use a fresh output directory for a new clinical run to avoid mixing results. Review, rejected and not-assessable records must not be treated as accepted lesions or silently recoded as normal vessels.

## ImageCAS geometry and robustness

Provide the original ImageCAS image/label pairs, such as `1.img.nii.gz` and `1.label.nii.gz`, directly inside one directory. Images must be in Hounsfield units, with spatially aligned masks. The source image and mask paths must remain accessible when the perturbation stage is run.

First extract the baseline records:

```bash
python robustness/concept_extraction_CTA.py --batch "../private-data/ImageCAS" --out "../trace-results/imagecas/cases" --workers 4
```

Then rebuild the aggregate outputs and run the boundary-perturbation analysis:

```bash
python robustness/concept_extraction_CTA.py --rebuild-master "../trace-results/imagecas/cases" --robustness-sample 1000 --workers 4
```

`--robustness-sample 1000` requests up to 1000 available examinations. Use the intended cohort size for another dataset; check the recorded number of completed examinations before interpreting the summaries. The analysis compares original masks with one-voxel erosion, dilation and morphological opening, using the same source images.

With the paths above, examination outputs and `cta_batch_run_log.csv` are stored under `../trace-results/imagecas/cases`; master tables and robustness outputs are stored under `../trace-results/imagecas`. Baseline QA images are enabled by default and can be disabled with `--no-qa`; numerical QA fields are still retained. Completed baseline cases are skipped by default on a repeated batch command, and the perturbation stage also resumes completed cases. Use a new output directory when inputs or parameters change. Avoid overwrite options unless regeneration of existing outputs is intended.

`--workers` controls parallel processes in this workflow. Start with a small worker count, measure memory use on representative volumes, and increase it within the machine's available memory and CPU capacity. On Windows, keep this setting at 60 or below. More workers are not a substitute for sufficient memory or storage throughput.

## Clinical statistics

The statistics script reads an external JSON endpoint ledger, not raw hospital reports or unfiltered lesion tables. Prepare one record per study–territory pair using the study's endpoint-selection and range-conversion rules. Original-output and reviewed-register records remain distinct.

```bash
python analysis/reproduce_clinical_statistics.py --input "../private-data/endpoint_ledger.json" --output "../trace-results/clinical_statistics.json"
```

The following fictional record illustrates the input format, not a dataset for estimating clinical performance:

```json
{
  "records": [
    {
      "study_code": "O_example",
      "territory": "LAD",
      "fully_assessable_raw": true,
      "P": {"analysis_value_pct": 37, "area_stenosis_pct": null, "lesion_length_mm": null},
      "H": {"analysis_value_pct": 40},
      "CAG": null,
      "locations": {"P": ["proximal"], "H": ["proximal"], "CAG": []}
    }
  ]
}
```

- `P`, `H` and `CAG` denote program CCTA, hospital CCTA and hospital coronary angiography, respectively.
- `analysis_value_pct` and `area_stenosis_pct` are numeric percentages from 0 to 100; `lesion_length_mm` is a nonnegative length in millimetres. Percentage strings and ranges must be resolved before this step.
- `null` denotes missing data; numeric `0` denotes a zero endpoint. They are not interchangeable.
- `study_code` and `territory` must be nonempty, contain no colon, and form a unique pair. `fully_assessable_raw` must be a Boolean value for the original-output assessability analysis.
- Default prefixes are `O` for original-output studies and `R` for reviewed-register studies. These can be changed with `--original-prefix` and `--reviewed-prefix`.
- Valid location labels are `proximal`, `proximal-mid`, `mid`, `mid-distal` and `distal`. Use an empty list when a location is unavailable.

The script calculates continuous and category-based Spearman correlations, weighted kappa, category agreement, threshold-based classification measures, ROC AUC and conditional location agreement. Severity categories distinguish 0%, greater than 0% but below 25%, 25–<50%, 50–<70%, 70–<100%, and 100%. Classification thresholds are 25%, 50% and 70%. Location comparisons require both endpoints to be at least 25% and to have comparable locations. Patient-cluster bootstrap intervals use 10,000 resamples with seed 20260804. Undefined estimates are written as `null`.

The output key `same_modality_46` is a retained schema label, not a requirement for exactly 46 studies. Actual denominators are included in each analysis. An optional `--expected "path/to/reference_statistics.json"` validates the results against a separately supplied aggregate. Without that argument, no reference match is claimed.

## Tested hardware

Completed project runs have been reported on the following configurations. These are tested systems, not minimum requirements or comparative benchmarks.

| Processor | Memory | Graphics |
|---|---:|---|
| 2 × AMD EPYC 7H12 | 512 GB | Integrated graphics |
| AMD Ryzen 9 9950X | 128 GB | NVIDIA GeForce GTX 1050 Ti |
| Intel Core i5-10300H | 64 GB | NVIDIA GeForce RTX 2060 |
| Intel Core Ultra 9 285H | 32 GB | NVIDIA GeForce RTX 5070 |

Memory demand depends on volume size, QA exports and the number of parallel processes. The clinical extraction entry point processes examinations sequentially; the robustness entry point supports `--workers`.

## Data and reproducibility

Institutional images, segmentations, hospital reports and participant-level derived records are not distributed, to protect patient confidentiality and respect ethical constraints on data sharing. Obtain ImageCAS separately from its original provider under its data-use terms. The repository does not contain the restricted endpoint ledger, so the public code alone does not reconstruct the institutional study results.

Retain the source-data manifest, parameter settings, package versions and generated run records with each analysis. The supplied `.gitignore` uses an explicit source-file allowlist; new files are excluded until deliberately added. This reduces accidental inclusion but does not anonymise input data or outputs. Inspect all files before publishing, including filenames, images, embedded paths and identifiers.

TRACE supports research quantification and review; its outputs are not a standalone clinical diagnosis.

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The associated study is **TRACE: Evidence-Linked Coronary Quantification from CCTA to Clinical Reports**, by Tianyi Guan, Ruiqing Feng, Xiaoxuan Gong and Wei Long.

## License

The code and documentation in this repository are distributed under the [MIT License](LICENSE). This license does not grant access to or redistribution rights for institutional clinical data or third-party datasets, which remain subject to their own permissions and terms.
