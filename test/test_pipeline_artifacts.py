import json
from pathlib import Path
import numpy as np
import pytest

STAGE_DIRS = [
    "00_import",
    "01_standard_orientation",
    "02_downsample",
    "03_template_reg",
    "04_brainmasking",
    "05_mirroring",
    "06_csf_seg",
    "07_refine_csf_mask",
    "08_smooth",
    "09_ratio",
    "10_depression_rgb",
    "11_lesion_masks",
]

KEY_FILES = [
    "processing_summary.json",
    "00_import/input_nifti.nii.gz",
    "01_standard_orientation/input_lps.nii.gz",
    "01_standard_orientation/input_lps_centered.nii.gz",
    "03_template_reg/N2T.txt",
    "03_template_reg/T2N.txt",
    "04_brainmasking/refined_bmask.nii.gz",
    "05_mirroring/flipped_brain.nii",
    "06_csf_seg/ipsi_csfsegmentation.nii.gz",
    "06_csf_seg/mirror_csfsegmentation.nii.gz",
    "07_refine_csf_mask/ipsi_refined_tissue_mask.nii.gz",
    "07_refine_csf_mask/mirror_refined_tissue_mask.nii.gz",
    "08_smooth/ipsi_smoothed.nii.gz",
    "08_smooth/mirror_smoothed.nii.gz",
    "09_ratio/pct_depression.nii.gz",
    "10_depression_rgb/rncct_rgb.nii",
    "11_lesion_masks/masks_RGB_overlay.png",
]


@pytest.mark.parametrize("stage", STAGE_DIRS)
def test_stage_directory_exists(pipeline_output: Path, stage: str) -> None:
    assert (pipeline_output / stage).is_dir(), f"Stage directory missing: {stage}"


@pytest.mark.parametrize("rel_path", KEY_FILES)
def test_output_file_exists_and_nonempty(pipeline_output: Path, rel_path: str) -> None:
    p = pipeline_output / rel_path
    assert p.exists(), f"Missing: {rel_path}"
    assert p.stat().st_size > 0, f"Empty file: {rel_path}"


def test_summary_json_structure(pipeline_output: Path) -> None:
    summary_path = pipeline_output / "processing_summary.json"
    data = json.loads(summary_path.read_text())

    for key in ("input", "output", "lesion_volumes", "elapsed_times"):
        assert key in data, f"Missing key in summary: {key}"


def test_summary_lesion_volumes(pipeline_output: Path) -> None:
    data = json.loads((pipeline_output / "processing_summary.json").read_text())
    lesion_volumes = data["lesion_volumes"]

    # Default thresholds are 1.0 and 4.9 — keys are stored as strings
    assert np.abs(lesion_volumes["1.00"] - 232.9) < 2, (
        f"Unexpected volume for threshold '1.0': {lesion_volumes['1.00']}"
    )
    assert np.abs(lesion_volumes["4.90"] - 6.8) < 1, (
        f"Unexpected volume for threshold '4.9': {lesion_volumes['4.90']}"
    )


def test_summary_elapsed_times(pipeline_output: Path) -> None:
    data = json.loads((pipeline_output / "processing_summary.json").read_text())
    elapsed = data["elapsed_times"]

    assert len(elapsed) > 0, "elapsed_times is empty"
    for stage, t in elapsed.items():
        assert isinstance(t, (int, float)), f"elapsed_times[{stage}] is not numeric"
