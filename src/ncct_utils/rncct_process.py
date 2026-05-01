#!/usr/bin/env python
"""
#STEPS
#1) DCM2VOLUME (GC corr and equidistant resampling)
#2) Template2NCCT
#3) Mask NCCT
#4) Estimation of flip2native linear XFM
#5) Nonlin flip2self
#6) CSF segmentation in native and flip
#7) Ratio map generation

In order to estimate the flip to self xfm we need the origin and cosines for moving+fixed

"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ncct_utils import ncct_paths
from ncct_utils.rncct_config import rNCCTConfig
from ncct_utils.rNCCTfunctions import (
    brainmasker_candidate,
    calc_ratio,
    csf_seg,
    depression_map2rgb,
    dicom2nifti,
    flipped2self,
    lesion_masks2rgb,
    refine_csf_mask,
    smooth,
    template_reg,
    volume_average_downsample,
)
from regutils import simpleelastix_utils as sutl

template_l = ncct_paths.scct_unsmooth
erodemaskloc = ncct_paths.eroded_mask
head_aura = ncct_paths.head_aura


def run_config(config: rNCCTConfig) -> None:
    if config.input is None or config.output is None:
        print(
            "Both --input (-i) and --output (-o) are required to run the pipeline."
            "Use --version to check the version."
        )
        return

    # Type guard: after validation, we know these are not None
    output_path: str = config.output

    Path(output_path).mkdir(exist_ok=True, parents=True)
    nifti_name = "input_nifti.nii"
    if Path(config.input).is_dir():
        config.input = dicom2nifti(
            infolder=config.input,
            outfolder=output_path,
            nifti_name=nifti_name,
            caching=config.caching,
        )
    else:  # verify it is nifti and copy in into the workfolder for consistency with DICOM pipeline
        try:
            sitk.ReadImage(config.input)
        except Exception:
            print(f"Input file {config.input} is not a valid NIfTI file.")
            return

        nifti_in_processing_folder = os.path.join(output_path, nifti_name)
        shutil.copyfile(config.input, nifti_in_processing_folder)
        config.input = nifti_in_processing_folder

    run_nifti(config)


def get_threshold_volumes(
    depression_map: sitk.Image, thresholds: list[float]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    threshold_volumes: dict[str, float] = {}
    threshold_masks: dict[str, np.ndarray] = {}
    voxel_volume = np.prod(depression_map.GetSpacing()) / 1000.0  # in mm^3
    depression_arr = sitk.GetArrayViewFromImage(depression_map)

    for thold in thresholds:
        thold_str = f"{thold:.2f}"
        threshold_masks[thold_str] = (depression_arr >= thold).astype(np.uint8)
        threshold_volumes[thold_str] = (
            np.sum(threshold_masks[thold_str]) * voxel_volume
        )  # in ml

    return threshold_volumes, threshold_masks


def write_processing_summary(
    config: rNCCTConfig, output_path: str, lesion_volumes: dict[str, float]
) -> None:
    # write summary json file
    summary_file = os.path.join(output_path, "processing_summary.json")
    summary_data = {
        "input": config.input,
        "output": config.output,
        "caching": config.caching,
        "thin2thick": config.thin2thick,
        "debug": config.debug,
        "xy_std": config.xy_std,
        "colormap_range": config.colormap_range,
        "z_std": config.z_std,
        "max_accept_HU": config.max_accept_HU,
        "thresholds": config.thresholds,
        "lesion_volumes": lesion_volumes,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)


def run_nifti(config: rNCCTConfig) -> None:
    # Type guard: ensure input and output are set (already validated in run_config)
    assert config.input is not None and config.output is not None
    input_path = config.input
    output_path = config.output

    Path(output_path).mkdir(exist_ok=True, parents=True)

    origncct = sitk.ReadImage(input_path)
    ncct_use_lps = sitk.DICOMOrient(origncct, "LPS")
    # change to float
    ncct_use_lps = sitk.Cast(ncct_use_lps, sitk.sitkFloat32)

    # check if we need to downsample
    if config.thin2thick and ncct_use_lps.GetSpacing()[2] < 3.0:
        ncct_use_lps = volume_average_downsample(
            ncct_use_lps, outputfolder=output_path, cachemode=config.caching
        )

    # modify ncct coordinate system to orthonormal with origo (not origin) at voxel center - this will make flipping operations easier
    # and we will re-populate the output image with the original header
    ncct_use_lps_orthonormal_centered = sitk.Image(ncct_use_lps)
    ncct_use_lps_orthonormal_centered.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    current_origin = np.array(ncct_use_lps_orthonormal_centered.GetOrigin())
    image_center = sutl.image_center(ncct_use_lps_orthonormal_centered)
    new_origin = current_origin - image_center
    ncct_use_lps_orthonormal_centered.SetOrigin(new_origin)

    # Template reg for masking
    t2n_xfm, n2t_xfm = template_reg(
        origncct=ncct_use_lps_orthonormal_centered,
        fixedmask=sitk.ReadImage(ncct_paths.head_aura),
        outputfolder=Path(output_path),
        cachemode=config.caching,
        debug=config.debug,
        debug_prefix="STEP1_template_reg_",
        registration_parameters_url=ncct_paths.template_registration_parameters,
    )

    ipsi_brainmask = brainmasker_candidate(
        ncct_use_lps_orthonormal_centered,
        t2n_xfm,
        erodemaskloc,
        outputfolder=Path(output_path),
        cachemode=config.caching,
        debug=config.debug,
        debug_prefix="STEP2_brainmasking_",
    )

    # flipping
    mirror_brain, mirror_brainmask = flipped2self(
        ncct_use_lps_orthonormal_centered,
        ipsi_brainmask,
        t2n_xfm,
        n2t_xfm,
        skull_et_interior=head_aura,
        outputfolder=output_path,
        cachemode=config.caching,
        debug=config.debug,
        debug_prefix="STEP3_mirroring_",
    )

    csf_ipsi_soft = csf_seg(
        ncct_use_lps_orthonormal_centered,
        ipsi_brainmask,
        output_path,
        prefix="ipsi_",
        cachemode=config.caching,
    )
    csf_mirror_soft = csf_seg(
        mirror_brain,
        mirror_brainmask,
        output_path,
        prefix="mirror_",
        cachemode=config.caching,
        debug=config.debug,
        debug_prefix="STEP4_CSF_segmentation_",
    )

    ipsi_tissue_mask = refine_csf_mask(
        ncct_use_lps_orthonormal_centered,
        ipsi_brainmask,
        csf_ipsi_soft,
        output_path,
        prefix="ipsi_",
        cachemode=config.caching,
        maxval=None,
        debug=config.debug,
        debug_prefix="STEP5_Refine_CSF_mask_",
    )
    mirror_tissue_mask = refine_csf_mask(
        mirror_brain,
        mirror_brainmask,
        csf_mirror_soft,
        output_path,
        prefix="mirror_",
        cachemode=config.caching,
        maxval=None,
        debug=config.debug,
        debug_prefix="STEP6_Refine_CSF_mirrormask_",
    )

    # ORIG 41   #48 works better in 169 for example 45 mayube ok
    # ipsi smoothing
    ipsi_smooth = smooth(
        ipsi_tissue_mask,
        ncct_use_lps_orthonormal_centered,
        output_path,
        valid_value_range=[0, config.max_accept_HU],
        std1=config.xy_std,
        std2=config.z_std,
        prefix="ipsi_",
        cachemode=config.caching,
    )
    # contra smoothing
    mirror_smooth = smooth(
        mirror_tissue_mask,
        mirror_brain,
        output_path,
        valid_value_range=[0, config.max_accept_HU],
        std1=config.xy_std,
        std2=config.z_std,
        prefix="mirror_",
        cachemode=config.caching,
    )

    # calc ratios
    depression_map = calc_ratio(
        ncct_use_lps_orthonormal_centered,
        ipsi_smooth,
        mirror_smooth,
        ipsi_tissue_mask,
        mirror_tissue_mask,
        output_path,
        cachemode=config.caching,
    )

    # get lesion volumes and masks
    lesion_volumes, lesion_masks = get_threshold_volumes(
        depression_map, config.get_thresholds()
    )

    # rgb colormap overlay
    depression_map2rgb(
        depression_map,
        ncct_use_lps_orthonormal_centered,
        output_path,
        depression_range=config.get_colormap_range(),
        background_min_max=(0, 60),
    )

    # quantitative output with lesion masks and volumes
    lesion_masks2rgb(
        outputfolder=output_path,
        origncct=ncct_use_lps_orthonormal_centered,
        lesion_masks=lesion_masks,
        lesion_volumes=lesion_volumes,
        background_min_max=(0, 60),
    )

    write_processing_summary(config, output_path, lesion_volumes)
