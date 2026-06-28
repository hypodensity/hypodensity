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
from pathlib import Path

import SimpleITK as sitk

from ncct_utils import ncct_paths
from ncct_utils.rncct_config import rNCCTConfig
from ncct_utils.rNCCTfunctions import (
    brainmasker_candidate,
    calc_ratio,
    csf_seg,
    depression_map2rgb,
    import_input,
    standardize_input,
    flipped2self,
    threshold_lesions,
    refine_csf_mask,
    smooth,
    template_reg,
    brainmasker_unet,
    volume_average_downsample,
)

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

    process(config)


def write_processing_summary(
    config: rNCCTConfig,
    output_path: str,
    lesion_volumes: dict[str, float],
    elapsed_times: dict[str, float],
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
        "elapsed_times": elapsed_times,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)


def process(config: rNCCTConfig) -> None:
    stage_names = [
        "import",
        "standard_orientation",
        "downsample",
        "template_reg",
        "brainmasking",
        "mirroring",
        "csf_seg",
        "refine_csf_mask",
        "smooth",
        "ratio",
        "depression_rgb",
        "lesion_masks",
    ]
    sd = {
        s: Path(config.output) / f"{indx:02d}_{s}" for indx, s in enumerate(stage_names)
    }
    for d in sd.values():
        d.mkdir(exist_ok=True, parents=True)
    elapsed_times = {}

    # Type guard: ensure input and output are set (already validated in run_config)
    assert config.input is not None and config.output is not None

    output_path = config.output

    Path(output_path).mkdir(exist_ok=True, parents=True)

    ncct_imported, elapsed_times["import"] = import_input(
        Path(config.input), output_path=sd["import"], caching=config.caching
    )

    ncct_lps, ncct_lps_centered, elapsed_times["standard_orientation"] = (
        standardize_input(ncct_imported, sd["standard_orientation"], config.caching)
    )

    # check if we need to downsample
    if config.thin2thick and ncct_lps_centered.GetSpacing()[2] < 3.0:
        ncct_lps_centered, elapsed_times["downsample"] = volume_average_downsample(
            ncct_lps_centered, outputfolder=sd["downsample"], cachemode=config.caching
        )

    # Template reg for masking
    t2n_xfm, n2t_xfm, elapsed_times["template_reg"] = template_reg(
        origncct=ncct_lps_centered,
        fixedmask=sitk.ReadImage(str(ncct_paths.head_aura)),
        outputfolder=sd["template_reg"],
        cachemode=config.caching,
        debug=config.debug,
        registration_parameters_url=ncct_paths.template_registration_parameters,
    )

    if config.unet_brainmask:
        model_path = config.unet_brainmask_model_path or ncct_paths.unet_brainmask_model
        if model_path is None:
            raise ValueError(
                "UNet brain masking requires a model path. "
                "Set --unet-brainmask-model or NCCT_UNET_BRAINMASK_MODEL."
            )
        ipsi_brainmask, elapsed_times["brainmasking"] = brainmasker_unet(
            ncct_lps_centered,
            outputfolder=sd["brainmasking"],
            model_path=model_path,
            cachemode=config.caching,
        )
    else:
        ipsi_brainmask, elapsed_times["brainmasking"] = brainmasker_candidate(
            ncct_lps_centered,
            t2n_xfm,
            erodemaskloc,
            outputfolder=sd["brainmasking"],
            cachemode=config.caching,
            debug=config.debug,
        )

    # flipping
    mirror_brain, mirror_brainmask, elapsed_times["mirroring"] = flipped2self(
        ncct_lps_centered,
        ipsi_brainmask,
        t2n_xfm,
        n2t_xfm,
        skull_et_interior=head_aura,
        outputfolder=sd["mirroring"],
        cachemode=config.caching,
        debug=config.debug,
    )

    csf_ipsi_soft, elapsed_times["csf_seg_ipsi"] = csf_seg(
        ncct_lps_centered,
        ipsi_brainmask,
        sd["csf_seg"],
        prefix="ipsi_",
        cachemode=config.caching,
    )

    csf_mirror_soft, elapsed_times["csf_seg_contra"] = csf_seg(
        mirror_brain,
        mirror_brainmask,
        sd["csf_seg"],
        prefix="mirror_",
        cachemode=config.caching,
        debug=config.debug,
    )

    ipsi_tissue_mask, elapsed_times["refine_csf_mask_ipsi"] = refine_csf_mask(
        ncct_lps_centered,
        ipsi_brainmask,
        csf_ipsi_soft,
        sd["refine_csf_mask"],
        prefix="ipsi_",
        cachemode=config.caching,
        maxval=None,
        debug=config.debug,
    )
    mirror_tissue_mask, elapsed_times["refine_csf_mask_contra"] = refine_csf_mask(
        mirror_brain,
        mirror_brainmask,
        csf_mirror_soft,
        sd["refine_csf_mask"],
        prefix="mirror_",
        cachemode=config.caching,
        maxval=None,
        debug=config.debug,
    )

    ipsi_smooth, elapsed_times["smooth_ipsi"] = smooth(
        ipsi_tissue_mask,
        ncct_lps_centered,
        sd["smooth"],
        valid_value_range=[0, config.max_accept_HU],
        std1=config.xy_std,
        std2=config.z_std,
        prefix="ipsi_",
        cachemode=config.caching,
    )
    mirror_smooth, elapsed_times["smooth_contra"] = smooth(
        mirror_tissue_mask,
        mirror_brain,
        sd["smooth"],
        valid_value_range=[0, config.max_accept_HU],
        std1=config.xy_std,
        std2=config.z_std,
        prefix="mirror_",
        cachemode=config.caching,
    )

    # we are done with coordinte related processing. Lets get the original directions put back in
    ncct_lps_original_origo = sitk.Image(ncct_lps_centered)
    ncct_lps_original_origo.CopyInformation(ncct_lps)
    ipsi_smooth.CopyInformation(ncct_lps)
    mirror_smooth.CopyInformation(ncct_lps)
    ipsi_tissue_mask.CopyInformation(ncct_lps)
    mirror_tissue_mask.CopyInformation(ncct_lps)
    # calc ratios
    depression_map = calc_ratio(
        ipsi_smooth,
        mirror_smooth,
        ipsi_tissue_mask,
        sd["ratio"],
        cachemode=config.caching,
    )

    # rgb colormap overlay
    depression_map2rgb(
        depression_map,
        origncct=ncct_lps_original_origo,
        outputfolder=sd["depression_rgb"],
        depression_range=config.get_colormap_range(),
        background_min_max=(0, 60),
    )

    # quantitative output with lesion masks and volumes
    lesion_volumes = threshold_lesions(
        outputfolder=sd["lesion_masks"],
        depression_map=depression_map,
        origncct=ncct_lps_original_origo,
        thresholds=config.get_thresholds(),
        background_min_max=(0, 60),
    )

    write_processing_summary(config, output_path, lesion_volumes, elapsed_times)
