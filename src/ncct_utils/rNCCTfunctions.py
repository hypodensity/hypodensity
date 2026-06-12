from __future__ import annotations

import datetime
import glob
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import imageio
import itk
import matplotlib as mpl
import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib import cm
from PIL import ImageDraw, ImageFont
from scipy.signal import convolve
from scipy.signal.windows import gaussian

import dicomutils.imageutils as imageutils
import regutils.simpleelastix_utils as sutl
from dicomutils.dicomutils import DicomSeries, sitk2generic_ct, dicomscan
from dicomutils.imageutils import sitk2montage
from ncct_utils.csf_segmentation_pytorch import csf_seg as csf_seg_pytorch


mpl.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from . import ncct_paths

BYPASSED_MSG = "Bypassed.."
debug: bool = True


def volume_average_downsample(
    img: sitk.Image, outputfolder: Path, cachemode: bool = False
) -> tuple[sitk.Image, float]:
    """
    Downsample volume by volume averaging. Hardcoded with the following behavior
    spacing >=3: assertion error
    spacing <3, integer factor getting closes to spacing of 5 we try integer factors of 2 to 10. We do not have anything less than 0.5mm
    """
    start_time = datetime.datetime.now()
    target_name = outputfolder / "downsampled.nii.gz"
    if cachemode and target_name.exists():
        return sitk.ReadImage(str(target_name)), (
            datetime.datetime.now() - start_time
        ).total_seconds()
    target_spacing = 5.0
    data_spacing = img.GetSpacing()[2]
    data_array = sitk.GetArrayFromImage(img)
    nslices = data_array.shape[0]
    factors = np.arange(2, 20).astype(np.float32)
    potential_spacing = data_spacing * factors

    minindex = np.argmin(np.abs(potential_spacing - target_spacing))

    factor_use = factors[minindex]

    new_spacing = factor_use * data_spacing

    # now let's volume average with the above factor
    sampling_intervals = np.arange(
        start=0, stop=nslices, step=factor_use, dtype=np.uint16
    )

    new_array = np.zeros_like(
        data_array,
        shape=(len(sampling_intervals) - 1, data_array.shape[1], data_array.shape[2]),
    )

    print(f"Input spacing: {data_spacing:2.2f}")
    print(f"output spacing: {new_spacing:2.2f} (factor={factor_use})")
    for sample_index in range(len(sampling_intervals) - 1):
        print(
            f"{sampling_intervals[sample_index]}-{sampling_intervals[sample_index + 1]}"
        )
        new_array[sample_index, :, :] = np.mean(
            data_array[
                sampling_intervals[sample_index] : sampling_intervals[sample_index + 1],
                :,
                :,
            ],
            axis=0,
        )

    # let the new origin be the midpoint of the bottom slab
    first_slice_bottom_slab_pos = img.GetOrigin()
    last_slice_bottom_slab_pos = img[:, :, int(sampling_intervals[1] - 1) :].GetOrigin()

    new_origin = 0.5 * (
        np.array(last_slice_bottom_slab_pos) - np.array(first_slice_bottom_slab_pos)
    ) + np.array(first_slice_bottom_slab_pos)

    new_img = sitk.GetImageFromArray(new_array)
    new_img.SetOrigin(
        new_origin.tolist()
    )  # technically, the new origin should probably be in the middle
    new_img.SetDirection(img.GetDirection())
    new_img.SetSpacing((img.GetSpacing()[0], img.GetSpacing()[1], float(new_spacing)))

    sitk.WriteImage(new_img, str(target_name))
    return new_img, (datetime.datetime.now() - start_time).total_seconds()


def verify_ncct_series_integrity(
    dicomseriesobject: list[DicomSeries],
) -> tuple[bool, str]:
    if len(dicomseriesobject) != 1:
        return (
            False,
            f"Expected a single dicom series, {len(dicomseriesobject)} found",
        )
    series = dicomseriesobject[0]
    if series.isophasic is False:  # we dont accept repeated slices
        return (False, "Repeated slices for same SliceLocations are not accepted")
    return (True, "")


def tilt_correct(dicom_series: DicomSeries) -> list[pydicom.dataset.Dataset]:
    """
    Read series from DICOM folder
    Tilt correct the series if needed
    Return the corrected dicom datasets as a list
    """
    datasets_by_ipp = [
        dicom_series.datasets[indx][0] for indx in dicom_series.sortorder[:, 0]
    ]

    ipps = np.array([ds.ImagePositionPatient for ds in datasets_by_ipp])

    slicetop_z = ipps[-1][2]
    slicebottom_z = ipps[0][2]
    midpoint = slicebottom_z + (slicetop_z - slicebottom_z) / 2
    minindx = np.argmin(np.abs(np.array(ipps)[:, 2] - midpoint))
    reference_zpos = ipps[minindx][2]  # spatial middle
    iop = np.array(datasets_by_ipp[0].ImageOrientationPatient).astype(np.float32)
    slice_perpendicular = np.cross(iop[0:3], iop[3:6])
    corner_to_corner_vec = ipps[1] - ipps[0]

    # if the slice perpendicular and the IPP to IPP vector are aligned, there is no tilt. and vice versa
    norm_dot = (
        np.round(
            np.dot(corner_to_corner_vec, slice_perpendicular)
            / (
                np.linalg.norm(corner_to_corner_vec)
                * np.linalg.norm(slice_perpendicular)
            )
            * 100000
        )
        / 100000
    )  # this is jsut a relative angle - no sign. We can grab the sign off the y component of the perpendicular
    iop_ipp_tilt = np.arccos(norm_dot) * np.sign(slice_perpendicular[1])

    if (
        np.abs(float(iop_ipp_tilt)) < 0.01
    ):  # 0.01 radian will result in displacement of ...
        return datasets_by_ipp

    displacements = np.sin(iop_ipp_tilt) * (ipps[:, 2] - reference_zpos)

    # make sure the DICOM are LPS (TODO)

    pixel_spacing: tuple[float, float] = (
        float(datasets_by_ipp[0].PixelSpacing[0]),
        float(datasets_by_ipp[0].PixelSpacing[1]),
    )
    for ipp, ds, cdisplacement in zip(
        ipps, datasets_by_ipp, displacements, strict=True
    ):
        pixel_data = ds.pixel_array

        # displacement along y cosine of cdisplacement mm
        ipp_corrected = ipp + cdisplacement * iop[3:6]
        ds.PixelData = displace_slice(
            pixel_data, pixel_spacing, cdisplacement
        ).tobytes()
        ds.ImagePositionPatient = [f"{ippc:.6f}" for ippc in ipp_corrected]

    return datasets_by_ipp


def dicom_datasets_to_itkimage(datasets: list[pydicom.dataset.Dataset]) -> itk.Image:
    # write dicoms to tempdir and use itk to read them ensuring correct metadata handling
    with tempfile.TemporaryDirectory() as tdir:
        for ds in datasets:
            ds.save_as(os.path.join(tdir, ds.SOPInstanceUID + ".dcm"))

        names_generator = itk.GDCMSeriesFileNames.New()
        names_generator.SetDirectory(tdir)
        series_uids = names_generator.GetSeriesUIDs()
        file_names = names_generator.GetFileNames(series_uids[0])
        return itk.imread(file_names)


def ncctdicom2volume(inputfiles: list[str], target: str) -> None:
    # read dicoms and store as nii file
    dicomseriesobjects = list(
        dicomscan(
            inputfiles, series_grouping_tags="SeriesInstanceUID", slicetolin=5
        ).values()
    )  # we dont care about tol here.    We performs slice by slice correction

    status, msg = verify_ncct_series_integrity(dicomseriesobjects)
    if status is False:
        raise ValueError(msg)

    tilt_corrected_dicom_datasets = tilt_correct(dicomseriesobjects[0])  #

    ncct_itk = dicom_datasets_to_itkimage(tilt_corrected_dicom_datasets)

    itk.imwrite(ncct_itk, target)


def make_kernel(std1=5, std2=5, thickness=5.0, inplane_res=0.45):
    # from gw 41, 50 / 15
    WINDOW_EDGE_GAUSS_MINIMUM = 0.10

    kernel99 = gaussian(99, std1 / inplane_res)  # std is 0.45 * 11 ~ 5

    firstvalid = np.argwhere(kernel99 >= WINDOW_EDGE_GAUSS_MINIMUM)[0][0]

    kernel_xy = kernel99[firstvalid:-firstvalid]

    kernel_z99 = gaussian(99, std2 / thickness)  # std ~ 5 (sinze 1xZ is 5mm
    firstvalid = np.argwhere(kernel_z99 >= WINDOW_EDGE_GAUSS_MINIMUM)[0][0]
    kernel_z = kernel_z99[firstvalid:-firstvalid]
    kernel_size_z = len(kernel_z)
    #
    #     [0.0039, 0.0066, 0.0111, 0.0181, 0.0286, 0.0439, 0.0657, 0.0956, 0.1353, 0.1863, 0.2494, 0.3247, 0.4111, 0.5063, 0.6065, 0.7066, 0.8007, 0.8825
    # 0.9460, 0.9862, 1.0000, 0.9862, 0.9460, 0.8825, 0.8007, 0.7066, 0.6065, 0.5063, 0.4111, 0.3247, 0.2494, 0.1863, 0.1353, 0.0956, 0.0657, 0.0439, 0.0286, 0.0181, 0.0111, 0.0066, 0.0039]

    kernel = np.reshape(kernel_xy, (len(kernel_xy), 1))
    kernel2d = np.matmul(kernel, kernel.T)  # now 2D

    kernel3d_rep = np.repeat(np.expand_dims(kernel2d, axis=2), kernel_size_z, axis=2)
    multvec = np.ones((1, 1, kernel_size_z))
    multvec[0, 0, :] = kernel_z
    kernel3d_zweighted = kernel3d_rep * multvec
    kernel3d_zweighted = kernel3d_zweighted / np.sum(kernel3d_zweighted)

    return kernel3d_zweighted


def smooth(
    mask,
    img,
    outputfolder: Path,
    valid_value_range=(0, 41),
    std1=5,
    std2=5,
    prefix="",
    cachemode=False,
) -> tuple[sitk.Image, float]:
    """changed oct 11 to calculate window sizes automatically based on std and voxelspacing"""
    start_time = datetime.datetime.now()
    target1 = outputfolder / f"{prefix}smoothed.nii.gz"
    print(f"Smoothing {prefix}")
    if cachemode and target1.exists():
        smoothed = sitk.ReadImage(str(target1))
        print(BYPASSED_MSG)
        return smoothed, (datetime.datetime.now() - start_time).total_seconds()

    inres = img.GetSpacing()
    # previously we used 0.4 0.4 5mm so we can use smaller kernel now   - prev 0.4*20  = 8 pix radius so will ju 8 radius now. Padding with 16 should do then

    # store this kernel in order to visually inspect it...

    # let us calculate the kernel windows based on

    kernel3d_zweighted = make_kernel(
        std1, std2, thickness=inres[2], inplane_res=inres[0]
    )

    mask_img = np.transpose(sitk.GetArrayFromImage(mask), axes=(1, 2, 0))

    brain_img = np.transpose(sitk.GetArrayFromImage(img), axes=(1, 2, 0))

    mask_img[
        np.logical_or(
            brain_img < valid_value_range[0], brain_img > valid_value_range[1]
        )
    ] = 0

    # now convolve mask+image with kernel
    # the convolved mask result should be divided up into the convolved image as acorrection factor - compare to classical output in terms of time and nummerical correspondence
    now = datetime.datetime.now()
    brain_c1 = convolve(
        brain_img * mask_img, kernel3d_zweighted, method="fft", mode="same"
    )
    mask_c1 = convolve(
        mask_img.astype(np.float32), kernel3d_zweighted, method="fft", mode="same"
    )

    mask_c1[mask_c1 < 0.001] = 0

    print(
        "Two convolutions took {} s".format(
            (datetime.datetime.now() - now).total_seconds()
        )
    )
    brain_corrected_vals = (
        brain_c1[mask_c1 > 0] / mask_c1[mask_c1 > 0]
    )  # below 0.1% of orig

    brain_corrected = brain_c1 * 0
    brain_corrected[mask_c1 > 0] = brain_corrected_vals
    brain_corrected[brain_corrected > 100] = 100
    brain_corrected[brain_corrected < 0] = 0
    itkordered = np.transpose(brain_corrected.astype(np.float32), axes=(2, 0, 1))
    smoothed = sutl.arr2img(itkordered, img)

    sitk2montage(
        sitkimg=img,
        opname=str(outputfolder / f"{prefix}smoothedAin.png"),
        value_range=(0, 60),
        every_nth_slice=1,
    )
    sitk2montage(
        sitkimg=smoothed,
        opname=str(outputfolder / f"{prefix}smoothedBsmoothedout.png"),
        value_range=(0, 60),
        every_nth_slice=1,
    )
    sitk2montage(
        sitkimg=mask,
        opname=str(outputfolder / f"{prefix}smoothedCmaskin.png"),
        value_range=(0, 1),
        every_nth_slice=1,
    )

    sitk.WriteImage(smoothed, str(target1))

    return smoothed, (datetime.datetime.now() - start_time).total_seconds()


def refine_csf_mask(
    ncct: sitk.Image,
    brainmask: sitk.Image,
    csf_soft: sitk.Image,
    outputfolder: Path,
    prefix: str = "",
    maxval: float | None = None,
    cachemode: bool = False,
    debug: bool = False,
) -> tuple[sitk.Image, float]:
    start_time = datetime.datetime.now()
    target = outputfolder / f"{prefix}refined_tissue_mask.nii.gz"
    print(f"Refining CSF mask {prefix}")
    if cachemode and target.exists():
        print(BYPASSED_MSG)
        return sitk.ReadImage(str(target)), (
            datetime.datetime.now() - start_time
        ).total_seconds()
    # Tissue is
    # inside brain mask
    # not in CSF mask (>0.25)
    # not above 41 HU

    ncct_arr = sitk.GetArrayFromImage(ncct)
    brainmask_arr = sitk.GetArrayFromImage(brainmask)
    csf_soft_arr = sitk.GetArrayFromImage(csf_soft)

    brainmask_arr_nocsf = np.logical_and(brainmask_arr, csf_soft_arr < 0.25)

    if maxval is not None:
        brainmask_no_hyperintensity_or_csf = np.logical_and(
            brainmask_arr_nocsf, ncct_arr < 41
        ).astype(np.uint8)
        brainmask_no_hyperintensity_or_csf_img = sutl.arr2img(
            brainmask_no_hyperintensity_or_csf, ncct
        )
    else:
        brainmask_no_hyperintensity_or_csf_img = sutl.arr2img(
            brainmask_arr_nocsf.astype(np.uint8), ncct
        )

    sitk.WriteImage(brainmask_no_hyperintensity_or_csf_img, str(target))

    if debug:
        sitk2montage(
            sitkimg=ncct,
            opname=str(outputfolder / f"{prefix}A_refinedcsfmask.png"),
            value_range=(0, 60),
            every_nth_slice=1,
        )
        sitk2montage(
            sitkimg=ncct,
            opname=str(outputfolder / f"{prefix}B_refinedcsfmask.png"),
            value_range=(0, 60),
            maskovl=sitk.GetArrayFromImage(brainmask_no_hyperintensity_or_csf_img),
            every_nth_slice=1,
        )

    return brainmask_no_hyperintensity_or_csf_img, (
        datetime.datetime.now() - start_time
    ).total_seconds()


def refine_mask(bmask, ncct, outputfolder, cachemode=False, fname=None):
    del ncct
    uchar3d_type = itk.Image[itk.UC, 3]

    if fname is None:
        target = outputfolder / "refined_bmask.nii.gz"
    else:
        target = outputfolder / fname
    if cachemode and target.exists():
        return sitk.ReadImage(str(target))

    bmask_itk = sutl.sitk2itk(bmask)
    # performs a 2 px in-plane erosion
    setype = itk.FlatStructuringElement[3]
    erode_type = itk.BinaryErodeImageFilter[uchar3d_type, uchar3d_type, setype]
    erode_filter = erode_type.New()
    ball = setype.Ball(3)
    rad = ball.GetRadius()
    rad[2] = 0
    ball.SetRadius(rad)
    erode_filter.SetKernel(ball)
    erode_filter.SetInput(bmask_itk)
    erode_filter.SetErodeValue(1)
    erode_filter.Update()
    tmpimg = erode_filter.GetOutput()
    bmask_sitk1 = sutl.itk2sitk(tmpimg)

    sitk.WriteImage(bmask_sitk1, str(target))

    return bmask_sitk1


def rescale_depression_map(
    depression_arr: np.ndarray,
    depression_range: tuple[float, float],
    target_range=(0, 1),
):
    """
    Rescale depression_arr to 0-1 range based on ratiorange.
    :param ratioimg_arr:
    :param ratiorange:
    :return:
    """

    imout = imageutils.imrescale(depression_arr, depression_range, target_range)

    return imout


def rescale_ratiomap(
    ratioimg_arr: np.ndarray,
    ratiorange: tuple[float, float],
    minimum_volume=None,
    minimumvolume_thold=None,
    target_range=(0, 1),
):
    del minimum_volume
    del minimumvolume_thold
    return rescale_depression_map(ratioimg_arr, ratiorange, target_range)


def rgbmerge_ratio_ncct(
    rncct01, ncct_img, ds=None, cropmask=None, rc=None, background_min_max=(0, 60)
):
    """
    Create a RGB overlay of the rNCCT01 map on top of the NCCT image.
    """

    jet = cm.jet(range(255))
    jet = np.concatenate((np.array([[0, 0, 0, 1]]), jet), axis=0)
    ncct_0_255 = np.uint8(255 * rncct01)
    ncct_0_255_yxz = sutl.reorder_yxz(ncct_0_255)

    rgbovl = np.zeros(
        (ncct_0_255_yxz.shape[0], ncct_0_255_yxz.shape[1], ncct_0_255_yxz.shape[2], 3)
    )

    for coord in np.argwhere(ncct_0_255_yxz > 0):
        val = ncct_0_255_yxz[coord[0], coord[1], coord[2]]
        rgbval = jet[val, 0:3]
        rgbovl[coord[0], coord[1], coord[2], :] = rgbval

    if cropmask is not None:
        rowmin, rowmax, colmin, colmax, slicemin, slicemax = (
            imageutils.get_mask_bounds_zyx(cropmask)
        )
        rgbovl = rgbovl[
            colmin : colmax + 1, rowmin : rowmax + 1, slicemin : slicemax + 1, :
        ]

    mont1 = imageutils.montage(rgbovl[:, :, :, 0], rc=rc)
    mont2 = imageutils.montage(rgbovl[:, :, :, 1], rc=rc)
    mont3 = imageutils.montage(rgbovl[:, :, :, 2], rc=rc)
    ovl_mont = np.stack((mont1, mont2, mont3), axis=2)

    if ds is not None:
        ovl_mont = ovl_mont[::ds, ::ds, :]

    ncct_montage = sitk2montage(
        sitkimg=ncct_img,
        value_range=background_min_max,
        every_nth_slice=1,
        rc=rc,
        ds=ds,
        cropmask=cropmask,
    )
    ovl_mont_lo = imageutils.rgbmaskonrgb(ncct_montage, 255 * ovl_mont)

    ovl_mont = (ovl_mont_lo).astype(np.uint8)

    ncct_arr = imageutils.imrescale(
        imin=sitk.GetArrayFromImage(ncct_img),
        from_range=background_min_max,
        to_range=(0, 1),
    )
    rgb_3dvolume = np.repeat(np.expand_dims(ncct_arr, -1), axis=-1, repeats=3)
    rgbovl_czxy = np.transpose(rgbovl, [2, 0, 1, 3])
    anycolor = np.repeat(np.sum(rgbovl_czxy, axis=3, keepdims=True), 3, 3)
    rgb_3dvolume[anycolor > 0] = rgbovl_czxy[anycolor > 0]
    return ovl_mont, rgb_3dvolume


def ratiomap2dcm(
    ratioimg,
    origncct,
    dcmtemplatefile,
    dcmoutfolder,
    ratiorange=(0.95, 0.85),
    minimum_volume=None,
    minimumvolume_thold=None,
):
    if os.path.exists(dcmoutfolder):
        shutil.rmtree(dcmoutfolder)

    os.makedirs(dcmoutfolder)

    ratioimg_arr = sitk.GetArrayFromImage(ratioimg)
    target_range = [100 * (1 - ratiorange[0]), 100 * (1 - ratiorange[1])]
    imout = rescale_ratiomap(
        ratioimg_arr,
        ratiorange,
        minimum_volume,
        minimumvolume_thold,
        target_range=target_range,
    )  # scales ratiorange to 0-1

    # DICOM wants to store integers, so lets multiply by 100 to get a better resolution

    # DICOM will do NEWVAL=slope*VALUE+intercept   . Value must be integer and we want 0 to map to 0 but >0 must map to
    imout100 = 100 * imout
    slope = 0.01
    if not os.path.exists(dcmoutfolder):
        os.makedirs(dcmoutfolder)
    dcmfile = dcmtemplatefile
    dcmhead = pydicom.read_file(dcmfile)
    studydate = dcmhead.StudyDate
    seriesdate = dcmhead.SeriesDate if "SeriesDate" in dcmhead else dcmhead.StudyDate
    studytime = dcmhead.StudyTime
    seriestime = dcmhead.SeriesTime if "SeriesTime" in dcmhead else dcmhead.StudyTime

    study_instance_uid = dcmhead.StudyInstanceUID
    frame_of_reference_uid = dcmhead.FrameOfReferenceUID

    ncctimg = sutl.ReadImage(origncct)

    sitk2generic_ct(
        ncctimg,
        dcmoutfolder,
        seriesheader={
            "SeriesDescription": "NCCT",
            "PatientID": dcmhead.PatientID,
            "PatientName": dcmhead.PatientName,
            "StudyDate": studydate,
            "SeriesDate": seriesdate,
            "StudyTime": studytime,
            "SeriesTime": seriestime,
            "FrameOfReferenceUID": frame_of_reference_uid,
            "StudyInstanceUID": study_instance_uid,
        },
    )

    pixelcount = np.sum(imout100 > 0)
    ratioimgint = sutl.ReadImage(sutl.arr2img(imout100, ratioimg))

    sitk2generic_ct(
        ratioimgint,
        dcmoutfolder,
        seriesheader={
            "SeriesDescription": "rNCCT",
            "PatientID": dcmhead.PatientID,
            "PatientName": dcmhead.PatientName,
            "StudyDate": studydate,
            "SeriesDate": seriesdate,
            "StudyTime": studytime,
            "SeriesTime": seriestime,
            "FrameOfReferenceUID": frame_of_reference_uid,
            "StudyInstanceUID": study_instance_uid,
            "RescaleSlope": slope,
            "RescaleIntercept": 0,
        },
    )

    return pixelcount


def quantitative_summary(
    depression_rgb: sitk.Image,
    ncct_bg: sitk.Image,
    cropmask: sitk.Image,
    output_folder,
    thresholds: str,
    value_range=tuple[float, float],
) -> None:
    del thresholds
    rows = ncct_bg.GetSize()[2]
    crop_mask = sitk.GetArrayFromImage(cropmask)
    ncct_row_mont = sitk2montage(
        sitkimg=ncct_bg, rc=(1, rows), cropmask=crop_mask, value_range=value_range
    )
    ncct_w_rgb_row_mont = sitk2montage(
        sitkimg=depression_rgb, rc=(1, rows), cropmask=crop_mask
    )

    merged = np.concatenate((ncct_row_mont, ncct_w_rgb_row_mont), axis=0)
    imageio.v2.imwrite(os.path.join(output_folder, "quant_summary.png"), merged)


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


def threshold_lesions(
    outputfolder: Path,
    depression_map: sitk.Image,
    origncct: sitk.Image,
    thresholds: list[float],
    background_min_max=(0, 60),
) -> dict[str, float]:
    """
    Create a RGB overlay of the lesion masks on top of the NCCT image.
    :param origncct: Original NCCT image
    :param depression_map: Depression map image
    :param thresholds: List of thresholds for lesion segmentation
    :param background_min_max: Min and max values for the background NCCT image
    :param outputfolder: Folder to save the output images
    """

    #

    # get lesion volumes and masks
    lesion_volumes, lesion_masks = get_threshold_volumes(depression_map, thresholds)

    assert len(lesion_volumes) < 5, "Just 4 thresholds supported for now"

    background_ncct_montage_rgb = imageutils.sitk2montage(
        sitkimg=origncct, opname=None, value_range=background_min_max
    )

    temporary_colormap = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

    for thold_index, (_thold_str, mask) in enumerate(lesion_masks.items()):
        mask_color = temporary_colormap[thold_index]
        rgb_mask = imageutils.np2montage(
            matin=mask * 0,
            opname=None,
            value_range=None,
            mask_mix=mask_color,
            maskovl=mask,
        )
        background_ncct_montage_rgb = imageutils.rgbmaskonrgb(
            background_ncct_montage_rgb, rgb_mask
        )

    # text
    background_ncct_montage_rgb_txt = add_volumes_to_montage(
        background_ncct_montage_rgb, lesion_volumes, colors=temporary_colormap
    )

    imageio.v2.imwrite(
        outputfolder / "masks_RGB_overlay.png",
        background_ncct_montage_rgb_txt,
    )

    return lesion_volumes


def add_volumes_to_montage(
    montage: np.ndarray, volumes: dict[str, float], colors: list[tuple[int, int, int]]
) -> np.ndarray:
    width = montage.shape[1]
    height = 110 * len(volumes)
    # create and image for PIL
    volume_image = Image.new("RGB", (width, height), (0, 0, 0))
    # get a drawing context
    drawing_context = ImageDraw.Draw(volume_image)

    font = ImageFont.truetype("UbuntuMono-R.ttf", size=82)

    square_rel_start = -80
    square_w = 70
    txt_increment = 85

    for indx, (thold, volume) in enumerate(volumes.items()):
        vol_str = f"{thold}%: {f'{volume:3.1f} ml':>8}"
        print(vol_str)
        drawing_context.text(
            xy=(int(3 * width / 8), 10 + indx * txt_increment),
            text=vol_str,
            fill=(255, 255, 255),
            font=font,
        )
        drawing_context.rectangle(
            xy=(
                (int(3 * width / 8) + square_rel_start, 10 + indx * txt_increment),
                (
                    int(3 * width / 8) + square_rel_start + square_w,
                    10 + indx * txt_increment + square_w,
                ),
            ),
            fill=colors[indx],
        )

    volume_image_arr = np.array(volume_image)
    return np.concatenate((montage, volume_image_arr), axis=0)


def depression_map2rgb(
    depression_img: sitk.Image,
    origncct: sitk.Image,
    outputfolder: Path | None,
    depression_range: tuple[float, float] = (5.0, 15.0),
    background_min_max: tuple[float, float] = (0, 60),
    ds=None,
    cropmask=None,
    rc=None,
):
    depression_arr = sitk.GetArrayFromImage(depression_img)

    imout = rescale_depression_map(
        depression_arr, depression_range
    )  # scales ratiorange to 0-1
    merged_rgb, rgb_3d_volume = rgbmerge_ratio_ncct(
        imout,
        origncct,
        ds=ds,
        cropmask=cropmask,
        rc=rc,
        background_min_max=background_min_max,
    )
    rgb_3d_img = sitk.GetImageFromArray(rgb_3d_volume)
    rgb_3d_img.CopyInformation(origncct)
    if outputfolder is not None:
        fig, ax = plt.subplots(
            figsize=(200 / 300, merged_rgb.shape[0] / 300),
            layout="constrained",
            dpi=300,
        )

        cmap = mpl.colormaps["jet"].resampled(256)
        norm = mpl.colors.Normalize(vmin=depression_range[0], vmax=depression_range[1])

        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="vertical",
            label="HU depression (%)",
        )
        cbar.ax.yaxis.label.set_color("white")

        cbar.ax.tick_params(colors="white", labelcolor="white")

        fig.patch.set_facecolor("black")
        # savefig fill render to file - quick and dirty and to be imprved later
        # put in a tempdir
        with tempfile.TemporaryDirectory() as tdir:
            plt.savefig(os.path.join(tdir, "colorbar_tmp.png"), dpi=300)
            plt.close(fig)
            cbar_arr = imageio.v2.imread(os.path.join(tdir, "colorbar_tmp.png"))[
                :, :, 0:3
            ]

        imageio.imwrite(
            outputfolder / "rNCCT_A.png",
            np.concatenate((merged_rgb, cbar_arr), axis=1),
        )

        ncct_rgb = sitk2montage(
            sitkimg=origncct,
            opname=None,
            value_range=background_min_max,
            ds=ds,
            cropmask=cropmask,
        )
        imageio.imwrite(
            outputfolder / "rNCCT_B.png",
            np.concatenate((ncct_rgb, 0 * cbar_arr), axis=1),
        )

        sitk.WriteImage(rgb_3d_img, str(outputfolder / "rncct_rgb.nii"))

    return merged_rgb, rgb_3d_img


def standardize_input(
    ncct_img: sitk.Image, output_path: Path, caching: bool = False
) -> tuple[sitk.Image, sitk.Image, float]:
    start_time = datetime.datetime.now()
    # convert to LPS and float32 if not already
    target1 = output_path / "input_lps.nii.gz"
    target2 = output_path / "input_lps_centered.nii.gz"
    if caching and target1.exists() and target2.exists():
        return (
            sitk.ReadImage(str(target1)),
            sitk.ReadImage(str(target2)),
            (datetime.datetime.now() - start_time).total_seconds(),
        )

    # do we need to convert to LPS?
    direction = ncct_img.GetDirection()
    current_orientation = (
        sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(direction)
    )

    if current_orientation != "LPS":
        ncct_use_lps = sitk.DICOMOrient(ncct_img, "LPS")
        # change to float
    else:
        ncct_use_lps = ncct_img

    if ncct_use_lps.GetPixelID() != sitk.sitkFloat32:
        ncct_use_lps = sitk.Cast(ncct_use_lps, sitk.sitkFloat32)

    # modify ncct coordinate system to orthonormal with origo (not origin) at voxel center - this will make flipping operations easier
    # and we will re-populate the output image with the original header
    ncct_use_lps_orthonormal_centered = sitk.Image(ncct_use_lps)
    ncct_use_lps_orthonormal_centered.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    current_origin = np.array(ncct_use_lps_orthonormal_centered.GetOrigin())
    image_center = sutl.image_center(ncct_use_lps_orthonormal_centered)
    new_origin = current_origin - image_center
    ncct_use_lps_orthonormal_centered.SetOrigin(new_origin)

    sitk.WriteImage(ncct_use_lps, target1)
    sitk.WriteImage(ncct_use_lps_orthonormal_centered, target2)
    return (
        ncct_use_lps,
        ncct_use_lps_orthonormal_centered,
        (datetime.datetime.now() - start_time).total_seconds(),
    )


def import_input(
    input_path: Path, output_path: Path, caching: bool = False
) -> tuple[sitk.Image, float]:
    start_time = datetime.datetime.now()
    nifti_name = "input_nifti.nii.gz"
    target = output_path / nifti_name
    if caching and target.exists():
        print(f"Input file {target} already exists. Using cached version.")
        return sitk.ReadImage(str(target)), (
            datetime.datetime.now() - start_time
        ).total_seconds()

    if input_path.is_dir():
        dicom2nifti(
            infolder=str(input_path), output_file_url=str(target), caching=caching
        )
    else:
        if not input_path.exists():
            raise ValueError(f"Input file {str(input_path)} does not exist")
        try:
            sitk.ReadImage(str(input_path))
        except Exception:
            raise ValueError(
                f"Input file {str(input_path)} could not be read"
            ) from None

        shutil.copyfile(str(input_path), target)

    return sitk.ReadImage(str(target)), (
        datetime.datetime.now() - start_time
    ).total_seconds()


def calc_ratio(
    ipsi_smooth,
    mirror_smooth,
    ipsi_tissue_mask,
    outputfolder: Path,
    cachemode=False,
):

    target = outputfolder / "pct_depression.nii.gz"
    if cachemode and target.exists():
        return sitk.ReadImage(str(target))
    print("calculating ratios")
    smoothed_ipsi_arr = sitk.GetArrayFromImage(ipsi_smooth)
    smoothed_mirror_arr = sitk.GetArrayFromImage(mirror_smooth)

    ipsi_tissue_mask_arr = sitk.GetArrayFromImage(ipsi_tissue_mask)

    samplingmask = ipsi_tissue_mask_arr

    ratios = smoothed_ipsi_arr[samplingmask > 0] / smoothed_mirror_arr[samplingmask > 0]

    # now lets convert ratios so they express percentage HU depression as a positive number in percent. Eg 0.9 becomes 10% depression

    depression = 100.0 * (1 - ratios)

    # clamp depression to max ratio 200 pct
    depression[depression > 200] = 200

    newimg = np.ones_like(smoothed_ipsi_arr, dtype=np.float32) * -np.inf
    newimg[samplingmask > 0] = depression

    depression_img = sutl.arr2img(newimg, ipsi_smooth)
    sitk.WriteImage(depression_img, str(target))

    return depression_img


def dicom2nifti(infolder: str, output_file_url: str, caching: bool = False):
    if caching and os.path.exists(output_file_url):
        return sitk.ReadImage(str(output_file_url))
    Path(os.path.dirname(output_file_url)).mkdir(parents=True, exist_ok=True)
    inputfiles = glob.glob(os.path.join(infolder, "*"))
    ncctdicom2volume(inputfiles, output_file_url)

    return output_file_url


def template_reg(
    *,
    origncct: sitk.Image,
    outputfolder: Path,
    registration_parameters_url: str,
    fixedmask: sitk.Image,
    cachemode: bool = False,
    template_l: str | None = None,
    debug: bool = False,
) -> tuple[itk.ParameterObject, itk.ParameterObject, float]:
    start_time = datetime.datetime.now()
    if template_l is None:
        template_l = ncct_paths.scct_unsmooth

    n2t_loc = outputfolder / "N2T.txt"
    t2n_loc = outputfolder / "T2N.txt"

    if cachemode and n2t_loc.exists() and t2n_loc.exists():
        n2t = itk.ParameterObject.New()
        t2n = itk.ParameterObject.New()
        n2t.ReadParameterFile(str(n2t_loc))
        t2n.ReadParameterFile(str(t2n_loc))
        return t2n, n2t, -1

    template, _ = sutl.imgnarray(template_l)

    # clamp ncct and template
    ncct = sitk.Clamp(origncct, lowerBound=0, upperBound=100)
    template = sitk.Clamp(template, lowerBound=0, upperBound=100)

    ncct_itk = sutl.sitk2itk(ncct, dtype=np.float32)
    template_itk = sutl.sitk2itk(template, dtype=np.float32)
    fixedmask_itk = None
    if fixedmask is not None:
        fixedmask_itk = sutl.sitk2itk(fixedmask, dtype=np.uint8)

    # NCCT TO TEMPLATE - USED FOR GUIDING MIRROR XFM
    retinfo = sutl.dti_affine(
        moving=ncct_itk,
        fixed=template_itk,
        iterationinfo=debug,
        custom_param_url=registration_parameters_url,
        fixedmask=fixedmask_itk,
    )

    n2t = retinfo["xfmmaps"]
    print(n2t.GetParameterMap(0))
    print(str(n2t_loc))
    itk.ParameterObject.WriteParameterFile(n2t.GetParameterMap(0), str(n2t_loc))

    # INVERT
    t2n = sutl.invert_dti_xfm_itk(ncct_itk, n2t)
    itk.ParameterObject.WriteParameterFile(t2n.GetParameterMap(0), str(t2n_loc))

    if debug:
        # show the template in nattive space to check the registration visually
        template_native_itk = sutl.itk_resample(template_itk, t2n)
        template_native = sutl.itk2sitk(template_native_itk)

        sitk2montage(
            sitkimg=ncct,
            opname=str(outputfolder / "A_native.png"),
            value_range=(0, 60),
        )
        sitk2montage(
            sitkimg=template_native,
            opname=str(outputfolder / "B_Tlnative.png"),
            value_range=(0, 60),
        )

    elapsed_time = datetime.datetime.now() - start_time
    return t2n, n2t, elapsed_time.total_seconds()


def plot_axis_on_image(
    axis_points: np.ndarray, moving_image: sitk.Image, outpng: str, flip_image=False
):

    voxel_points = np.zeros_like(axis_points, dtype=np.int32)
    for row in range(axis_points.shape[0]):
        voxel_points[row, :] = moving_image.TransformPhysicalPointToIndex(
            axis_points[row, :]
        )

    midslice = int(voxel_points[3, 2])
    native_ncct_slice_rgb = sitk2montage(
        sitkimg=moving_image[:, :, midslice : midslice + 1],
        opname=None,
        value_range=(0, 60),
    )
    if flip_image:
        native_ncct_slice_rgb = np.flip(native_ncct_slice_rgb, axis=1).astype(np.uint8)

    rot_img_slice_rgb = cv2.arrowedLine(
        native_ncct_slice_rgb,
        tuple(voxel_points[3, 0:2].astype(int)),
        tuple(voxel_points[0, 0:2].astype(int)),
        color=(255, 0, 0),
        thickness=6,
        tipLength=0.15,
    )
    rot_img_slice_rgb = cv2.arrowedLine(
        rot_img_slice_rgb,
        tuple(voxel_points[3, 0:2].astype(int)),
        tuple(voxel_points[1, 0:2].astype(int)),
        color=(0, 255, 0),
        thickness=6,
        tipLength=0.15,
    )
    rot_img_slice_rgb = cv2.arrowedLine(
        rot_img_slice_rgb,
        tuple(voxel_points[3, 0:2].astype(int)),
        tuple(voxel_points[2, 0:2].astype(int)),
        color=(0, 0, 255),
        thickness=6,
        tipLength=0.15,
    )
    imageio.v2.imwrite(outpng, rot_img_slice_rgb)


def get_transform_axis_from_dti_transform(
    n2t_xfm: itk.elxParameterObjectPython.elastixParameterObject,
    moving_image: sitk.Image,
    supress_affine: bool = False,
) -> np.ndarray:
    n2t_xfm_use = itk.ParameterObject.New()

    if supress_affine:
        # remove shear and scaling from the transform
        n2t_xfm_noshear = n2t_xfm.GetParameterMap(0)
        transform_parameters = list(n2t_xfm_noshear["TransformParameters"])
        transform_parameters[3] = "0.0"  # ShearX
        transform_parameters[4] = "0.0"  # ShearY
        transform_parameters[5] = "0.0"  # ShearZ
        transform_parameters[6] = "1.0"  # ScaleX
        transform_parameters[7] = "1.0"  # ScaleY
        transform_parameters[8] = "1.0"  # ScaleZ
        n2t_xfm_noshear["TransformParameters"] = transform_parameters
        n2t_xfm_use.AddParameterMap(n2t_xfm_noshear)
    else:
        n2t_xfm_use.AddParameterMap(n2t_xfm.GetParameterMap(0))

    orthonormal_axes_points = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [0, 0, 0]])
    transformed_orthonormal_axes_points = sutl.xform_point_set(
        orthonormal_axes_points, n2t_xfm_use, moving_image
    )
    # now devise an euler transform that maps a flipped version of the transformed points to the transformed points
    return np.array(transformed_orthonormal_axes_points)


def flip_axis(axis_points: np.ndarray) -> np.ndarray:
    # flip x coordinate and recalculate the x vector
    flipped_axis = axis_points.copy()

    flipped_axis[:, 0] = -flipped_axis[:, 0]
    # recalculate the x vector to be orthogonal to y and z using cross product
    vec_x = flipped_axis[0, :] - flipped_axis[3, :]
    vec_x_inverted = -vec_x
    x_inverted = vec_x_inverted + flipped_axis[3, :]

    flipped_axis[0, :] = x_inverted
    return flipped_axis


def estimate_rigid(p, q):
    # p, q: (N,3) matching points
    mu_p, mu_q = p.mean(axis=0), q.mean(axis=0)
    p0, q0 = p - mu_p, q - mu_q
    h = p0.T @ q0
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = mu_q - rotation @ mu_p
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = translation

    return affine


def init_affine_transform_from_template_reg(
    n2t_xfm: itk.elxParameterObjectPython.elastixParameterObject,
    ncct: sitk.Image,
    outputfolder: Path,
    debug: bool = False,
) -> itk.elxParameterObjectPython.mapstringvectorstring:
    """
    Given a DTI transform from NCCT to template, estimate an initial affine transform that maps the flipped NCCT to the NCCT.
    For convenience, the resampling grid in the transform is that of the NCCT
    """
    transformed_axis_points = get_transform_axis_from_dti_transform(
        n2t_xfm, ncct, supress_affine=True
    )

    axis_points_post_flip = flip_axis(transformed_axis_points)

    initial_transform = estimate_rigid(transformed_axis_points, axis_points_post_flip)

    affine_xfm = itk.ParameterObject.New()
    affine_xfm.ReadParameterFile(ncct_paths.affine_transform_template)

    affine_xfm_map = sutl.set_resampling_grid_on_transform(
        affine_xfm.GetParameterMap(0), ncct
    )
    affine_xfm_map["TransformParameters"] = [
        str(v) for v in initial_transform[0:3, :3].flatten(order="C")
    ] + [str(v) for v in initial_transform[0:3, 3]]

    if debug:
        plot_axis_on_image(
            transformed_axis_points,
            ncct,
            outpng=str(outputfolder / "pre_xfm_A_debug_transformed_axis.png"),
            flip_image=False,
        )

        plot_axis_on_image(
            axis_points_post_flip,
            ncct,
            outpng=str(outputfolder / "pre_xfm_B_debug_transformed_axis_flipped.png"),
            flip_image=True,
        )

    return affine_xfm_map


def flipped2self(
    origncct: sitk.Image,
    ipsi_brainmask: sitk.Image,
    t2n_xfm: itk.elxParameterObjectPython.elastixParameterObject,
    n2t_xfm: itk.elxParameterObjectPython.elastixParameterObject,
    skull_et_interior: str,
    outputfolder: Path,
    cachemode: bool = False,
    transformparameters_bspline_opts: dict | None = None,
    bsplinelevels: int = 3,
    transformparameters_affine_opts: dict | None = None,
    debug: bool = False,
) -> tuple[sitk.Image, sitk.Image, float]:
    start_time = datetime.datetime.now()

    target0 = outputfolder / "flip_to_self_init.txt"
    target1 = outputfolder / "flip_to_self_affine.txt"
    target1a = outputfolder / "flip_to_self_nlin.txt"
    target2 = outputfolder / "flipped_brain.nii"
    target3 = outputfolder / "flip2selfimg_mask.nii"
    target4 = outputfolder / "flipped_brain_usereg.nii"
    target5 = outputfolder / "ncct_usereg.nii"

    initial_affine_map = init_affine_transform_from_template_reg(
        n2t_xfm,
        origncct,
        outputfolder,
        debug=debug,
    )

    if (
        cachemode
        and target1.exists()
        and target1a.exists()
        and target2.exists()
        and target3.exists()
        and target4.exists()
        and target5.exists()
    ):
        flip2selfimg_full = sitk.ReadImage(str(target2))
        flip2selfimg_mask = sitk.ReadImage(str(target3))

        return (
            flip2selfimg_full,
            flip2selfimg_mask,
            (datetime.datetime.now() - start_time).total_seconds(),
        )

    # lets mask the NCCT to remove headholder
    skull_et_interior_like_n = sutl.itk_resample(itk.imread(skull_et_interior), t2n_xfm)

    skull_et_interior_like_n_arr = itk.GetArrayFromImage(skull_et_interior_like_n)

    # clamp ncct
    native_arr = sitk.GetArrayFromImage(origncct)
    tmparray = native_arr.copy()
    tmparray[tmparray < 0] = 0
    tmparray[tmparray > 100] = 100

    tmparray[skull_et_interior_like_n_arr < 0.99] = 0

    ncct = sutl.arr2img(tmparray, origncct)

    ncct_flipped_mat = np.flip(tmparray, axis=2)
    ncct_flipped_img = sutl.arr2img(ncct_flipped_mat, ncct)

    ncct_flipped_mat_fullrange = np.flip(native_arr, axis=2)
    ncct_flipped_img_fullrange = sutl.arr2img(ncct_flipped_mat_fullrange, ncct)

    ipsi_brainmask_flipped = sutl.arr2img(
        np.flip(sitk.GetArrayFromImage(ipsi_brainmask), axis=2), ncct
    )

    transformparameters_affine = {}

    transformparameters_affine["MaximumStepLength"] = [
        "10",
        "7",
        "5",
        "1",
    ]  # only for adaptive GD
    transformparameters_affine["ImagePyramidSchedule"] = [
        "8",
        "8",
        "2",
        "4",
        "4",
        "2",
        "2",
        "2",
        "2",
        "1",
        "1",
        "1",
    ]
    transformparameters_affine["MaximumNumberOfIterations"] = ["200"]

    transformparameters_affine["AutomaticScalesEstimation"] = [
        "true"
    ]  # overrides a lot of the stuff below
    transformparameters_affine["Scales"] = [
        str(k)
        for k in [1000, 1000, 1000, 10000, 10000, 10000, 10000, 10000, 10000, 1, 1, 1]
    ]
    transformparameters_affine["AutomaticTransformInitializationMethod"] = [
        "CenterOfGravity"
    ]
    transformparameters_affine["SP_a"] = ["10", "5", "2", "1"]
    transformparameters_affine["SP_alpha"] = ["0.8"]
    transformparameters_affine["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    transformparameters_affine["FinalBSplineInterpolationOrder"] = ["0"]

    if transformparameters_affine_opts is not None:
        for item, val in transformparameters_affine_opts.items():
            transformparameters_affine[item] = val

    if (
        transformparameters_bspline_opts is None
    ):  # we dont have any defaults here - this should probably be changed
        transformparameters_bspline = {}
    else:
        transformparameters_bspline = transformparameters_bspline_opts

    retdict = sutl.affine_nonlin(
        ncct_flipped_img,
        ncct,
        initial_xfm=initial_affine_map,
        transformparameters_affine=transformparameters_affine,
        transformparameters_bspline=transformparameters_bspline,
        bsplinelevels=bsplinelevels,
    )
    flip_to_self = retdict["xfmmaps"]

    init_map = flip_to_self.GetParameterMap(0)
    affine_map = flip_to_self.GetParameterMap(1)
    nlin_map = flip_to_self.GetParameterMap(2)
    nlin_map["FinalBSplineInterpolationOrder"] = ["0"]

    full_mirror = itk.ParameterObject.New()
    full_mirror.AddParameterMap(init_map)
    full_mirror.AddParameterMap(affine_map)
    full_mirror.AddParameterMap(nlin_map)

    to_affine_only = itk.ParameterObject.New()
    to_affine_only.AddParameterMap(init_map)
    to_affine_only.AddParameterMap(affine_map)

    init_only = itk.ParameterObject.New()
    init_only.AddParameterMap(init_map)

    non_lin_only = itk.ParameterObject.New()
    non_lin_only.AddParameterMap(nlin_map)

    # apply to full range image
    flip2selfimg_full = sutl.itk_resample(
        sutl.sitk2itk(ncct_flipped_img_fullrange), full_mirror
    )
    flip2selfmask_full = sutl.itk_resample(
        sutl.sitk2itk(ipsi_brainmask_flipped), full_mirror
    )

    flip2selfimg_to_affine_only = sutl.itk_resample(
        sutl.sitk2itk(ncct_flipped_img_fullrange), to_affine_only
    )
    flip2selfimg_init_only = sutl.itk_resample(
        sutl.sitk2itk(ncct_flipped_img_fullrange), init_only
    )

    itk.ParameterObject.WriteParameterFile(
        to_affine_only.GetParameterMap(0), str(target1)
    )
    itk.ParameterObject.WriteParameterFile(init_only.GetParameterMap(0), str(target0))
    itk.ParameterObject.WriteParameterFile(
        non_lin_only.GetParameterMap(0), str(target1a)
    )

    itk.imwrite(flip2selfimg_full, str(target2))
    itk.imwrite(flip2selfmask_full, str(target3))
    sitk.WriteImage(ncct_flipped_img, str(target4))
    sitk.WriteImage(ncct, str(target5))

    if debug:
        sitk2montage(
            sitkimg=ncct,
            opname=str(outputfolder / "A_flipped2self_self.png"),
            value_range=(0, 60),
        )
        sitk2montage(
            sitkimg=sutl.itk2sitk(flip2selfimg_full),
            opname=str(outputfolder / "B_flipped2self_mirror.png"),
            value_range=(0, 60),
        )
        shutil.copyfile(
            outputfolder / "A_flipped2self_self.png",
            outputfolder / "C_flipped2self_self.png",
        )
        sitk2montage(
            sitkimg=sutl.itk2sitk(flip2selfimg_to_affine_only),
            opname=str(outputfolder / "D_flipped2self_affinemirror.png"),
            value_range=(0, 60),
        )
        sitk2montage(
            sitkimg=sutl.itk2sitk(flip2selfimg_init_only),
            opname=str(outputfolder / "E_flipped2self_init.png"),
            value_range=(0, 60),
        )

    return (
        sutl.itk2sitk(flip2selfimg_full),
        sutl.itk2sitk(flip2selfmask_full),
        (datetime.datetime.now() - start_time).total_seconds(),
    )


# def brainmasker(origncct,template_l,T2Nxfm,erodemaskloc,outputfolder,prefix='',cachemode=False,dbg=None,  b = 100,a = -5,cs = 0.3,ps = 1.0):
#     '''

#     :param origncct:
#     :param template_l:
#     :param T2Nxfm:
#     :param erodemaskloc:
#     :param outputfolder:
#     :param prefix:
#     :param cachemode:
#     :return:
#     '''
#     # The above should be considered the initial affine starting point

#     # Now brain mask (or simply remove head-holder) and then run the non-linear reg

#     if cachemode and os.path.exists(target):


#     # smooth the eroded mask to give it a ramp (required by distance filter)

#     sitk2montage(sitkimg=origncct, opname=os.path.join(outputfolder,f"{prefix}noteroded.png"), value_range=(0, 60),
#                       maskovl=sitk.GetArrayFromImage(eroded_mask_native), mask_mix=(0, 255, 0))


#     itkmask = level_set_brainmask(rectamploc, outputfolder + '/{}erodedtmp.nii'.format(prefix), cid="johndoe",
#                                                     PS=ps,dbg=outputfolder)  # H5::DataSpaceIException, not when run outside of pycharm


#     sitk2montage(sitkimg=origncct, opname= os.path.join(outputfolder,f"{prefix}brainmaskedA.png"), value_range=(0, 60),
#                       mask_mix=[0, 255, 0])

#     sitk2montage(sitkimg=origncct, opname=os.path.join(outputfolder,f"{prefix}brainmaskedB.png"), value_range=(0, 60),
#                       maskovl=sitk.GetArrayFromImage(refined_mask), mask_mix=[0, 255, 0])


def brainmasker_candidate(
    origncct: sitk.Image,
    t2n_xfm: itk.elxParameterObjectPython.elastixParameterObject,
    erodemaskloc: str,
    outputfolder: Path,
    cachemode: bool = False,
    debug: bool = False,
    b=100,
    a=-5,
    cs=0.3,
    ps=1.0,
) -> tuple[sitk.Image, float]:
    """
    :param origncct:
    :param t2n_xfm:
    :param erodemaskloc:
    :param outputfolder:
    :param cachemode:
    :param debug:
    :param b: sigmoid beta parameter for level set brainmasking
    :param a: sigmoid alpha parameter for level set brainmasking
    :param cs: curvature scaling parameter for level set brainmasking
    :param ps: propagation scaling parameter for level set brainmasking
    :return: refined brain mask as a SimpleITK image
    """
    start_time = datetime.datetime.now()

    target = outputfolder / "refined_mask.nii"
    if cachemode and target.exists():
        refined_mask = sitk.ReadImage(str(target))
        return refined_mask, -1

    # create speed image from level set.
    ncct_arr = sitk.GetArrayFromImage(origncct).astype(np.float32)
    negative_values = ncct_arr[ncct_arr < 0]
    ncct_arr[ncct_arr < 0] = -10.0 * negative_values

    rectamp = sutl.arr2img(ncct_arr, origncct)

    direct_sitk = sitk.ReadImage(erodemaskloc)

    eroded_mask_native_itk = sutl.itk_resample(
        sutl.sitk2itk(direct_sitk, dtype=np.float32), t2n_xfm
    )
    eroded_mask_native = sutl.itk2sitk(eroded_mask_native_itk)
    # smooth the eroded mask to give it a ramp (required by distance filter)
    eroded_mask_native = sitk.DiscreteGaussian(eroded_mask_native, [1.0, 1.0, 1.0], 1)

    itkmask = level_set_brainmask(
        sutl.sitk2itk(rectamp),
        sutl.sitk2itk(eroded_mask_native),
        sigmoid_alpha=a,
        sigmoid_beta=b,
        use_gradient=False,
        curvature_scaling=cs,
        propagation_scaling=ps,
        outputfolder=outputfolder,
        debug=debug,
    )
    bmask_sitk = sutl.itk2sitk(itkmask)

    refined_mask = refine_mask(bmask_sitk, origncct, outputfolder, cachemode=cachemode)

    if debug:
        sitk2montage(
            sitkimg=origncct,
            opname=str(outputfolder / "A_eroded_start.png"),
            value_range=(0, 60),
            maskovl=sitk.GetArrayFromImage(eroded_mask_native),
            mask_mix=(0, 255, 0),
        )

        sitk2montage(
            sitkimg=origncct,
            opname=str(outputfolder / "B_brainmaskedA.png"),
            value_range=(0, 60),
            mask_mix=(0, 255, 0),
        )

        sitk2montage(
            sitkimg=origncct,
            opname=str(outputfolder / "C_brainmaskedB.png"),
            value_range=(0, 60),
            maskovl=sitk.GetArrayFromImage(refined_mask),
            mask_mix=(0, 255, 0),
        )

    sitk.WriteImage(refined_mask, str(target))

    return refined_mask, (datetime.datetime.now() - start_time).total_seconds()


def csf_seg(
    img: sitk.Image,
    mask: sitk.Image,
    outputfolder: Path,
    prefix: str = "",
    cachemode: bool = False,
    debug: bool = False,
) -> tuple[sitk.Image, float]:
    start_time = datetime.datetime.now()
    target = outputfolder / f"{prefix}csfsegmentation.nii.gz"
    print("CSF segmenting {}".format(prefix))
    if cachemode and target.exists():
        print(BYPASSED_MSG)
        return sitk.ReadImage(str(target)), -1

    predmat_img = csf_seg_pytorch(img=img, mask=mask)

    if debug:
        imageutils.sitk2montage(
            sitkimg=img,
            opname=str(outputfolder / f"{prefix}A_csfmask.png"),
            value_range=(0, 60),
        )
        imageutils.sitk2montage(
            sitkimg=img,
            opname=str(outputfolder / f"{prefix}B_csfmask.png"),
            value_range=(0, 60),
            maskovl=predmat_img,
        )

    sitk.WriteImage(predmat_img, str(target))

    return predmat_img, (datetime.datetime.now() - start_time).total_seconds()


def level_set_brainmask(
    inputimg: itk.Image,
    seedmask: itk.Image,
    outputfolder: Path,
    sigmoid_alpha=-2,
    sigmoid_beta=30,
    use_gradient=False,
    curvature_scaling=0.9,
    propagation_scaling=0.9,
    debug: bool = False,
):
    float3dtype = itk.Image[itk.F, 3]
    uchar3dtype = itk.Image[itk.UC, 3]
    floatwriter = itk.ImageFileWriter[float3dtype].New()

    thresholdingfiltertype = itk.BinaryThresholdImageFilter[float3dtype, uchar3dtype]

    def write_floatimg(img, name):
        floatwriter.SetFileName(name)
        floatwriter.SetInput(img)
        floatwriter.Update()

    if debug:

        def my_command():
            if levelsetfilter.GetElapsedIterations() % 25 == 0:
                k = levelsetfilter.GetElapsedIterations()
                thresholder = thresholdingfiltertype.New()
                thresholder.SetLowerThreshold(-1000.0)
                thresholder.SetUpperThreshold(0.0)
                thresholder.SetOutsideValue(0)
                thresholder.SetInsideValue(1)
                thresholder.SetInput(levelsetfilter)
                thresholder.Update()
                print(levelsetfilter.GetMTime())
                print(thresholder.GetMTime())
                print("{} ----".format(k))

                img = thresholder.GetOutput()
                sutl.sitk2montage(
                    sitkimg=sutl.itk2sitk(inputimg),
                    opname=str(outputfolder / f"IM_{k:04d}.png"),
                    value_range=[0, 100],
                    maskovl=itk.GetArrayFromImage(img),
                    mask_mix=[0, 255, 0],
                )

    sigma = 1

    maximum_rmse = 0.002
    number_of_iterations = 250

    maskwriter = itk.ImageFileWriter[uchar3dtype].New()

    mask_distancemap = itk.ApproximateSignedDistanceMapImageFilter[
        float3dtype, float3dtype
    ].New()

    mask_distancemap.SetInput(seedmask)

    mask_distancemap.SetInsideValue(1.0)
    mask_distancemap.SetOutsideValue(0)

    mask_distancemap.Update()

    if debug:
        write_floatimg(
            mask_distancemap.GetOutput(),
            str(outputfolder / "mask_distance.mha"),
        )

    if use_gradient:
        smoothingfiltertype = itk.CurvatureAnisotropicDiffusionImageFilter[
            float3dtype, float3dtype
        ]
        smoothing = smoothingfiltertype.New()
        smoothing.SetTimeStep(0.02)
        smoothing.SetNumberOfIterations(10)
        smoothing.SetConductanceParameter(100.0)
        smoothing.SetInput(inputimg)
        gradientfiltertype = itk.GradientMagnitudeRecursiveGaussianImageFilter[
            float3dtype, float3dtype
        ]
        gradientmagnitude = gradientfiltertype.New()
        gradientmagnitude.SetSigma(sigma)
        gradientmagnitude.SetInput(smoothing.GetOutput())
        #
        if debug:
            write_floatimg(
                smoothing.GetOutput(),
                str(outputfolder / "smoothed.mha"),
            )
            write_floatimg(
                gradientmagnitude.GetOutput(),
                str(outputfolder / "gradientmag.mha"),
            )

    sigmoidfiltertype = itk.SigmoidImageFilter[float3dtype, float3dtype]

    # define change interval os the x axis interval that maps from 0-1. In the regular form that is about 14 1/(1+np.exp(7))=0.0009110511944006454
    # so
    sigmoid = sigmoidfiltertype.New()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(sigmoid_alpha)
    sigmoid.SetBeta(sigmoid_beta)
    # use HU to generate sigmoid 0-50  maps to 1-0

    sigmoid.SetInput(inputimg)

    if debug:
        write_floatimg(sigmoid.GetOutput(), str(outputfolder / "sigmoid.mha"))
    # preprocess NCCT

    shapedetectionlevelsetimagefiltertype = itk.ShapeDetectionLevelSetImageFilter[
        float3dtype, float3dtype, itk.F
    ]
    levelsetfilter = shapedetectionlevelsetimagefiltertype.New()
    levelsetfilter.SetCoordinateTolerance(0.001)
    levelsetfilter.SetPropagationScaling(propagation_scaling)
    levelsetfilter.SetCurvatureScaling(curvature_scaling)
    levelsetfilter.SetMaximumRMSError(maximum_rmse)
    levelsetfilter.SetNumberOfIterations(number_of_iterations)
    levelsetfilter.SetInput(mask_distancemap.GetOutput())
    levelsetfilter.SetFeatureImage(sigmoid.GetOutput())

    thresholder = thresholdingfiltertype.New()
    thresholder.SetLowerThreshold(-1000.0)
    thresholder.SetUpperThreshold(0.0)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(1)
    thresholder.SetInput(levelsetfilter.GetOutput())

    if debug:
        levelsetfilter.AddObserver(itk.ProgressEvent(), my_command)

    if debug:
        maskwriter.SetInput(thresholder.GetOutput())
        maskwriter.SetFileName(str(outputfolder / "dbgmaskout.mha"))
        maskwriter.Update()

    thresholder.Update()
    maskimg = thresholder.GetOutput()
    return maskimg


def displace_slice(
    matin: np.ndarray, steps: tuple[float, float], displacement: float
) -> np.ndarray:
    # create sitk image with set step, origo does not matter
    # create a translation xfm with NN or spline interp
    # excute
    # convert to numpy array and return 2D array

    imgin = itk.GetImageFromArray(matin)
    imgin.SetSpacing(steps)

    transformiximagefilter = itk.TransformixFilter.New(imgin)

    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(
        os.path.join(ncct_paths.two_d_transform_template)
    )
    translation_map = parameter_object.GetParameterMap(0)

    translation_map["Size"] = [str(matin.shape[1]), str(matin.shape[0])]
    translation_map["Spacing"] = [str(steps[0]), str(steps[1])]
    translation_map["TransformParameters"] = [str(0.0), str(displacement)]
    translation_map["FinalBSplineInterpolationOrder"] = ["0"]
    parameter_object.SetParameterMap(0, translation_map)
    transformiximagefilter.SetTransformParameterObject(parameter_object)
    transformiximagefilter.UpdateLargestPossibleRegion()

    result = transformiximagefilter.GetOutput()
    return itk.GetArrayFromImage(result)
