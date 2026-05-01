from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile
import time
from importlib import resources
import itk
import itk.elxParameterObjectPython
import numpy as np
import SimpleITK as sitk
from dicomutils import imageutils

datafolder = str(resources.files("regutils").joinpath("data"))
curpath = datafolder
sitk2montage = imageutils.sitk2montage


class sitkimgclass:
    def __init__(self, imgname):
        if isinstance(imgname, str):
            if imgname.find("*"):
                imgname = glob.glob(imgname)[0]
            self.img, self.arr = imgnarray(imgname)
        else:  # assumeimage
            self.img = imgname
            self.arr = sitk.GetArrayFromImage(imgname)

    def setarr(self, arr):
        self.arr = arr
        self.img = arr2img(arr, self.img)

    def update_arr(self):
        self.img = arr2img(self.arr, self.img)


# def ReadImage(infile):
#     return sitkimgclass(infile)


# def invert_euler(xfm, img):
#     tdir = tempfile.mkdtemp()
#     xfmloc = tdir + "/xfm.txt"
#     imgloc = tdir + "/img.nii"
#     oploc = tdir + "/xfmout.txt"
#     sitk.WriteParameterFile(xfm[0], tdir + "/xfm.txt")
#     sitk.WriteImage(img, tdir + "/img.nii")
#     elx_invert_cmd = "/usr/local/elastix/bin/elxInvertTransform"
#     assert os.path.exists(elx_invert_cmd)
#     osexec(f"elx_invert_cmd -tp {xfmloc} -out {oploc} -m {imgloc}")
#     xfmout = sitk.ReadParameterFile(oploc)

#     shutil.rmtree(tdir)
#     return xfmout


# def invert_rigid_xfm(img, xfm):

#     # this is not working well

#     tdir = tempfile.mkdtemp()
#     xfmloc = os.path.join(tdir, "tmp.txt")
#     sitk.WriteParameterFile(xfm, xfmloc)

#     params = {
#         "Metric": ["DisplacementMagnitudePenalty"],
#         "HowToCombineTransforms": ["Compose"],
#         "InitialTransformParametersFileName": [xfmloc],
#     }
#     # now obtain inverse
#     retdict_d = rigidA2B(img, img, transformparameters=params, customxfm=None)

#     inverse_xfm = retdict_d["xfmmaps"]
#     retdict_d["moving_l_fixed"]
#     retdict_d["iterationinfo"]

#     del inverse_xfm[0]["InitialTransformParametersFileName"]
#     shutil.rmtree(tdir)

#     return inverse_xfm


# def get_euler_template():
#     template = sitk.ReadParameterFile(f"{datafolder}/3dxfmtemplate.txt")
#     template["TransformParameters"] = [str(k) for k in [0, 0, 0, 0, 0, 0]]
#     return template


# def invert_DTI_xfm(img: itk.Image, xfm: itk.elxParameterObject):
#     tdir = tempfile.mkdtemp()
#     xfmloc = os.path.join(tdir, "tmp.txt")
#     sitk.WriteParameterFile(xfm, xfmloc)

#     params = {
#         "Metric": ["DisplacementMagnitudePenalty"],
#         "HowToCombineTransforms": ["Compose"],
#         "InitialTransformParametersFileName": [xfmloc],
#     }
#     # now obtain inverse
#     retdict = DTIaffineA2B(img, img, transformparameters=params)

#     inverse_xfm = retdict["xfmmaps"]
#     retdict["moving_l_fixed"]
#     retdict["iterationinfo"]

#     if "InitialTransformParametersFileName" in inverse_xfm[0]:
#         del inverse_xfm[0]["InitialTransformParametersFileName"]

#     if "InitialTransformParameterFileName" in inverse_xfm[0]:
#         del inverse_xfm[0]["InitialTransformParameterFileName"]

#     shutil.rmtree(tdir)
#     return inverse_xfm


def invert_dti_xfm_itk(
    img: itk.Image, xfm: itk.elxParameterObject
) -> itk.elxParameterObject:
    assert xfm.GetNumberOfParameterMaps() == 1, (
        "Currently only supports single parameter map for DTI affine inversion"
    )
    tdir = tempfile.mkdtemp()
    xfmloc = os.path.join(tdir, "tmp.txt")
    itk.ParameterObject.WriteParameterFile(xfm.GetParameterMap(0), xfmloc)

    params = {
        "Metric": ["DisplacementMagnitudePenalty"],
        "HowToCombineTransforms": ["Compose"],
    }
    #   "InitialTransformParametersFileName":[xfmloc]}
    # now obtain inverse
    retdict = dti_affine(
        img, img, transformparameters=params, initial_xfm_filename=xfmloc
    )

    inverse_xfm = retdict["xfmmaps"]

    if "InitialTransformParametersFileName" in inverse_xfm.GetParameterMap(0):
        inverse_xfm.SetParameter(0, "InitialTransformParametersFileName", [""])

    if "InitialTransformParameterFileName" in inverse_xfm.GetParameterMap(0):
        inverse_xfm.SetParameter(0, "InitialTransformParameterFileName", "")

    shutil.rmtree(tdir)
    return inverse_xfm


# def invert_DTI_nlin_xfm(img, xfm, parammap):
#     """
#     Implemented using cmd line since the initital transform functionality is buggy in SE
#     :param img:
#     :param xfm: holds the transforms
#     :param parammap:  holds the parameter files for affine and non lin
#     :return:
#     """

#     tdir = tempfile.mkdtemp()
#     affine_param = sitk.SimpleITK.ParameterMap(parammap[0])
#     bspline_param = sitk.SimpleITK.ParameterMap(parammap[1])
#     affine_param["InitialTransformParametersFileName"] = [""]
#     bspline_param["InitialTransformParametersFileName"] = [""]

#     # now mod these xfms for inversion estimation - not sure if needed for both really
#     affine_param["HowToCombineTransforms"] = ["Compose"]
#     bspline_param["HowToCombineTransforms"] = ["Compose"]
#     affine_param["Metric"] = ["DisplacementMagnitudePenalty"]
#     bspline_param["Metric"] = ["DisplacementMagnitudePenalty"]
#     bspline_param["InitialTransformParametersFileName"] = [f"{tdir}/affine_disp.txt"]
#     sitk.WriteParameterFile(affine_param, f"{tdir}/affine_disp.txt")
#     sitk.WriteParameterFile(bspline_param, f"{tdir}/bspline_disp.txt")

#     affine_xfm = sitk.SimpleITK.ParameterMap(xfm[0])
#     bspline_xfm = sitk.SimpleITK.ParameterMap(xfm[1])
#     affine_xfm["InitialTransformParametersFileName"] = [""]
#     bspline_xfm["InitialTransformParametersFileName"] = [f"{tdir}/affine_xfm.txt"]

#     sitk.WriteParameterFile(affine_xfm, f"{tdir}/affine_xfm.txt")
#     sitk.WriteParameterFile(bspline_xfm, f"{tdir}/bspline_xfm.txt")

#     sitk.WriteImage(img, f"{tdir}/Timg.nii")

#     os.makedirs(f"{tdir}/est")
#     os.makedirs(f"{tdir}/trans")

#     osexec(
#         f"elastix -t0 {tdir}/bspline_xfm.txt -p {tdir}/affine_disp.txt -p {tdir}/bspline_disp.txt -m {tdir}/Timg.nii -f {tdir}/Timg.nii -out {tdir}/est"
#     )

#     # read in the estimated XFMs
#     xfm0 = sitk.ReadParameterFile(f"{tdir}/est/TransformParameters.0.txt")
#     xfm1 = sitk.ReadParameterFile(f"{tdir}/est/TransformParameters.1.txt")
#     xfm0["InitialTransformParametersFileName"] = ["NoInitialTransform"]
#     xfm1["InitialTransformParametersFileName"] = ["NoInitialTransform"]
#     osexec("rm -rf " + tdir)
#     return [xfm0, xfm1]


# def invert_nlin_xfm(img, xfm, parammap):
#     tdir = tempfile.mkdtemp()
#     xfmloc0 = os.path.join(tdir, "tmp0.txt")  # this will be the transform to counteract
#     if "InitialTransformParametersFileName" in xfm:
#         del xfm["InitialTransformParametersFileName"]
#     sitk.WriteParameterFile(xfm, xfmloc0)

#     bspline_param = sitk.SimpleITK.ParameterMap(parammap)

#     bspline_param["Metric"] = ["DisplacementMagnitudePenalty"]
#     bspline_param["HowToCombineTransforms"] = ["Compose"]
#     bspline_param["InitialTransformParametersFileName"] = [xfmloc0]
#     # now obtain inverse
#     inverse_xfm, resampled_img = nonlinA2B(img, img, parameterMap=bspline_param)
#     del inverse_xfm[0]["InitialTransformParametersFileName"]
#     osexec("rm -rf " + tdir)
#     return inverse_xfm


def sitk2itk(sitkimg: sitk.Image, dtype: type | None = None) -> itk.Image:
    if dtype is None:
        dtype = sitk.GetArrayViewFromImage(sitkimg).dtype
    itkimg = itk.GetImageFromArray(sitk.GetArrayFromImage(sitkimg).astype(dtype))
    spacing = sitkimg.GetSpacing()
    direction = sitkimg.GetDirection()
    origin = sitkimg.GetOrigin()
    itkimg.SetSpacing(spacing)
    itkimg.SetOrigin(origin)
    direction_np = np.array(
        [list(direction[0:3]), list(direction[3:6]), list(direction[6:9])]
    )
    np_dir_vnl = itk.GetVnlMatrixFromArray(direction_np)
    direction_ref = itkimg.GetDirection()
    direction_ref.GetVnlMatrix().copy_in(np_dir_vnl.data_block())
    itkimg.SetDirection(direction_ref)
    return itkimg


def xform_point_set(
    points: np.ndarray, xform: itk.ElastixParameterObject, moving: sitk.Image
) -> list[np.ndarray]:

    tdir = tempfile.mkdtemp()
    npoints = len(points)
    with open(f"{tdir}/coords.pts", "w") as f:
        f.write(f"point\n{npoints}\n")
        for k in points:
            f.write(f"{k[0]} {k[1]} {k[2]}\n")

    moving_itk = sitk2itk(moving, dtype=np.int16)
    transformiximagefilter = itk.TransformixFilter.New(moving_itk)

    transformiximagefilter.SetOutputDirectory(tdir)
    transformiximagefilter.SetTransformParameterObject(xform)
    transformiximagefilter.SetFixedPointSetFileName(f"{tdir}/coords.pts")

    transformiximagefilter.UpdateLargestPossibleRegion()

    txt = open(os.path.join(tdir, "outputpoints.txt")).read()

    m = re.findall("OutputPoint = \[ ([0-9.-]+ [0-9.-]+ [0-9.-]+)", txt)
    new_coords = [np.fromstring(k, sep=" ") for k in m]
    shutil.rmtree(tdir)
    return new_coords


# def get_edges(sutlimg, iterations=3):
#     uint8mask = reorder_yxz(sutlimg.arr)

#     erodedarr = uint8mask * 0
#     for islice in range(uint8mask.shape[2]):
#         erodedarr[:, :, islice] = scipy.ndimage.morphology.binary_erosion(
#             uint8mask[:, :, islice], iterations=iterations
#         )

#     edges = (uint8mask - erodedarr).astype(np.uint8).copy()

#     return ReadImage(arr2img(reorder_zxy(edges), sutlimg.img))


def reorder_yxz(matin):
    return np.transpose(matin, axes=(1, 2, 0))


def reorder_zxy(matin):
    return np.transpose(matin, axes=(2, 0, 1))


def itk2sitk(itkimg):
    itk_direction = itkimg.GetDirection()
    itk_origin = itkimg.GetOrigin()
    itk_spacing = itkimg.GetSpacing()
    imgarr = itk.GetArrayFromImage(itkimg)
    sitkimg = sitk.GetImageFromArray(imgarr)
    direction = (
        itk_direction.GetVnlMatrix().get(0, 0),
        itk_direction.GetVnlMatrix().get(0, 1),
        itk_direction.GetVnlMatrix().get(0, 2),
        itk_direction.GetVnlMatrix().get(0, 3),
        itk_direction.GetVnlMatrix().get(0, 4),
        itk_direction.GetVnlMatrix().get(0, 5),
        0,
        0,
        1,
    )

    sitkimg.SetDirection(direction)

    sitkimg.SetSpacing((itk_spacing[0], itk_spacing[1], itk_spacing[2]))
    sitkimg.SetOrigin((itk_origin[0], itk_origin[1], itk_origin[2]))
    return sitkimg


def _parse_iteration_file(fname, metricname):
    with open(fname, "r") as f:
        lines = f.readlines()

    iterdict = {}
    headers = lines[0].strip("\n").split("\t")
    for item in headers:
        iterdict[item] = []

    for line in lines[1:]:
        values = [float(k) for k in line.strip("\n").split("\t")]

        for indx, item in enumerate(headers):
            iterdict[item].append(values[indx])

    # now to numpy
    npdict = {}
    for item in headers:
        npdict[item] = iterdict[item]
    npdict["MetricName"] = metricname
    return npdict


def _register(
    moving: itk.Image,
    fixed: itk.Image,
    parameter_map: itk.elxParameterObjectPython,
    fixedmask: itk.Image | None = None,
    movingmask: itk.Image | None = None,
    getresultimage: bool = True,
    iterationinfo: bool = False,
    initial_xfm_filename: str | None = None,
) -> dict:
    """
    For multiple parametermaps use VectorOfParameterMap
    :param moving:
    :param fixed:
    :param parameterMap:
    :param dbg_capture:
    :param fixedmask:
    :param movingmask:
    :return:
    turn:
    """
    # if isinstance(moving,str):

    # if isinstance(fixed,str):

    elastix_object = itk.ElastixRegistrationMethod.New(fixed, moving)
    elastix_object.SetParameterObject(parameter_map)

    if initial_xfm_filename is not None:
        elastix_object.SetInitialTransformParameterFileName(initial_xfm_filename)

    if fixedmask is not None:
        elastix_object.SetFixedMask(fixedmask)
    if movingmask is not None:
        elastix_object.SetMovingMask(movingmask)

    elastix_object.SetLogToConsole(True)

    td = tempfile.mkdtemp()
    elastix_object.SetOutputDirectory(td)
    t_start = time.time()
    elastix_object.UpdateLargestPossibleRegion()
    elapsed_reg = time.time() - t_start

    # if  isinstance(parameterMap,sitk.VectorOfParameterMap):
    #     for l in range(1,len(parameterMap)):
    #         # if dbg_capture is not None:

    #     if parameterMap.has_key("InitialTransformParametersFileName") and  parameterMap["InitialTransformParametersFileName"][0]:

    # if dbg_capture is not None or last30 is not None:

    # if fixedmask is not None:

    # if movingmask is not None:

    # TODO - run as subprocess as a workaround to logging: https://github.com/SuperElastix/SimpleElastix/issues/104

    # if iterationinfo:

    # if initial_xfm is not None:
    #     assert False, "TODO!"
    #     if not isinstance(initial_xfm,str):

    transform_parameter_map = elastix_object.GetTransformParameterObject()
    # in case of an initial transform, this contains just a reference to that file - which is temporary. We need to return both transform maps
    # if initial_xfm is not None:
    #     assert False, "TODO - this is not working, seems to be ignored"
    #     # remove the ref to the temp file

    if getresultimage:
        t = time.time()
        moving_l_fixed = elastix_object.GetOutput()
        elapsed_resample = time.time() - t
    else:
        moving_l_fixed = None
        elapsed_resample = None

    dbg_xfms = None
    if iterationinfo:  # TODO - itk placeholders here
        xfms = sorted(glob.glob(os.path.join(td, "TransformParameters.*.R*")))
        if initial_xfm_filename is not None:
            initial_xfm = itk.ReadParameterFile(initial_xfm_filename)
            dbg_xfms = {
                os.path.basename(k): [initial_xfm, itk.ReadParameterFile(k)]
                for k in xfms
            }
        else:
            dbg_xfms = {os.path.basename(k): [itk.ReadParameterFile(k)] for k in xfms}

    iterfiles = sorted(
        glob.glob(os.path.join(td, "IterationInfo*"))
    )  # this does not seem to include non linear part

    iterationinfo_records = []
    for _indx, iterfile in enumerate(iterfiles):
        metricname = parameter_map.GetParameter(0, "Metric")
        iterationinfo_records.append(
            _parse_iteration_file(iterfile, metricname)
        )  # TODO - must be capable of translation multi param regs

    shutil.rmtree(td)

    redict = {
        "parammap": parameter_map,
        "xfmmaps": transform_parameter_map,
        "moving_l_fixed": moving_l_fixed,
        "iterationinfo": iterationinfo_records,
        "elapsed_resample": elapsed_resample,
        "elapsed_reg": elapsed_reg,
        "dbg_xfms": dbg_xfms,
    }

    return redict


# def _DTIaffineA2B(
#     moving: sitk.Image,
#     fixed: sitk.Image,
#     transformparameters: dict | None = None,
#     custom_param_url: str | None = None,
#     fixedmask: sitk.Image | None = None,
#     iterationinfo: bool = False,
# ) -> dict:
#     """

#     :rtype: objectows import gaussian
#     from scipy.signal import convolve
#     import datetime
#     from matplotlib import cm
#     import numpy as np
#     import pydicom
#     from dicomutils.dicomutils import sitk2generic_ct
#     from dicomutils.ncctdcm2nii import tilt_correct, uniform_slice_thickness
#     from . import ncct_paths
#     """

#     if custom_param_url is not None:
#         parameterMap = sitk.ReadParameterFile(custom_param_url)
#     else:
#         parameterMap = sitk.ReadParameterFile(os.path.join(datafolder, "affineDTI.txt"))

#     if transformparameters is not None:
#         for key, val in transformparameters.items():
#             parameterMap[key] = val

#     retdict = _register(
#         moving, fixed, parameterMap, fixedmask=fixedmask, iterationinfo=iterationinfo
#     )
#     retdict["xfmmaps"]
#     retdict["moving_l_fixed"]
#     iterationinfo = retdict["iterationinfo"]
#     retdict["elapsed_reg"]
#     retdict["elapsed_resample"]
#     retdict["dbg_xfms"]

#     return retdict


def dti_affine(
    moving: itk.Image,
    fixed: itk.Image,
    transformparameters: dict | None = None,
    custom_param_url: str | None = None,
    fixedmask: itk.Image | None = None,
    iterationinfo: bool = False,
    initial_xfm_filename: str | None = None,
) -> dict:
    """

        :rtype: objectows import gaussian
    from scipy.signal import convolve
    import datetime
    from matplotlib import cm
    import numpy as np
    import pydicom
    from dicomutils.dicomutils import sitk2generic_ct
    from dicomutils.ncctdcm2nii import tilt_correct, uniform_slice_thickness
    from . import ncct_paths
    """
    parameter_object = itk.ParameterObject.New()
    if custom_param_url is not None:
        parameter_object.AddParameterFile(custom_param_url)
    else:
        parameter_object.AddParameterFile(os.path.join(datafolder, "affineDTI.txt"))

    # if transformparameters is not None:
    #     for key, val in transformparameters.items():
    if transformparameters is not None:
        for key, val in transformparameters.items():
            parameter_object.SetParameter(key, val)

    retdict = _register(
        moving,
        fixed,
        parameter_object,
        fixedmask=fixedmask,
        iterationinfo=iterationinfo,
        initial_xfm_filename=initial_xfm_filename,
    )

    return retdict


# def affineA2B(moving, fixed, transformparameters=None, dbg_capture=None):
#     """

#     :rtype: object
#     """
#     parameterMap = sitk.ParameterMap(
#         sitk.GetDefaultParameterMap("affine")
#     )  # sitk.ReadParameterFile(os.path.join(curpath,"affineDTI.txt"))
#     parameterMap["AutomaticTransformInitialization"] = ["true"]
#     parameterMap["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

#     if transformparameters:
#         for key, val in transformparameters.items():
#             parameterMap[key] = val

#     retdict = _register(moving, fixed, parameterMap, dbg_capture)
#     retdict["xfmmaps"]
#     retdict["moving_l_fixed"]
#     retdict["iterationinfo"]
#     retdict["elapsed_reg"]
#     retdict["elapsed_resample"]

#     return retdict


def imgnarray(fname):
    img = sitk.ReadImage(fname)

    imgarray = sitk.GetArrayFromImage(img)

    if (img.GetNumberOfComponentsPerPixel() > 0) and (
        len(img.GetSize()) == 2
    ):  # vector image loses singular first dim
        imgarray = np.expand_dims(imgarray, axis=0)
        img = sitk.GetImageFromArray(imgarray, isVector=True)

    return (img, imgarray)


def itk_resample(
    moving: itk.Image,
    xfm: itk.elxParameterObjectPython.elastixParameterObject,
    setlogtoconsole: bool = False,
    iporder_str: str | None = None,
    like: itk.Image | None = None,
) -> itk.Image:

    xfm_use = itk.ParameterObject.New()
    for i in range(xfm.GetNumberOfParameterMaps()):
        xfm_use.AddParameterMap(xfm.GetParameterMap(i))

    nmaps = xfm_use.GetNumberOfParameterMaps()
    if iporder_str is not None:
        xfm_use.GetParameterMap(nmaps - 1)["FinalBSplineInterpolationOrder"] = [
            iporder_str
        ]
    if like is not None:
        xfm_use.GetParameterMap(nmaps - 1)["Size"] = [str(s) for s in like.GetSize()]
        xfm_use.GetParameterMap(nmaps - 1)["Spacing"] = [
            str(s) for s in like.GetSpacing()
        ]
        xfm_use.GetParameterMap(nmaps - 1)["Origin"] = [
            str(s) for s in like.GetOrigin()
        ]

        # direction needs to be column majpr in sitk so
        tp = np.reshape(np.array(like.GetDirection()), [3, 3], order="F")
        xfm_use.GetParameterMap(nmaps - 1)["Direction"] = [str(s) for s in tp.flatten()]

    tformix = itk.TransformixFilter.New(moving)
    tformix.SetLogToConsole(setlogtoconsole)
    tformix.SetTransformParameterObject(xfm)

    tformix.UpdateLargestPossibleRegion()

    resampled = tformix.GetOutput()
    return resampled


# def resample(
#     moving: sitk.Image,
#     xfm: list[sitk.ParameterMap],
#     invert: bool = False,
#     SetLogToConsole: bool = False,
#     iporder_str: str | None = None,
#     like: sitk.Image | None = None,
# ):
#     tformix = sitk.TransformixImageFilter()
#     if isinstance(moving, str):
#         moving = sitk.ReadImage(moving)

#     tformix.SetMovingImage(moving)

#     tformix.SetTransformParameterMap(xfm[0])
#     tformix.SetLogToConsole(SetLogToConsole)

#     for x in xfm[1:]:
#         tformix.AddTransformParameterMap(x)
#     # for cxfm in xfm:

#     if invert:  # reg brain to itself using this as initial and
#         sitk.ElastixImageFilter()

#     if iporder_str is not None:
#         tformix.SetTransformParameter("FinalBSplineInterpolationOrder", [iporder_str])
#     if like is not None:
#         tformix.SetTransformParameter("Size", [str(s) for s in like.GetSize()])
#         tformix.SetTransformParameter("Spacing", [str(s) for s in like.GetSpacing()])
#         tformix.SetTransformParameter("Origin", [str(s) for s in like.GetOrigin()])

#         # direction needs to be column majpr in sitk so
#         tp = np.reshape(np.array(like.GetDirection()), [3, 3], order="F")
#         tformix.SetTransformParameter("Direction", [str(s) for s in tp.flatten()])

#     tformix.LogToConsoleOn()
#     tformix.Execute()
#     resampled = tformix.GetResultImage()
#     return resampled


def set_resampling_grid_on_transform(
    xfm_in: itk.elxParameterObjectPython.mapstringvectorstring, like: sitk.Image
) -> itk.elxParameterObjectPython.mapstringvectorstring:

    xfm_use = itk.elxParameterObjectPython.mapstringvectorstring(xfm_in)
    xfm_use["Size"] = [str(s) for s in like.GetSize()]
    xfm_use["Spacing"] = [str(s) for s in like.GetSpacing()]
    xfm_use["Origin"] = [str(s) for s in like.GetOrigin()]

    # direction needs to be column majpr in sitk so
    tp = np.reshape(np.array(like.GetDirection()), [3, 3], order="F")
    xfm_use["Direction"] = [str(s) for s in tp.flatten()]

    return xfm_use


# def invert_euler_transform(transform: sitk.ParameterMap) -> sitk.ParameterMap:

#     center_of_rotation = [float(v) for v in transform["CenterOfRotationPoint"]]
#     pertubation_radians = [float(v) for v in transform["TransformParameters"]]
#     e3d = sitk.Euler3DTransform(
#         center_of_rotation, *pertubation_radians[0:3], pertubation_radians[3:6]
#     )

#     e3d_inverse = e3d.GetInverse()
#     rads = (e3d_inverse.GetAngleX(), e3d_inverse.GetAngleY(), e3d_inverse.GetAngleZ())
#     trans_list = e3d_inverse.GetTranslation()

#     transform["TransformParameters"] = [str(rad_angle) for rad_angle in rads] + [
#         str(trans) for trans in trans_list
#     ]

#     return transform


# def template_reg(template_l, origncct, outputfolder, cachemode=False):

#     N2Tloc = outputfolder + "/N2T.txt"
#     T2Nloc = outputfolder + "/T2N.txt"

#     if cachemode and os.path.exists(N2Tloc) and os.path.exists(T2Nloc):
#         N2T = [sitk.ReadParameterFile(outputfolder + "/N2T.txt")]
#         T2N = [sitk.ReadParameterFile(outputfolder + "/T2N.txt")]
#         return T2N, N2T

#     template, template_arr = imgnarray(template_l)

#     # clamp ncct
#     native_arr = sitk.GetArrayFromImage(origncct)
#     tmparray = native_arr.copy()
#     tmparray[tmparray < 0] = 0
#     tmparray[tmparray > 100] = 100
#     ncct = arr2img(tmparray, origncct)

#     {
#         "filename": outputfolder + "/ncct2template_affine.avi",
#         "every_n_xfm": 30,
#         "everyNslice": 2,
#         "DS": 2,
#     }

#     # NCCT TO TEMPLATE - USED FOR GUIDING MIRROR XFM
#     retinfo = DTIaffineA2B(ncct, template, saveas="myaffine.txt")

#     N2T = retinfo["xfmmaps"]
#     ncct_affineL = retinfo["moving_l_fixed"]
#     retinfo["iterationinfo"]
#     sitk.WriteParameterFile(N2T[0], N2Tloc)
#     sitk.WriteImage(ncct_affineL, outputfolder + "/ncct_affineL.nii")

#     # INVERT
#     T2N = invert_DTI_xfm(ncct, N2T[0])
#     sitk.WriteParameterFile(T2N[0], T2Nloc)

#     T_L_native = resample(template, T2N)

#     # save xfms

#     sitk2montage(
#         template_l,
#         outputfolder + "/template_reg_A_template.png",
#         range=[0, 60],
#         everyNthslice=5,
#     )
#     sitk2montage(ncct, outputfolder + "/template_reg_B_native.png", range=[0, 60])
#     sitk2montage(
#         T_L_native, outputfolder + "/template_reg_C_Tlnative.png", range=[0, 60]
#     )

#     return T2N, N2T


# def template_reg_nlin(template_l, origncct, outputfolder, cachemode=False):
#     # False   #not implemented

#     N2Tloc = outputfolder + "/N2Tnlin[X].txt"
#     T2Nloc = outputfolder + "/T2Nnlin[X].txt"

#     if (
#         cachemode
#         and os.path.exists(N2Tloc.replace("[X]", "1"))
#         and os.path.exists(T2Nloc.replace("[X]", "1"))
#     ):
#         N2T = []
#         N2T.append(sitk.ReadParameterFile(outputfolder + "/N2Tnlin0.txt"))
#         N2T.append(sitk.ReadParameterFile(outputfolder + "/N2Tnlin1.txt"))
#         T2N = []
#         T2N.append(sitk.ReadParameterFile(outputfolder + "/T2Nnlin0.txt"))
#         T2N.append(sitk.ReadParameterFile(outputfolder + "/T2Nnlin1.txt"))
#         return T2N, N2T

#     template, _ = imgnarray(template_l)

#     # clamp ncct
#     native_arr = sitk.GetArrayFromImage(origncct)
#     tmparray = native_arr.copy()
#     tmparray[tmparray < 0] = 0
#     tmparray[tmparray > 100] = 100
#     ncct = arr2img(tmparray, origncct)


#     retinfo = affine_nonlinA2B(ncct, template)

#     N2T = retinfo["xfmmaps"]
#     N2T[0]["InitialTransformParametersFileName"] = ["NoInitialTransform"]
#     N2T[1]["InitialTransformParametersFileName"] = ["NoInitialTransform"]

#     ncct_affineNL = retinfo["moving_l_fixed"]
#     retinfo["iterationinfo"]
#     sitk.WriteParameterFile(N2T[0], N2Tloc.replace("[X]", "0"))
#     sitk.WriteParameterFile(N2T[1], N2Tloc.replace("[X]", "1"))
#     sitk.WriteImage(ncct_affineNL, outputfolder + "/ncct_nlinL.nii")

#     # INVERT
#     T2N = invert_DTI_nlin_xfm(template, N2T, retinfo["parammap"])

#     # T2N now resamples in T space, set to N space
#     T2N[1]["Spacing"] = [str(k) for k in origncct.GetSpacing()]
#     T2N[1]["Origin"] = [str(k) for k in origncct.GetOrigin()]
#     T2N[1]["Direction"] = [str(k) for k in origncct.GetDirection()]
#     T2N[1]["Size"] = [str(k) for k in origncct.GetSize()]
#     sitk.WriteParameterFile(T2N[0], T2Nloc.replace("[X]", "0"))
#     sitk.WriteParameterFile(T2N[1], T2Nloc.replace("[X]", "1"))

#     T_L_native = resample(template, T2N)

#     # save xfms

#     sitk2montage(
#         template_l,
#         outputfolder + "/template_nlinreg_A_template.png",
#         range=[0, 60],
#         everyNthslice=5,
#     )
#     sitk2montage(ncct, outputfolder + "/template_nlinreg_B_native.png", range=[0, 60])
#     sitk2montage(
#         T_L_native, outputfolder + "/template_nlinreg_C_Tlnative.png", range=[0, 60]
#     )

#     return T2N, N2T


# def reg_mr2ncct(
#     mrloc,
#     ncctloc,
#     outputfolder,
#     png_txt=None,
#     caching=False,
#     roiloc=None,
#     clampmr=False,
#     mask_ncct=False,
# ):
#     """
#     Intended for MR images to NCCT.

#     :return:
#     """
#     xfmfile = os.path.join(outputfolder, "mr2ncct.txt")

#     if os.path.exists(xfmfile) and caching:
#         mrimg = ReadImage(mrloc)
#         ncctimg = ReadImage(ncctloc)
#         mr2ncctxfm = [sitk.ReadParameterFile(xfmfile)]
#         mr_l_ncct = resample(mrimg.img, mr2ncctxfm)
#     else:
#         if not os.path.exists(outputfolder):
#             os.makedirs(outputfolder)
#         mrimg = ReadImage(mrloc)
#         if mrimg.img.GetPixelIDTypeAsString() == "vector of 16-bit unsigned integer":
#             mrarr = mrimg.arr.astype(np.float32)
#             mrimg.setarr(mrarr)
#             mrimg.update_arr()

#         if clampmr:
#             mrarr = mrimg.arr
#             qtiles = np.quantile(mrarr[mrarr > 0], [0.50, 0.99])
#             mrarr[mrarr < 30] = 0
#             mrarr[mrarr > qtiles[1]] = qtiles[1]
#             mrimg.update_arr()

#         ncctimg = ReadImage(ncctloc)
#         ncctimg.arr[ncctimg.arr < 0] = 0
#         ncctimg.arr[ncctimg.arr > 100] = 100
#         ncctimg.update_arr()
#         fixedmask = None
#         if (
#             mask_ncct
#         ):  # get a brain mask by reg to standard space. Use existing reg if found

#             template_l = os.path.join(datafolder, "scct_unsmooth_RAI_clamped.nii")
#             T2N, N2T = template_reg(
#                 template_l, ncctimg.img, outputfolder, cachemode=caching
#             )
#             mask_standard_space = ReadImage(
#                 os.path.join(datafolder, "template_mask.nii")
#             )
#             fixedmaskF = resample(mask_standard_space.img, T2N, iporder_str="0")

#             fixedmask = sitk.Cast(fixedmaskF, sitk.sitkUInt8)
#             sitk.WriteImage(fixedmask, join(outputfolder, "fixed_mask.nii"))

#         T2_arr_rescaled = mrimg.arr.copy()
#         img_use = arr2img(T2_arr_rescaled, mrimg.img)

#         regop = rigidA2B(
#             img_use,
#             ncctimg.img,
#             transformparameters=None,
#             customxfm=os.path.join(curpath, "MUTUAL.txt"),
#             logtoconsole=False,
#             fixedmask=fixedmask,
#         )

#         mr2ncctxfm = regop["xfmmaps"]

#         mr_l_ncct = resample(mrimg.img, mr2ncctxfm)
#         regop["iterationinfo"]
#         sitk.WriteParameterFile(mr2ncctxfm[0], xfmfile)

#     mr_l_ncct_img = ReadImage(mr_l_ncct)
#     sitk.WriteImage(mr_l_ncct, join(outputfolder, "mr_l_ncct.nii"))

#     # we also need a ones image for mr - even is this is usually redundant
#     mr2ncctxfm[0]["FinalBSplineInterpolationOrder"] = ["1"]  # stay float
#     ones = np.ones(mrimg.arr.shape)
#     onesimg = arr2img(ones, mrimg.img)
#     ones_res = resample(onesimg, mr2ncctxfm)
#     sitk.WriteImage(ones_res, join(outputfolder, "ones_mr_l_ncct.nii"))
#     sitk2montage(ncctimg.img, join(outputfolder, "A_ncct.png"), range=[0, 60], DS=2)
#     if png_txt is not None:
#         tmprgb = sitk2montage(
#             mr_l_ncct_img.img,
#             "",
#             range=[0, np.percentile(mr_l_ncct_img.arr[mr_l_ncct_img.arr > 0], 99)],
#         )
#         tmprgb = imageutils.textonrgb(
#             tmprgb, png_txt, 1500, 90, fs=80, pastepos=[tmprgb.shape[0] - 90, 0]
#         )
#         imageio.imwrite(outputfolder + "B_mr_l_ncct.png", tmprgb)

#     else:
#         sitk2montage(
#             mr_l_ncct_img.img,
#             os.path.join(outputfolder, "B_mr_l_ncct.png"),
#             range=[0, mr_l_ncct_img.arr.max() * 0.97],
#             DS=2,
#         )

#     if roiloc is not None:
#         mr2ncctxfm[0]["FinalBSplineInterpolationOrder"] = ["0"]
#         mrimg = sitk.ReadImage(mrloc)
#         roiimg = sitk.ReadImage(roiloc)
#         roiimg.CopyInformation(mrimg)
#         roi_res = resample(roiimg, mr2ncctxfm)
#         sitk.WriteImage(roi_res, join(outputfolder, "mrroi_l_ncct.nii"))
#         sitk2montage(
#             mr_l_ncct_img.img,
#             os.path.join(outputfolder, "C_mrroi_l_ncct.png"),
#             range=[0, mr_l_ncct_img.arr.max() * 0.97],
#             DS=2,
#             maskovl=roi_res,
#         )

#     return mr_l_ncct_img.img


# def rigidA2B(
#     moving,
#     fixed,
#     transformparameters=None,
#     customxfm=None,
#     fixedmask=None,
#     movingmask=None,
#     getresultimage=True,
#     iterationinfo=False,
#     logtoconsole=True,
#     initial_xfm: sitk.ParameterMap | None = None,
#     make_movie: str | None = None,
# ):
#     if make_movie:
#         iterationinfo = True

#     if type(customxfm) is str:
#         parameterMap = sitk.ReadParameterFile(customxfm)
#     elif type(customxfm) is sitk.SimpleITK.ParameterMap:
#         parameterMap = customxfm
#     else:
#         parameterMap = sitk.ReadParameterFile(
#             os.path.join(datafolder, "crosscorr_D3.txt")
#         )

#     if transformparameters is not None:
#         for key, val in transformparameters.items():
#             parameterMap[key] = val

#     initial_xfm_filename = None
#     if initial_xfm is not None:
#         initial_xfm_filename = tempfile.mktemp("_xfm.txt")
#         sitk.WriteParameterFile(initial_xfm, initial_xfm_filename)

#     retdict = _register(
#         moving,
#         fixed,
#         parameterMap,
#         fixedmask=fixedmask,
#         movingmask=movingmask,
#         getresultimage=getresultimage,
#         iterationinfo=iterationinfo,
#         logtoconsole=logtoconsole,
#         initial_xfm_filename=initial_xfm_filename,
#     )

#     dbgxfms = retdict["dbg_xfms"]
#     if make_movie is not None:
#         xfms2movie(
#             dbgxfms,
#             moving,
#             fixed,
#             make_movie,
#             everyNslice=1,
#             every_n_xfm=20,
#             mmrange=None,
#         )

#     # if initial_xfm is not None:
#     # # add the translation directly to the final transform. Assume initial is translation only
#     #
#     # for k in range(3):
#     #

#     return retdict


def affine_nonlin(
    moving: sitk.Image,
    fixed: sitk.Image,
    transformparameters_affine: dict | None = None,
    transformparameters_bspline=None,
    initial_xfm: None | itk.ParameterObject = None,
    fixedmask: sitk.Image | None = None,
    bsplinelevels: int = 3,
):

    parameter_map_vector = itk.ParameterObject.New()
    parameter_map_vector.AddParameterFile(os.path.join(datafolder, "affineDTI.txt"))
    parameter_map_vector.AddParameterMap(
        itk.ParameterObject.GetDefaultParameterMap("bspline", bsplinelevels)
    )

    affine_map = parameter_map_vector.GetParameterMap(0)
    bspline_map = parameter_map_vector.GetParameterMap(1)

    bspline_map["GridSpacingSchedule"] = ["4", "2", "1"]
    bspline_map["FinalGridSpacingInPhysicalUnits"] = ["4", "4", "4"]
    bspline_map["MaximumNumberOfIterations"] = ["1024"]
    bspline_map["MaximumNumberOfSamplingAttempts"] = ["2"]

    if transformparameters_affine:
        for key, val in transformparameters_affine.items():
            affine_map[key] = val

    if transformparameters_bspline:
        for key, val in transformparameters_bspline.items():
            bspline_map[key] = val

    parameter_map_vector.SetParameterMap(0, affine_map)
    parameter_map_vector.SetParameterMap(1, bspline_map)

    # create file for initial xfm if provided
    if initial_xfm is not None:
        initial_xfm_filename = os.path.join(tempfile.gettempdir(), "initial_xfm.txt")
        itk.ParameterObject.WriteParameterFile(initial_xfm, initial_xfm_filename)
    else:
        initial_xfm_filename = None

    retdict = _register(
        sitk2itk(moving),
        sitk2itk(fixed),
        parameter_map_vector,
        fixedmask=fixedmask,
        initial_xfm_filename=initial_xfm_filename,
    )

    if initial_xfm is not None:
        xfmmaps = retdict["xfmmaps"]
        xfmmaps_with_initial = itk.ParameterObject.New()
        # let's prepend the initial xfm to the list of xfms in retdict
        xfmmaps_with_initial.AddParameterMap(initial_xfm)
        for i in range(xfmmaps.GetNumberOfParameterMaps()):
            xfmmaps_with_initial.AddParameterMap(xfmmaps.GetParameterMap(i))
        retdict["xfmmaps"] = xfmmaps_with_initial

    return retdict


# def nonlinA2B(
#     moving, fixed, transformparameters=None, parameterMap=None, dbg_capture=""
# ):
#     if parameterMap is not None:  # default
#         parameterMap = sitk.GetDefaultParameterMap("bspline", 3)
#         parameterMap["GridSpacingSchedule"] = ["4", "2", "1"]
#         parameterMap["FinalGridSpacingInPhysicalUnits"] = ["4", "4", "4"]
#         parameterMap["MaximumNumberOfIterations"] = ["1024"]
#         parameterMap["MaximumNumberOfSamplingAttempts"] = ["2"]

#     if transformparameters is not None:
#         for key, val in transformparameters.items():
#             parameterMap[key] = val

#     retdict = _register(moving, fixed, parameterMap)
#     transformParameterMaps = retdict["xfmmaps"]
#     resampled_img = retdict["moving_l_fixed"]

#     return transformParameterMaps, resampled_img


def arr2img(nparr, imgtemplate, is_vector=False):
    if nparr.dtype == bool:
        sitkimg = sitk.GetImageFromArray(nparr.astype(np.uint8), isVector=is_vector)
    else:
        sitkimg = sitk.GetImageFromArray(nparr, isVector=is_vector)

    sitkimg.SetOrigin(imgtemplate.GetOrigin())
    sitkimg.SetSpacing(imgtemplate.GetSpacing())
    sitkimg.SetDirection(imgtemplate.GetDirection())
    return sitkimg


# def xfms2movie_single_slice(
#     xfmfiles, moving, fixed, opname, every_n_xfm=10, DS=1, dims_viz=None
# ):
#     if dims_viz is None:
#         dims_viz = [1, 2]

#     all_dims = set([0, 1, 2])
#     slice_along_dim = list(all_dims - set(dims_viz))[0]
#     origo = np.array([0, 0, 0])
#     origo[slice_along_dim] = int(np.round((fixed.GetSize()[slice_along_dim] - 1) / 2))
#     new_origo = fixed.TransformIndexToPhysicalPoint(origo.tolist())
#     total_FOV = np.array(fixed.GetSpacing()) * np.array(fixed.GetSize())

#     new_spacing = np.array([1, 1, 1])
#     new_size = (total_FOV / new_spacing).astype(np.uint32)
#     new_size[slice_along_dim] = 1
#     new_spacing[slice_along_dim] = 1

#     xfmfilenames = list(xfmfiles.keys())
#     ident_xfm = sitk.SimpleITK.ParameterMap(xfmfiles[xfmfilenames[0]])
#     ident_xfm["TransformParameters"] = [str(k) for k in [0, 0, 0, 0, 0, 0]]
#     del ident_xfm["InitialTransformParametersFileName"]
#     ident_xfm["Size"] = [str(s) for s in new_size]
#     ident_xfm["Spacing"] = [str(s) for s in new_spacing]
#     ident_xfm["Origin"] = [str(s) for s in new_origo]
#     # lets extract a center slice (Coronal,Sagital or Axial in fixed space) for fixed (x1) and moving (x number of iterations/everyN)
#     # So we need an origo and direction

#     fixed_reference = resample(fixed, [ident_xfm], iporder_str="1")
#     fixed_array_col_major = np.transpose(
#         sitk.GetArrayFromImage(fixed_reference), [2, 1, 0]
#     )

#     # the specified dims are relative to col major conventions
#     fixed_array_img = np.squeeze(
#         np.transpose(fixed_array_col_major, dims_viz + [slice_along_dim])
#     )
#     fixed_array_img = fixed_array_img / fixed_array_img.max()

#     cxfm = xfmfiles[xfmfilenames[0]]
#     cxfm["Size"] = [str(s) for s in new_size]
#     cxfm["Spacing"] = [str(s) for s in new_spacing]
#     cxfm["Origin"] = [str(s) for s in new_origo]

#     writer = imageio.get_writer(opname, fps=20)

#     for k in tqdm(range(0, len(xfmfilenames), every_n_xfm)):
#         cxfm = xfmfiles[xfmfilenames[k]]
#         cxfm["Size"] = [str(s) for s in new_size]
#         cxfm["Spacing"] = [str(s) for s in new_spacing]
#         cxfm["Origin"] = [str(s) for s in new_origo]
#         cresampled = resample(moving, [cxfm], iporder_str="1")
#         resarray_col_major = np.transpose(sitk.GetArrayFromImage(cresampled), [2, 1, 0])
#         resarray_img = np.squeeze(
#             np.transpose(resarray_col_major, dims_viz + [slice_along_dim])
#         )
#         resarray_img = resarray_img / resarray_img.max()
#         resarray_img[:, 0 : (int(resarray_img.shape[1] / 2))] = fixed_array_img[
#             :, 0 : (int(resarray_img.shape[1] / 2))
#         ]
#         writer.append_data((255 * resarray_img).astype(np.uint8))

#     writer.close()


# def xfms2movie(
#     xfmfiles,
#     moving,
#     fixed,
#     opname: str,
#     every_n_xfm=10,
#     every_n_slice=10,
#     DS=1,
#     mmrange=None,
# ) -> None:
#     tmp = sitk.GetArrayFromImage(fixed)
#     fixed_arr = reorder_yxz(tmp)
#     if mmrange is None:
#         mmrange_use = (fixed_arr.min(), fixed_arr.max())
#     else:
#         mmrange_use = mmrange
#     fixed_arr = imageutils.imrescale(
#         fixed_arr[::DS, ::DS, ::every_n_slice], mmrange_use, (0, 255)
#     )

#     fixed_mont = imageutils.stack2rgbmont(fixed_arr, (1, 1, 1)).astype(np.uint8)
#     writer = imageio.v2.get_writer(opname, mode="I", fps=20)

#     paste_mask = fixed_arr.astype(bool)
#     paste_mask[:] = True
#     paste_mask[:, 0 : int(paste_mask.shape[1] / 2), :] = False
#     paste_mask_mont = imageutils.stack2rgbmont(paste_mask, (1, 1, 1)).astype(np.uint8)
#     xfmfilenames = list(xfmfiles.keys())
#     for k in tqdm(range(0, len(xfmfilenames), every_n_xfm)):
#         cxfm = xfmfiles[xfmfilenames[k]]

#         cresampled = resample(moving, cxfm)
#         resarray = sitk.GetArrayFromImage(cresampled)
#         resarray = reorder_yxz(resarray)

#         if not mmrange:
#             mmrange_use = (resarray.min(), resarray.max())
#         else:
#             mmrange_use = mmrange
#         resarray = imageutils.imrescale(
#             resarray[::DS, ::DS, ::every_n_slice], mmrange_use, (0, 255)
#         )
#         immont = imageutils.stack2rgbmont(resarray, (1, 1, 1))
#         fixed_montcp = fixed_mont.copy()
#         fixed_montcp[paste_mask_mont > 0] = immont[paste_mask_mont > 0]
#         writer.append_data(fixed_montcp)

#     writer.close()


def image_center(ncct):
    vox_center = np.array(ncct.GetSize()) / 2

    return ncct.TransformContinuousIndexToPhysicalPoint(vox_center)
