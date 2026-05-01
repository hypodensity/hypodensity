from __future__ import annotations

import imageio
import numpy as np
import SimpleITK as sitk


# pylint: disable=too-many-arguments
def sitk2montage(
    *,
    sitkimg: sitk.Image | None,
    opname: str | None = None,
    value_range: tuple[float, float] | None = None,
    maskovl: sitk.Image | np.ndarray | None = None,
    mask_mix: tuple[int, int, int] = (0, 255, 0),
    every_nth_slice: int = 1,
    ds: int | None = None,
    cropmask: np.ndarray | None = None,
    rc: tuple[int, int] | None = None,
) -> np.ndarray:
    """Wraps np2montage - see description here
    Args:
        sitkimg (sitk.Image): SimpleITK image, assumed LPS
        opname (str | None, optional): Output url or none, imageio supported formats only. Defaults to None.
        value_range (tuple[float,float] | None, optional): min and max for gray level mapping. Defaults to None.
        maskovl (np.ndarray | sitk.Image| None, optional): mask for overlay. Defaults to None.
        mask_mix (tuple[float,float,float], optional): mask color. Defaults to [0, 255, 0].
        every_nth_slice (int, optional): decimate along slice dimention. Defaults to 1.
        ds (int | None, optional): decimate in plane resolution. Defaults to None.
        cropmask (np.ndarray | None, optional): mask to crop the output. Defaults to None.
        rc (tuple[int,int] | None, optional): specify rows and column of montage, must multiply to number of slices. Defaults to None.

    Returns:
        np.ndarray: The resultim montage as uint8
    """

    if isinstance(sitkimg, str):
        sitkimg = sitk.ReadImage(sitkimg)
    if sitkimg is None and maskovl is None:
        raise ValueError("sitkimg or maskovl must be provided!")

    if maskovl is not None:
        if isinstance(maskovl, sitk.Image):
            mask_arr = sitk.GetArrayFromImage(maskovl)
        else:
            mask_arr = maskovl
    else:
        mask_arr = None

    if sitkimg is not None:
        mat = sitk.GetArrayFromImage(sitkimg)
    else:  # empty image defaults to zero array of mask shape
        assert mask_arr is not None
        mat = np.zeros(mask_arr.shape, dtype=np.float32)

    rgb = np2montage(
        matin=mat,
        opname=opname,
        value_range=value_range,
        maskovl=mask_arr,
        mask_mix=mask_mix,
        every_nth_slice=every_nth_slice,
        ds=ds,
        cropmask=cropmask,
        rc=rc,
    )
    return rgb


def reorder_yxz(matin: np.ndarray) -> np.ndarray:
    if matin.ndim == 3:
        return np.transpose(matin, axes=(1, 2, 0))
    if matin.ndim == 4:
        return np.transpose(matin, axes=(1, 2, 0, 3))
    raise AssertionError("Not supported")


def reorder_zxy(matin: np.ndarray) -> np.ndarray:
    return np.transpose(matin, axes=(2, 0, 1))


# pylint: disable=too-many-locals
def np2montage(  # NOSONAR
    *,
    matin: np.ndarray,
    opname: str | None,
    value_range: tuple[float, float] | None = None,
    maskovl: np.ndarray | None = None,
    mask_mix: tuple[int, int, int] = (0, 255, 0),
    every_nth_slice: int = 1,
    ds: int | None = None,
    rc: tuple[int, int] | None = None,
    cropmask: np.ndarray | None = None,
    zyx: bool = True,
) -> np.ndarray:
    """Create montage from 3D or 4D (RGB) numpy array
       If the array is of 4d( RGB) type, then no scalar to rgb conversion is done

    Args:
        matin (np.ndarray): input 3D image to convert into a 2D montage. Can be RGB in which case data type is assumed to be uint78
        opname (str | None, optional): Output url or none, imageio supported formats only. Defaults to None.
        value_range (tuple[float,float] | None, optional): min and max for gray level mapping. Defaults to None.
        maskovl (np.ndarray | None, optional): mask for overlay. Defaults to None.
        mask_mix (tuple[float,float,float], optional): mask color. Defaults to [0, 255, 0].
        every_nth_slice (int, optional): decimate along slice dimention. Defaults to 1.
        ds (int | None, optional): decimate in plane resolution. Defaults to None.
        cropmask (np.ndarray | None, optional): mask to crop the output. Defaults to None.
        rc (tuple[int,int] | None, optional): specify rows and column of montage, must multiply to number of slices. Defaults to None.
        zyx (bool, optional): is data zyx (as the array in SimpleITK LPS images) or xyz. Defaults to True.

    Returns:
        np.ndarray: resultimg montage as uint8
    """
    is_rgb = matin.ndim == 4

    if maskovl is not None:
        assert is_rgb is False, "RGB not supported with mask overlays yet"

    if not zyx:
        assert is_rgb is False, "RGB not supported for xyz ordering yet"
        matin = reorder_zxy(matin)

        if maskovl is not None:
            assert is_rgb is False, "RGB not supported with mask overlays yet"
            maskovl = reorder_zxy(maskovl)

    if cropmask is not None:
        rowmin, rowmax, colmin, colmax, slicemin, slicemax = _get_mask_bounds_zyx(
            cropmask
        )
        if is_rgb:
            matin = matin[
                slicemin : slicemax + 1, colmin : colmax + 1, rowmin : rowmax + 1, :
            ]
        else:
            matin = matin[
                slicemin : slicemax + 1, colmin : colmax + 1, rowmin : rowmax + 1
            ]

        if maskovl is not None:
            maskovl = maskovl[
                slicemin : slicemax + 1, colmin : colmax + 1, rowmin : rowmax + 1
            ]

    mat_xyz = reorder_yxz(matin).copy()

    if not is_rgb:
        if value_range is not None:
            mat_xyz = imrescale(mat_xyz, from_range=value_range, to_range=(0, 1))
        else:
            mat_xyz = imrescale(mat_xyz, from_range=None, to_range=(0, 1))
        rgbmont = stack2rgbmont(
            mat_xyz, (255, 255, 255), every_nth_slice=every_nth_slice, rc=rc
        ).astype(np.uint8)
    else:
        rgbmont = stack2rgbmont(mat_xyz, every_nth_slice=every_nth_slice, rc=rc)

    if ds:
        rgbmont = rgbmont[::ds, ::ds, :]

    if maskovl is not None:
        maskovl = reorder_yxz(maskovl)
        maskovl = stack2rgbmont(
            maskovl.astype(np.float32), mask_mix, every_nth_slice=every_nth_slice, rc=rc
        ).astype(np.uint8)
        if ds:
            assert isinstance(maskovl, np.ndarray)
            maskovl = maskovl[::ds, ::ds, :]

        rgbmont = rgbmaskonrgb(rgbmont, maskovl)

    if opname is not None:
        imageio.imwrite(opname, rgbmont)
    return rgbmont


def montage(
    stackin: np.ndarray,
    every_nth_slice: int = 1,
    rc: tuple[int, int] | None = None,
    zyx: bool = False,
) -> np.ndarray:
    """_summary_

    :param stackin: image to convert to 2D montage. Either 3D or 4D (RGB) with last dimention color channels
    :param every_nth_slice: slice decimation, defaults to 1
    :param rc: row and column count of montage, defaults to None for auto layout
    :param zyx: is input zyx?, defaults to False
    :return: montage as spcified
    """

    is_rgb = stackin.ndim == 4
    if zyx:
        if is_rgb:
            stackin = np.transpose(stackin, [1, 2, 0, 3])
        else:
            stackin = np.transpose(stackin, [1, 2, 0])

    stackin = stackin[:, :, ::every_nth_slice]  # stack must be 3D
    indims = stackin.shape

    ylen = indims[0]
    xlen = indims[1]
    numims = indims[2]

    if rc is None:
        sqval = np.sqrt(numims)
        sqval = int(np.ceil(sqval))
        rownum = sqval
        if (sqval - 1) * sqval >= numims:
            rownum = sqval - 1
        rownum = int(rownum)
    else:
        sqval = rc[1]
        rownum = rc[0]

    if len(indims) > 3:
        numchannels = indims[3]
        mont = np.zeros((ylen * rownum, xlen * sqval, numchannels), dtype=stackin.dtype)
    else:
        mont = np.zeros((ylen * rownum, xlen * sqval), dtype=stackin.dtype)
        numchannels = 1

    for nim in range(numims):
        colindx = nim % sqval
        rowindx = int(np.floor(float(nim) / float(sqval)))

        colstart = colindx * xlen
        rowstart = rowindx * ylen

        rowend = rowstart + ylen
        colend = colstart + xlen

        if numchannels > 1:
            mont[rowstart:rowend, colstart:colend, :] = stackin[:, :, nim, :]
        else:
            mont[rowstart:rowend, colstart:colend] = stackin[:, :, nim]

    return mont


def imrescale(
    imin: np.ndarray,
    from_range: tuple[float, float] | None = None,
    to_range: tuple[float, float] = (0, 1),
) -> np.ndarray:
    """Rescale image from one range to another. Clamps to values output range values

    :param imin: image to rescale
    :param from_range: range to rescale from, eg 0,60 or None for min-max, defaults to None
    :param to_range: range to rescale to, defaults to (0, 1)
    :return: rescaled image
    """
    if from_range is None:
        from_range = (np.min(imin), np.max(imin))

    imout = (imin - from_range[0]) / (from_range[1] - from_range[0])

    imout = imout * (to_range[1] - to_range[0]) + to_range[0]

    # clamp
    imout[imout > to_range[1]] = to_range[1]
    imout[imout < to_range[0]] = to_range[0]
    return imout


def _get_mask_bounds_zyx(cropmask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    row_w_content = np.argwhere(np.sum(cropmask, axis=(0, 1)))[[0, -1]]
    col_w_content = np.argwhere(np.sum(cropmask, axis=(0, 2)))[[0, -1]]
    slice_w_content = np.argwhere(np.sum(cropmask, axis=(1, 2)))[[0, -1]]

    margin_x = 2
    margin_y = 2
    margin_z = 1
    rowmin = np.maximum(row_w_content[0][0] - margin_y, 0)
    rowmax = np.minimum(row_w_content[1][0] + margin_y, cropmask.shape[2] - 1)
    colmin = np.maximum(col_w_content[0][0] - margin_x, 0)
    colmax = np.minimum(col_w_content[1][0] + margin_x, cropmask.shape[1] - 1)
    slicemin = np.maximum(slice_w_content[0][0] - margin_z, 0)
    slicemax = np.minimum(slice_w_content[1][0] + margin_z, cropmask.shape[0] - 1)

    return (rowmin, rowmax, colmin, colmax, slicemin, slicemax)


def rgbmaskonrgb(
    rgbimg: np.ndarray, maskimg: np.ndarray, alpha: float | None = None
) -> np.ndarray:
    rgbimgo = rgbimg.copy()
    logic = np.sum(maskimg, 2) > 0  # assumes total black [0,0,0] is transparent

    logic_3d = np.dstack((logic, logic, logic))
    if alpha is None:
        rgbimgo[logic_3d] = maskimg[logic_3d]
    else:
        rgbimgo[logic_3d] = (
            rgbimgo[logic_3d].astype(np.float32) * (1 - alpha)
            + maskimg[logic_3d].astype(np.float32) * alpha
        )

    return rgbimgo


def stack2rgbmont(
    stack: np.ndarray,
    rgb: tuple[float, float, float] = (255, 255, 255),
    every_nth_slice: int = 1,
    rc: tuple[int, int] | None = None,
) -> np.ndarray:
    if stack.ndim == 3:
        mont = montage(stack.astype(np.float32), every_nth_slice=every_nth_slice, rc=rc)
        t1 = mont * np.float32(rgb[0])
        t2 = mont * np.float32(rgb[1])
        t3 = mont * np.float32(rgb[2])

    elif stack.ndim == 4 and stack.shape[3] == 3:
        t1 = montage(stack[:, :, :, 0], every_nth_slice=every_nth_slice, rc=rc)
        t2 = montage(stack[:, :, :, 1], every_nth_slice=every_nth_slice, rc=rc)
        t3 = montage(stack[:, :, :, 2], every_nth_slice=every_nth_slice, rc=rc)
    else:
        raise AssertionError("Not supported")
    return np.stack((t1, t2, t3), axis=2)
