import numpy as np
import SimpleITK as sitk
from dicomutils.dicomutils import sitk2generic_ct
from ncct_utils import ncct_paths
from ncct_utils.rNCCTfunctions import import_input, template_reg
from regutils.simpleelastix_utils import itk2sitk, itk_resample, sitk2itk

from pathlib import Path

NCCT_PATH = ""


def simcase(ncct: sitk.Image, depression: float, radius: float = 10) -> sitk.Image:
    coord_x_y_z = (142, 237, 10)
    spacing = ncct.GetSpacing()
    size = ncct.GetSize()
    z, y, x = np.meshgrid(
        np.arange(1, size[2] + 1),
        np.arange(1, size[1] + 1),
        np.arange(1, size[0] + 1),
        indexing="ij",
    )
    dist = (
        np.square(spacing[2] * (z - coord_x_y_z[2]))
        + np.square(spacing[1] * (y - coord_x_y_z[1]))
        + np.square(spacing[0] * (x - coord_x_y_z[0]))
    )
    dist = np.sqrt(dist)
    dist[coord_x_y_z[2] - 1, coord_x_y_z[1] - 1, coord_x_y_z[0] - 1] = 0

    ball_mask = dist < radius

    ncctHU = sitk.GetArrayFromImage(ncct)

    ncctHU[ball_mask] = ncctHU[ball_mask] - depression

    ncctHU4_img = sitk.GetImageFromArray(ncctHU)
    ncctHU4_img.CopyInformation(ncct)

    return ncctHU4_img


if __name__ == "__main__":
    tdir = Path("/home/sorenc/Desktop/NCCT_SIM/")
    tdir.mkdir(exist_ok=True)
    # read in case
    dicom_url = Path(NCCT_PATH)

    ncct_imported, _ = import_input(dicom_url, output_path=tdir)
    ncct_itk = sitk2itk(ncct_imported)
    # register to template
    t2n, n2t, _ = template_reg(
        origncct=ncct_imported, outputfolder=tdir, cachemode=True
    )

    t2n.SetParameter("DefaultPixelValue", "-1024")
    template_like_native = itk_resample(
        sitk2itk(sitk.ReadImage(ncct_paths.scct_unsmooth)), t2n
    )

    template_like_native_sitk = itk2sitk(template_like_native)

    sitkimg = simcase(template_like_native_sitk, depression=2, radius=10)

    sitk.WriteImage(sitkimg, tdir / "simulated_lesion.nii.gz")

    sitk2generic_ct(sitkimg, tdir / "DCM")
