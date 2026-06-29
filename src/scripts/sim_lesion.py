import imageio
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import SimpleITK as sitk
from torch import imag
from dicomutils.dicomutils import sitk2generic_ct
from ncct_utils import ncct_paths
from ncct_utils import rncct_config
from ncct_utils.rNCCTfunctions import import_input, template_reg
from regutils.simpleelastix_utils import itk2sitk, itk_resample, sitk2itk, xform_point_set
from dicomutils.imageutils import sitk2montage
from pathlib import Path
from ncct_utils.rncct_config import rNCCTConfig
from ncct_utils.rncct_process import  process
import imageio.v2 as iio
NCCT_PATH = ""


def simcase(ncct: sitk.Image, depression: float, radius: float = 10,coord_x_y_z = (142, 237, 10),
            motionscheme: list[tuple[float,float,float,float,float,float]] =((0,0,0,0,0,0),)) -> dict[str,tuple[sitk.Image,sitk.Image]]:
    motion_img_dict = {}
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

    ball_mask_img = sitk.GetImageFromArray(ball_mask.astype(np.uint8))
    ball_mask_img.CopyInformation(ncct)
    ncctHU4_img = sitk.GetImageFromArray(ncctHU)
    ncctHU4_img.CopyInformation(ncct)

    for motion in motionscheme:
        transform = sitk.Euler3DTransform()
        #transform.SetCenter(0,0,0)
        transform.SetRotation(*motion[0:3])
        transform.SetTranslation(motion[3:6])
        ncctHU4_img_rot = sitk.Resample(ncctHU4_img, transform, sitk.sitkLinear, -1024, ncctHU4_img.GetPixelID())
        mask = sitk.Resample(ball_mask_img, transform, sitk.sitkNearestNeighbor, 0, ball_mask_img.GetPixelID())
        motion_img_dict[f"rot_{motion[0]}_{motion[1]}_{motion[2]}_trans_{motion[3]}_{motion[4]}_{motion[5]}"] = (ncctHU4_img_rot,mask)
    return motion_img_dict


def template_sim():
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



def subject_symmetry():
    # read in case with this slices, register to template space with affine. Flip unaffected hemisphere to affected. 
    cases_folder = Path("EXTERNAL_DATA")
    outputfolder = Path("EXTERNAL_DATA") / "sims"
    outputfolder.mkdir(exist_ok=True, parents=True)
    laterality = ["L","L","R"]
    template_image  = sitk.ReadImage(ncct_paths.scct_unsmooth)
    template_center = template_image.TransformContinuousIndexToPhysicalPoint( [s/2 for s in template_image.GetSize()])
    for indx,case in enumerate(sorted(cases_folder.glob("C*nii"))):
        img = sitk.ReadImage(case ) 
        oututfolder_reg = outputfolder / f"{case.stem}_template_reg"
        oututfolder_reg.mkdir(exist_ok=True, parents=True)
        t2n_xfm, n2t_xfm,_ = template_reg(
        origncct=img,
        cachemode=True,
        fixedmask=sitk.ReadImage(ncct_paths.head_aura),
        outputfolder=oututfolder_reg,
        registration_parameters_url=ncct_paths.template_registration_parameters
        )
        # cast to float
        img = sitk.Cast(img, sitk.sitkFloat32)
        #transform to template space, but keep the original spacing and size of the image.
        # the native FOV is larger, so we need to recalculate the origin to maintain the image centered in template space.
        # The fix point is the center of the image in template space and the FOV of the original image.

        # the directions are orthonormal so we can use the spacing and size to calculate the origin in template space.
        origin_in_template_space = template_center - np.array(img.GetSize()) * np.array(img.GetSpacing()) / 2

        n2t_xfm.SetParameter(0,"Spacing", [f"{img.GetSpacing()[0]}", f"{img.GetSpacing()[1]}", f"{img.GetSpacing()[2]}"])
        n2t_xfm.SetParameter(0,"Size", [f"{img.GetSize()[0]}", f"{img.GetSize()[1]}", f"{img.GetSize()[2]}"])
        n2t_xfm.SetParameter(0,"Origin", [f"{origin_in_template_space[0]}", f"{origin_in_template_space[1]}", f"{origin_in_template_space[2]}"])
        img_resampled = itk_resample(sitk2itk(img), n2t_xfm)
        img_resampled_sitk = sitk.DICOMOrient(itk2sitk(img_resampled))

        sitk2montage(sitkimg=img_resampled_sitk, opname= outputfolder / f"{case.stem}_resampled_CASE.png",
                     value_range=(0, 60),every_nth_slice=5)
        sitk2montage(sitkimg=template_image, opname= outputfolder / f"{case.stem}_resampled_REFR.png",
                     value_range=(0, 60),every_nth_slice=5)
        
        img_resampled_arr = sitk.GetArrayFromImage(img_resampled_sitk)
        if laterality[indx] == "L":
            img_resampled_arr[:,:,(img_resampled_arr.shape[2]//2):] = np.flip(img_resampled_arr[:,:,0:img_resampled_arr.shape[2]//2],axis=2)
        else:
            img_resampled_arr[:,:,0:img_resampled_arr.shape[2]//2] = np.flip(img_resampled_arr[:,:,img_resampled_arr.shape[2]//2:],axis=2)
        
        img_resampled_mirr_sitk = sitk.GetImageFromArray(img_resampled_arr)
        img_resampled_mirr_sitk.CopyInformation(img_resampled_sitk)      
        sitk2montage(sitkimg=img_resampled_mirr_sitk, opname= outputfolder / f"{case.stem}_resampled_MIRR.png",
                     value_range=(0, 60),every_nth_slice=5)
        
        sitk.WriteImage(img_resampled_mirr_sitk, outputfolder / f"{case.stem}_resampled_symmetric.nii.gz")   


def symmetry_lesion_sim():
    z_rotations = [np.deg2rad(angle) for angle in [-20,-10,0,10,20]]
    y_rotations = [np.deg2rad(angle) for angle in [-10,0,10]]

    motionscheme = [(0, y_rot, z_rot, 0, 0, 0) for z_rot in z_rotations for y_rot in y_rotations]
    depression = 3
    for case in sorted(Path("EXTERNAL_DATA/sims").glob("*resampled_symmetric.nii.gz")):
        img = sitk.ReadImage(case)

        coord_x_y_z = (162, 237,img.GetSize()[2]//2) 



        sim_imgs = simcase(img, depression=depression, radius=10, coord_x_y_z=coord_x_y_z,motionscheme=motionscheme)
        
        for sim_name, (sim_img, sim_mask) in sim_imgs.items():
            sitk.WriteImage(sim_img, case.parent / f"{case.stem[0]}_{sim_name}.nii.gz")
            sitk2montage(sitkimg=sim_img, opname= case.parent / f"{case.stem[0]}_{sim_name}.png",
                            value_range=(0, 60),every_nth_slice=5)
            sitk2montage(sitkimg=sim_img, value_range=(0, 60), maskovl=sim_mask, opname= case.parent / f"{case.stem[0]}_{sim_name}_mask.png",
                            every_nth_slice=5)



def run_symmetric_with_lesion():
    for case in sorted(Path("EXTERNAL_DATA/sims").glob("*_rot*trans*.nii.gz")):

        runconfig = rNCCTConfig(input=str(case), output=str(case.parent / f"{case.stem}_rncct"), caching=True,thin2thick=True, debug=True)
        process(runconfig)


def movie():

    for case in ["A","B","C"]:
        vid =  imageio.get_writer(f'/home/sorenc/CODE/hypodensity/EXTERNAL_DATA/{case}_video.mp4', fps=4,codec="libx264",
format="FFMPEG")
        for sim in sorted(Path("EXTERNAL_DATA/sims").glob(f"{case}_*rncct")):
            rncct_w_ovl = imageio.imread(str(sim / "10_depression_rgb" / "rNCCT_A.png"))
            vid.append_data(rncct_w_ovl)
        vid.close()

if __name__ == "__main__":
    #template_sim()
    #subject_symmetry()




    symmetry_lesion_sim()

    #run_symmetric_with_lesion()

    #movie()