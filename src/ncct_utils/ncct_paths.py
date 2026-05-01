import os
from importlib import resources

resources_path = str(resources.files("ncct_utils").joinpath("resources"))
datafolder = str(resources.files("regutils").joinpath("data"))
scct_unsmooth = os.path.join(resources_path, "scct_unsmooth_RAI_C.nii.gz")
eroded_mask = os.path.join(resources_path, "erodedmask_RAI_C.nii.gz")
head_aura = os.path.join(resources_path, "template_mask_RAI_C.nii.gz")
affine_transform_template = os.path.join(
    resources_path, "affine_transform_template.txt"
)

two_d_transform_template = os.path.join(resources_path, "2dtrans_ident.txt")
three_d_transform_template = os.path.join(resources_path, "3dtrans_ident.txt")


template_registration_parameters = os.path.join(
    resources_path, "affineDTI_template.txt"
)
