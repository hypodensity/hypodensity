import os

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import Conv2d, MaxPool2d, Parameter, ReLU, Softmax, Upsample

import regutils.simpleelastix_utils as sutl

from . import ncct_paths

# Define the neural network
layer_weights: dict[str, dict[str, np.ndarray]] = {}
with h5py.File(os.path.join(ncct_paths.resources_path, "last_weights.h5"), "r") as f:
    for group_key_0 in f:
        for group_key_1 in f[group_key_0]:
            layer_weights[group_key_1] = {}
            for group_key_2 in f[group_key_0][group_key_1]:
                layer_weights[group_key_1][group_key_2] = f[group_key_0][group_key_1][
                    group_key_2
                ][()]


def get_conv_layer(name, inchannels, outchannels, kernel_size=(3, 3)):
    conv_layer = Conv2d(
        inchannels, outchannels, kernel_size=kernel_size, padding="same"
    )
    conv_layer.weight = Parameter(
        torch.from_numpy(np.transpose(layer_weights[name]["kernel:0"], (3, 2, 0, 1))),
        requires_grad=False,
    )
    conv_layer.bias = Parameter(
        torch.from_numpy(layer_weights[name]["bias:0"]), requires_grad=False
    )
    return conv_layer


def get_unet_pytorch(channels=3):
    def my_unet_function(x):
        x = ReLU()(get_conv_layer("conv2d_1", channels, 32)(x))

        x = conv1 = ReLU()(get_conv_layer("conv2d_2", 32, 32)(x))
        x = MaxPool2d((2, 2))(x)

        x = ReLU()(get_conv_layer("conv2d_3", 32, 64)(x))

        x = conv2 = ReLU()(get_conv_layer("conv2d_4", 64, 64)(x))
        x = MaxPool2d((2, 2))(x)

        x = ReLU()(get_conv_layer("conv2d_5", 64, 128)(x))

        x = ReLU()(get_conv_layer("conv2d_6", 128, 128)(x))

        x = Upsample(scale_factor=2)(x)
        x = torch.cat([conv2, x], axis=1)
        x = ReLU()(get_conv_layer("conv2d_7", 192, 64)(x))

        x = ReLU()(get_conv_layer("conv2d_8", 64, 64)(x))

        x = Upsample(scale_factor=2)(x)
        x = torch.cat([conv1, x], axis=1)
        x = ReLU()(get_conv_layer("conv2d_9", 96, 32)(x))

        x = ReLU()(get_conv_layer("conv2d_10", 32, 32)(x))

        x = ReLU()(get_conv_layer("conv2d_11", 32, 2, kernel_size=(1, 1))(x))
        x = Softmax(dim=1)(x)

        return x

    return my_unet_function


model = get_unet_pytorch(channels=3)


def csf_seg(img: sitk.Image, mask: sitk.Image) -> sitk.Image:
    mat = sitk.GetArrayFromImage(img).astype(np.float32)
    maskmat = sitk.GetArrayFromImage(mask)

    if mat.shape[1] > 512:  # split images in lower and upper 512 respectively
        mat1 = np.expand_dims(mat[:, 0:512, :], 0)
        mat2 = np.expand_dims(mat[:, -513:-1, :], 0)
        matconcat = np.concatenate((mat1, mat2), axis=0)

        maskmat1 = np.expand_dims(maskmat[:, 0:512, :], 0)
        maskmat2 = np.expand_dims(maskmat[:, -513:-1, :], 0)
        maskmatconcat = np.concatenate((maskmat1, maskmat2), axis=0)
    elif mat.shape[1] == 512:
        matconcat = np.expand_dims(mat, 0)
        maskmatconcat = np.expand_dims(maskmat, 0)
    else:  # so smaller - zero pad
        mat1 = np.zeros((mat.shape[0], 512, 512), np.float32)
        mat1[:, 0 : mat.shape[1], 0 : mat.shape[2]] = mat
        maskmat1 = np.zeros((mat.shape[0], 512, 512), np.uint8)
        maskmat1[:, 0 : mat.shape[1], 0 : mat.shape[2]] = maskmat
        matconcat = np.expand_dims(mat1, 0)
        maskmatconcat = np.expand_dims(maskmat1, 0)

    is_w512 = True if maskmatconcat.shape[3] == 512 else False

    actual_w = maskmatconcat.shape[3]
    if not is_w512:
        crop_w1 = int(np.round((actual_w - 512) / 2))
        crop_w2 = actual_w - 512 - crop_w1
        precropdims = matconcat.shape
        matconcat = matconcat[:, :, :, crop_w1:-crop_w2]
        maskmatconcat = maskmatconcat[:, :, :, crop_w1:-crop_w2]

    predmat = np.zeros(matconcat.shape, np.float32)
    for iframe in range(matconcat.shape[0]):
        cmat = matconcat[iframe, :, :, :]
        cmaskmat = maskmatconcat[iframe, :, :, :]
        cmat[cmaskmat < 0.99] = 0

        for islice in range(1, cmat.shape[0] - 1):
            X = cmat[islice - 1 : islice + 2, :, :]
            X[X < 0] = 0
            X = np.expand_dims(X, 0)
            y = model(torch.from_numpy(X)).numpy()

            predmat[iframe, islice, :, :] = y[0, 0, :, :]

    # now resynth if >512
    if mat.shape[1] > 512:
        predmat_synth = np.zeros(mat.shape, np.float32)
        predmat_synth_indicator = np.zeros(mat.shape, bool)
        predmat_synth_indicator[:, 0:512, :] = 1
        predmat_synth_indicator[:, -513:-1, :] = (
            predmat_synth_indicator[:, -513:-1, :] + 1
        )  # overlap region is now 2
        predmat_synth[:, 0:512, :] = predmat[0, :, :, :]
        predmat_synth[:, -513:-1, :] = predmat[1, :, :, :]
        predmat_synth[predmat_synth_indicator == 2] = (
            0.5 * predmat_synth[predmat_synth_indicator == 2]
        )
        predmat = predmat_synth
    elif mat.shape[1] == 512:
        predmat = predmat[0, :, :, :]
    else:
        predmat = predmat[0, :, 0 : mat.shape[1], 0 : mat.shape[2]]

    if not is_w512:
        predmat_tmp = np.zeros(precropdims[1:], np.float32)
        predmat_tmp[:, :, crop_w1:-crop_w2] = predmat
        predmat = predmat_tmp

    predmatimg = sutl.arr2img(predmat, img)

    return predmatimg
