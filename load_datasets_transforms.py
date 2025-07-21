from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    ResizeWithPadOrCropd,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd,
    #ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    MapTransform,
)

import numpy as np
from collections import OrderedDict
import glob

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
        
def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir))

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16
    elif dataset == 'mo':
        out_classes = 14
    elif dataset == 'BTCV8':
        out_classes = 9
    elif dataset == 'BTCV13':
        out_classes = 14
    elif dataset == 'Task01_BrainTumour':
        out_classes = 3 # for sigmoid
    elif dataset == 'Task10_Colon':
        out_classes = 2
    elif dataset == 'Task06_Lung':
        out_classes = 2
    elif dataset == 'Task02_Heart':
        out_classes = 2
    elif dataset == 'Task03_Liver':
        out_classes = 3
    elif dataset == 'Task04_Hippocampus':
        out_classes = 3
    elif dataset == 'Task09_Spleen':
        out_classes = 2
    elif dataset == 'Task07_Pancreas':
        out_classes = 3
    elif dataset == 'Task08_HepaticVessel':
        out_classes = 3
    elif dataset == 'Task05_Prostate':
        out_classes = 3

    if (args.mode == 'train' or args.mode == 'validation'):
        train_samples = {}
        valid_samples = {}

        ## Input training data
        #print(os.path.join(root_dir, 'imagesTr', 'x.nii.gz'))
        train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        #print(len(train_img))
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        ## Input validation data
        valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input inference data
        test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes


def data_transforms(args):
    dataset = args.dataset
    if (args.mode == 'train' or args.mode == 'validation'):
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'feta':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'flare':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeWithPadOrCropd(keys=["image"], spatial_size=(168,168,128), mode=("constant")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'amos':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512,512,512), mode=("constant")), # added by me
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size, #96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )
    elif dataset == 'BTCV8':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512,512,192), mode=("constant")), # added by me
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size, #96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )
    elif dataset == 'BTCV13':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                #AddChanneld(keys=["image", "label"]),
                #ScaleIntensityRanged(
                #    keys=["image"], a_min=-125, a_max=275,
                #    b_min=0.0, b_max=1.0, clip=True,
                #),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512,512,192), mode=("constant")), # added by me
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                EnsureTyped(keys=["image", "label"]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size, #96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                #AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                #ScaleIntensityRanged(
                #    keys=["image"], a_min=-125, a_max=275,
                #    b_min=0.0, b_max=1.0, clip=True,
                #),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                #AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                #ScaleIntensityRanged(
                #    keys=["image"], a_min=-125, a_max=275,
                #    b_min=0.0, b_max=1.0, clip=True,
                #),
                #ToTensord(keys=["image"]),
            ]
        )  
    elif dataset == 'Task09_Spleen':
        train_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True,),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from big
                # image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50,),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
                RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=args.img_size, rotate_range=(0, 0, np.pi / 30), scale_range=(0.1, 0.1, 0.1)),
            ]
        )

        # NOTE: No random cropping in the validation data,
        # we will evaluate the entire image using a sliding window.
        val_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-125, # -57
                    a_max=275, # 164
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        ) 
        test_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-125, # -57
                    a_max=275, # 164
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        ) 
    
    elif dataset == 'Task03_Liver':
        train_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True,),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from big
                # image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.10,),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.20,),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.20,),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.20,),
                RandRotate90d(keys=["image", "label"], prob=0.20, max_k=3,),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=args.img_size, rotate_range=(0, 0, np.pi / 30), scale_range=(0.1, 0.1, 0.1)),
            ]
        )

        # NOTE: No random cropping in the validation data,
        # we will evaluate the entire image using a sliding window.
        val_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-21, # -57
                    a_max=189, # 164
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        test_transforms = Compose(
            [
                # LoadImaged with image_only=True is to return the MetaTensors
                # the additional metadata dictionary is not returned.
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.0, 1.0, 1.0), #(1.5, 1.5, 2.0)
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-21, # -57
                    a_max=189, # 164
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        ) 

    elif dataset == 'Task01_BrainTumour':
        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandSpatialCropd(keys=["image", "label"], roi_size=args.img_size, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"), #new added
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )  
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image"]),
                #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear"),
                ),
                CropForegroundd(keys=["image"], source_key="image"), #new added
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    elif dataset == 'Task05_Prostate':
        train_transforms = Compose(
            [
                # load 2 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                AddChanneld(keys="label"),
                EnsureTyped(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest"),),#
                #SpatialPadd(keys=["image", "label"], spatial_size=args.img_size),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                CropForegroundd(keys=["image", "label"], source_key="image"), #new added
                #RandSpatialCropd(keys=["image", "label"], roi_size=args.img_size, random_size=False),
                RandCropByPosNegLabeld(
            	    keys=["image", "label"],
            	    label_key="label",
            	    spatial_size=args.img_size,
            	    pos=1,
            	    neg=1,
            	    num_samples=crop_samples,
            	    image_key="image",
            	    image_threshold=0,
        	    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3,),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size, #(-1, -1, -1), #
                    rotate_range=(0, 0, np.pi),
                    scale_range=(0.3, 0.3, 0.0)),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                AddChanneld(keys="label"),
                EnsureTyped(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest"),),#0.5, 0.5, 0.5
                #SpatialPadd(keys=["image", "label"], spatial_size=args.img_size),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                CropForegroundd(keys=["image", "label"], source_key="image"), # new added
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear"),),#0.5, 0.5, 0.5
                #SpatialPadd(keys=["image", "label"], spatial_size=args.img_size),
                Orientationd(keys=["image"], axcodes="RAS"),
                CropForegroundd(keys=["image"], source_key="image"), # new added
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    elif dataset == 'Task10_Colon':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-57,
            	    a_max=175,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image", "label"], source_key="image"),
        	Orientationd(keys=["image", "label"], axcodes="RAS"),
        	Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        	RandCropByPosNegLabeld(
            	    keys=["image", "label"],
            	    label_key="label",
            	    spatial_size=args.img_size,
            	    pos=1,
            	    neg=1,
            	    num_samples=crop_samples,
            	    image_key="image",
            	    image_threshold=0,
        	    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3,),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        	RandAffined(
        	    keys=['image', 'label'],
        	    mode=('bilinear', 'nearest'),
        	    prob=1.0, spatial_size=args.img_size,
        	    rotate_range=(0, 0, np.pi/15),
        	    scale_range=(0.1, 0.1, 0.1)),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-57,
            	    a_max=175,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image", "label"], source_key="image"),
        	Orientationd(keys=["image", "label"], axcodes="RAS"),
        	Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-57,
            	    a_max=175,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image"], source_key="image"),
        	Orientationd(keys=["image"], axcodes="RAS"),
        	Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ]
        )
        
    elif dataset == 'Task06_Lung':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-1000,
            	    a_max=1000,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	    ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=2,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.30,
                    max_k=3,
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)
                ),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-1000,
            	    a_max=1000,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image", "label"], source_key="image"),
        	Orientationd(keys=["image", "label"], axcodes="RAS"),
        	Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-1000,
            	    a_max=1000,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image"], source_key="image"),
        	Orientationd(keys=["image"], axcodes="RAS"),
        	Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ]
        )
    
    elif dataset == 'Task07_Pancreas':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-87,
            	    a_max=199,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	    ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.25,
                    max_k=3,
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),                
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-87,
            	    a_max=199,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image", "label"], source_key="image"),
        	Orientationd(keys=["image", "label"], axcodes="RAS"),
        	#Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=-87,
            	    a_max=199,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image"], source_key="image"),
        	Orientationd(keys=["image"], axcodes="RAS"),
            ]
        )

    elif dataset == 'Task04_Hippocampus':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),                
        	    Spacingd(keys=["image", "label"], pixdim=(0.2, 0.2, 0.2), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image", "label"], source_key="image"),
        	    Orientationd(keys=["image", "label"], axcodes="RAS"),
        	    RandCropByPosNegLabeld(
            	    keys=["image", "label"],
            	    label_key="label",
            	    spatial_size=args.img_size,
            	    pos=1,
            	    neg=1,
            	    num_samples=4,
            	    image_key="image",
            	    image_threshold=0,
        	    ),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
        	    RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),                
        	    Spacingd(keys=["image", "label"], pixdim=(0.2, 0.2, 0.2), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image", "label"], source_key="image"),
        	    Orientationd(keys=["image", "label"], axcodes="RAS"),                
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),                
        	    Spacingd(keys=["image"], pixdim=(0.2, 0.2, 0.2), mode=("bilinear")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image"], source_key="image"),
        	    Orientationd(keys=["image"], axcodes="RAS"),                
            ]
        )
        
    elif dataset == 'Task08_HepaticVessel':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=0,
            	    a_max=230,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	    ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.img_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.25,
                    max_k=3,
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),                
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=0,
            	    a_max=230,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image", "label"], source_key="image"),
        	Orientationd(keys=["image", "label"], axcodes="RAS"),
        	#Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
            	    a_min=0,
            	    a_max=230,
            	    b_min=0.0,
            	    b_max=1.0,
            	    clip=True,
        	),
        	CropForegroundd(keys=["image"], source_key="image"),
        	Orientationd(keys=["image"], axcodes="RAS"),
        	#Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ]
        )
    
    elif dataset == 'Task02_Heart':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),                
        	    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image", "label"], source_key="image"),
        	    Orientationd(keys=["image", "label"], axcodes="RAS"),
        	    RandCropByPosNegLabeld(
            	    keys=["image", "label"],
            	    label_key="label",
            	    spatial_size=args.img_size,
            	    pos=2,
            	    neg=1,
            	    num_samples=4,
            	    image_key="image",
            	    image_threshold=0,
        	    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                    spatial_size=args.img_size,
                    mode='constant'
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        	    RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=args.img_size,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)
                ),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),                
        	    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image", "label"], source_key="image"),
        	    Orientationd(keys=["image", "label"], axcodes="RAS"),                
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),                
        	    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        	    CropForegroundd(keys=["image"], source_key="image"),
        	    Orientationd(keys=["image"], axcodes="RAS"),                
            ]
        )

    if args.mode == 'train' or args.mode == 'validation':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return val_transforms, test_transforms

def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        #EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            #device="cpu",
        ),
        AsDiscreted(keys="pred", argmax=True),#, to_onehot=out_classes),#, to_onehot=out_classes
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="", output_ext=".nii.gz", resample=True),
    ])
    

    return post_transforms
    
def infer_post_transforms_brats(args, test_transforms, out_classes):

    post_transforms = Compose([
        #EnsureTyped(keys="pred"),
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            #device="cpu",
        ),
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="", output_ext=".nii.gz", resample=True),
    ])
    

    return post_transforms



