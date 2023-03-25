import numbers
import random

import cv2
import numpy as np
from imgaug import augmenters as iaa


def get_augmentation_pipeline(ours=True):
    """
    Defining the strong augmentation pipeline for contrastive learning based STR model.
    """

    # Ours Augmentation
    augmentations = iaa.Sequential([
        # Color
        iaa.OneOf([
            iaa.ChannelShuffle(p=0.35),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.KMeansColorQuantization(),
            iaa.HistogramEqualization(),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.GammaContrast(gamma=(0.5, 1.5)),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.ChangeColorTemperature((1100, 10000))
        ]),
        # Blurring
        iaa.OneOf([
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
            iaa.OneOf([
                iaa.GaussianBlur((0.5, 1.5)),
                iaa.AverageBlur(k=(2, 6)),
                iaa.MedianBlur(k=(3, 7)),
                iaa.MotionBlur(k=5)
            ]),
            # Noise
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.ImpulseNoise(0.1),
                iaa.MultiplyElementwise((0.5, 1.5))
            ]),

        ]),
        # Geometry, the strength of geometry transformation has been improved than SeqCLR
        iaa.OneOf([
            iaa.LinearContrast((0.5, 1.0)),
            iaa.Crop(percent=((0, 0.4), (0, 0), (0, 0.4), (0, 0.0)), keep_size=True),
            iaa.Crop(percent=((0, 0.0), (0, 0.02), (0, 0.0), (0, 0.02)), keep_size=True),
            iaa.ElasticTransformation(alpha=(0.0, 1.0), sigma=0.25),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        ])
    ])

    return augmentations
