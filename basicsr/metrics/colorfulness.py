import cv2
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_cf(img, **kwargs):
    """Calculate Colorfulness.
    """
    (B, G, R) = cv2.split(img.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R+G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)
