import cv2
import numpy as np
from skimage.segmentation import clear_border

def lp_filter(image):
    """
    Applies thresholding and morphological operations to license plate
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

    thresh = cv2.erode(thresh, kernel, iterations = 1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 100:
            result[labels == i + 1] = 255
    
    return result, clear_border(result)