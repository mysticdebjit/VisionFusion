import cv2
import numpy as np

def apply_mask(image, mask):
    """
    Apply a binary mask to an image.
    """
    return cv2.bitwise_and(image, image, mask=mask)

def refine_mask(mask, kernel_size=3):
    """
    Refine the mask using morphological operations.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask

def get_contours(mask):
    """
    Get contours from a binary mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours