import cv2
import numpy as np
from skimage import color, filters, restoration
from skimage.transform import rotate
from PIL import Image, ImageEnhance

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Preprocess the input image for the YOLO model.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Convert to LAB color space for better color handling
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Normalize the image
    img_lab = img_lab.astype(np.float32) / 255.0
    
    return img_lab

def enhance_image(image):
    # Convert image to LAB color space
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Apply histogram equalization to the L channel
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l.astype(np.uint8))  # Ensure L channel is 8-bit unsigned integer

    # Convert L channel back to float32 to match other channels
    l = l.astype(np.float32) / 255.0

    # Merge channels
    img_lab = cv2.merge([l, a, b])

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    return enhanced_img

def color_histogram_comparison(image1, image2):
    """
    Compare color histograms of two images.
    """
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def convert_to_grayscale(image):
    return color.rgb2gray(image)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def adaptive_threshold(image):
    T = filters.threshold_local(image, 11, offset=10, method="gaussian")
    return (image > T).astype(np.uint8) * 255

def denoise_image(image):
    sigma_est = np.mean(restoration.estimate_sigma(image, multichannel=True))
    return restoration.denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, 
                                        patch_size=5, patch_distance=3, multichannel=True)

def deskew(image):
    angles = np.arange(-30, 30, 0.1)
    scores = []
    for angle in angles:
        rotated = rotate(image, angle, resize=True, mode='constant', cval=1)
        score = np.sum(filters.gaussian(np.sum(rotated, axis=1)))
        scores.append(score)
    best_angle = angles[np.argmax(scores)]
    return rotate(image, best_angle, resize=True, mode='constant', cval=1)

def enhance_contrast(image):
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(2)  # Enhance contrast
    return np.array(enhanced).astype(np.float32) / 255.0

def scale_image(image, factor):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * factor), int(h * factor)))

def clean_text(text):
    import re
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def filter_results(results, confidence_threshold=0.5):
    return [result for result in results if result[2] > confidence_threshold]
