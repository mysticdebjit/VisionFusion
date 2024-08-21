import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess the image by resizing and converting it to RGB."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def enhance_image(image):
    """Enhance the image using CLAHE in the LAB color space."""
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_lab = cv2.merge([l, a, b])
    enhanced_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return enhanced_img

def sharpen_image(image):
    """Sharpen the image using a custom kernel."""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_img = cv2.filter2D(image, -1, kernel)
    return sharpened_img

def show_plots(img, model):
    """Display the original, preprocessed, and YOLO result images."""
    img_enhanced = enhance_image(img)
    img_sharpened = sharpen_image(img_enhanced)
    
    results = model(img_sharpened)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(img_sharpened)
    ax2.set_title("Preprocessed Image")
    ax2.axis('off')
    
    ax3.imshow(results[0].plot())
    ax3.set_title("Gemstone Detection")
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    main()