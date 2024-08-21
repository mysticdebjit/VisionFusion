import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_segmentation(original_image, segmented_objects):
    """
    Visualize the original image and segmented objects.
    """
    plt.figure(figsize=(12, 6))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Display segmented objects
    plt.subplot(1, 2, 2)
    combined_mask = np.zeros_like(original_image)
    for obj in segmented_objects:
        mask = cv2.threshold(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(obj, obj, mask=mask))
    
    plt.imshow(cv2.cvtColor(combined_mask, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Objects")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def generate_output_image(original_image, segmented_objects, output_path):
    """
    Generate and save the final output image with segmented objects highlighted.
    """
    output_image = original_image.copy()
    for obj in segmented_objects:
        mask = cv2.threshold(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to {output_path}")