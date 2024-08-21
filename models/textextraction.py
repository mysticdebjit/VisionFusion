import pytesseract
import cv2
import matplotlib.pyplot as plt

import pytesseract
import cv2
import matplotlib.pyplot as plt

# Function to preprocess the image and enhance text detection
def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

    # Denoise the image (optional but improves detection)
    denoised_img = cv2.fastNlMeansDenoising(thresh_img, h=30)

    # Resize the image for better detection (upscale by 150%)
    resized_img = cv2.resize(denoised_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized_img

# Function to detect text from the preprocessed image
def process_text(image_path):
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Use pytesseract to do OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_img)

    # Display the preprocessed image
    plt.imshow(preprocessed_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Print the extracted text
    return text

if __name__ == "__main__":
    main()