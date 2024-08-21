#segmentaion.py

import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from utils.preprocessing import preprocess_image
from utils.postprocessing import apply_mask, refine_mask
from utils.data_mapping import create_object_mapping, save_mapping, load_mapping
from utils.visualization import visualize_segmentation

CSV_FILE = "analysis_log.csv"

def initialize_csv():
    """Initialize the CSV file if it does not exist."""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["master_id", "object_id", "image_name", "output_path"])
        df.to_csv(CSV_FILE, index=False)

def update_csv(master_id, object_id, image_name, output_path):
    """Update the CSV file with new entries."""
    df = pd.read_csv(CSV_FILE)
    new_entry = pd.DataFrame([[master_id, object_id, image_name, output_path]],
                             columns=["master_id", "object_id", "image_name", "output_path"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def get_csv_path():
    """Return the path of the CSV file."""
    return os.path.abspath(CSV_FILE)

def analyze_image(image_path, model):
    """Analyzes an image using the provided YOLO model.
    
    Args:
        image_path (str): Path to the image to be analyzed.
        model: YOLO model instance to use for analysis.
    
    Returns:
        str: The path to the folder containing the analysis results.
    """
    # Initialize CSV
    initialize_csv()

    # Get the filename of the image
    image_name = os.path.basename(image_path)

    # Load the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}")

    # Check if the image has already been processed
    df = pd.read_csv(CSV_FILE)
    if not df[df["image_name"] == image_name].empty:
        raise ValueError(f"Image {image_name} has already been processed.")

    # Create a folder for the new image
    master_id = len(df) + 1  # Unique ID for the new image
    master_folder = f"analysis/{master_id}"
    if not os.path.exists(master_folder):
        os.makedirs(master_folder)

    # Save the original image
    cv2.imwrite(f"{master_folder}/original_image.png", original_img)
    print(f"Original image saved to {master_folder}/original_image.png")

    # Create a copy for YOLO processing
    yolo_img = original_img.copy()

    # Get the detections
    results = model(yolo_img)

    # Initialize a counter for unique object IDs
    object_id = 1

    # Initialize mapping list
    mapping = []

    # Save the segmented objects as PNG images in the "analysis" folder
    for r in results:
        masks = r.masks  # Segmentation masks

        if masks is not None:
            for mask in masks.data:
                # Convert mask to numpy array and resize to match image dimensions
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (original_img.shape[1], original_img.shape[0]))

                # Refine the mask
                binary_mask = refine_mask((mask_np > 0.5).astype(np.uint8) * 255)

                # Apply the mask to the original image
                segmented_object = apply_mask(original_img, binary_mask)

                # Save the segmented object in the "analysis" folder
                object_image_name = f"object_{object_id}.png"
                output_path = f"{master_folder}/{object_image_name}"
                cv2.imwrite(output_path, segmented_object)

                # Update CSV
                update_csv(master_id, object_id, object_image_name, output_path)

                # Create and add mapping
                mapping.append(create_object_mapping(master_id, object_id, object_image_name))

                print(f"Object {object_id} saved in analysis folder")
                object_id += 1
        else:
            # If no masks are available, use bounding boxes instead
            boxes = r.boxes
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                segmented_object = original_img[y1:y2, x1:x2]
                object_image_name = f"object_{object_id}.png"
                output_path = f"{master_folder}/{object_image_name}"
                cv2.imwrite(output_path, segmented_object)

                # Update CSV
                update_csv(master_id, object_id, object_image_name, output_path)

                # Create and add mapping
                mapping.append(create_object_mapping(master_id, object_id, object_image_name))

                print(f"Object {object_id} saved in analysis folder")
                object_id += 1

    # Save the mapping
    save_mapping(mapping, master_folder)

    if len(mapping) == 0:
        raise ValueError("No objects detected in the image.")
    else:
        print(f"Analysis complete. {len(mapping)} objects detected and saved in {master_folder}")

    # Save a copy of the original image with bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(f"{master_folder}/original_with_boxes.png", original_img)
    print(f"Original image with bounding boxes saved to {master_folder}/original_with_boxes.png")

    return master_folder

def csv_return():
    """Return the contents of the CSV file as a DataFrame."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return df
    else:
        raise FileNotFoundError("CSV file does not exist.")



if __name__ == "__main__":
  main()

