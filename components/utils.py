import os
import logging
import traceback
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from ultralytics.engine.results import Results
from ultralytics import YOLO
from models.objectclassification import preprocess_image, enhance_image, sharpen_image, show_plots
from models.segmentation import analyze_image, csv_return, get_csv_path, initialize_csv
from models.summarizer import summarize_analysis
from models.textextraction import process_text
import shutil
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    "Gem Analysis": {
        "Detection": "best weights/classification/bestclassi.pt",
        "Segmentation": "best weights/segmentation/best.pt"
    },
    "Glioblastoma Early Diagnosis": {
        "Segmentation": "modelsroboflow/brain/brain.pt",
        "Detection": "modelsroboflow/brain/brain.pt"
    },
    "Simple Image Text Extraction": {
        "Extract": None  # No model needed for text extraction
    }
}

def visualize_segmentation(original_image, segmented_objects):
    """
    Visualize the original image and segmented objects.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display original image
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Display segmented objects
    combined_mask = np.zeros_like(original_image)
    for obj in segmented_objects:
        mask = cv2.threshold(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(obj, obj, mask=mask))

    ax2.imshow(cv2.cvtColor(combined_mask, cv2.COLOR_BGR2RGB))
    ax2.set_title("Segmented Objects")
    ax2.axis("off")

    st.pyplot(fig)

def generate_output_image(original_image, segmented_objects):
    """
    Generate the final output image with segmented objects highlighted.
    """
    output_image = original_image.copy()
    for obj in segmented_objects:
        mask = cv2.threshold(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    return output_image

def process_image(selected_models, analysis_type):
    results = {}
    try:
        logger.info(f"Starting image analysis with type: {analysis_type}")


        for category, models in selected_models.items():
            st.subheader(f"{category} Results")
            for model_name in models:
                st.write(f"Running {model_name}...")
                if model_name == "Extract":
                    # Perform text extraction
                    extracted_text = process_text("temp_image.jpg")
                    results["Text Extraction"] = extracted_text
                    st.session_state.extracted_text = extracted_text
                else:
                  model_path = MODELS[category][model_name]
                  model = YOLO(model_path)
                  try:                # Use the analyze_image function from the segmentation code
                    st.session_state.master_folder = analyze_image("temp_image.jpg", model)
                    st.success(f"Image processed successfully. Results saved in folder: {st.session_state.master_folder}")

                    img_path = os.path.join(st.session_state.master_folder, "original_image.png")
                    img = preprocess_image(img_path)
                    img_enhanced = enhance_image(img)
                    img_sharpened = sharpen_image(img_enhanced)

                    model_results = model(img_sharpened)
                    results[model_name] = []

                    for r in model_results:
                        if isinstance(r, Results):
                            for box in r.boxes:
                                results[model_name].append({
                                    "name": r.names[int(box.cls)],
                                    "confidence": float(box.conf),
                                    "bbox": box.xyxy.tolist()[0]
                                })

                    try:
                        fig = show_plots(img, model)
                        st.pyplot(fig)
                        plt.savefig(os.path.join(st.session_state.master_folder, f"{model_name}_result.png"))
                        logger.info(f"{model_name} results plotted successfully")
                    except Exception as e:
                        logger.error(f"Error in show_plots for {model_name}: {str(e)}")
                        st.error(f"Error in show_plots for {model_name}: {str(e)}")
                        st.error(traceback.format_exc())
                  except ValueError as e:
                    st.error(str(e))
                    st.warning("No objects were detected in the image.")


        # Generate CSV
        df_data = []
        for model, items in results.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and 'name' in item and 'confidence' in item:
                        df_data.append((model, item['name'], item['confidence']))

        if df_data:
            df = pd.DataFrame(df_data, columns=['Model', 'Object', 'Confidence'])
            csv_path = os.path.join(st.session_state.master_folder, "results.csv")
            df.to_csv(csv_path, index=False)
            st.session_state.csv_path = csv_path
        else:
            logger.warning("No valid data to create CSV")

        return results

    except Exception as e:
        logger.error(f"Error in process_image function: {str(e)}")
        st.error(f"Error: {str(e)}")
        st.error(traceback.format_exc())
        return None

def generate_analysis_string(results):
    analysis_string = "Analysis Results:\n"
    for model_name, model_results in results.items():
        analysis_string += f"\nModel: {model_name}\n"
        for result in model_results:
            analysis_string += f"- Detected: {result['name']} (Confidence: {result['confidence']:.2f})\n"
            if 'bbox' in result:
                analysis_string += f"  Bounding Box: {result['bbox']}\n"
    return analysis_string

def display_file_download(folder_path):
    files = os.listdir(folder_path)
    file_types = {".png": "📷", ".csv": "📊", ".txt": "📝"}

    st.write("Available files:")
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_ext = os.path.splitext(file)[1]
        icon = file_types.get(file_ext, "📄")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="fade-in">{icon} {file}</div>', unsafe_allow_html=True)
        with col2:
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=file,
                    mime="application/octet-stream",
                    key=f"download_{file}"
                )



def display_analysis_results():
    if hasattr(st.session_state, 'analysis_results'):
        st.header("Analysis Results")
        # col1= st.columns(1)

        # with col1:
        #     if st.button("📊 Summarize Results", key="summarize_button"):
        #         with st.spinner("Generating summary..."):

        #             summary = summarize_analysis(analysis_string)
        #             with st.expander("View Summary", expanded=False):
        #                 st.markdown(f'<div class="fade-in">{summary}</div>', unsafe_allow_html=True)
        #     else:
        #         with st.expander("View Summary", expanded=False):
        #             st.markdown(f'<div class="fade-in">Click the "Summarize Results" button to generate the analysis summary.</div>', unsafe_allow_html=True)

        # with col1:
        if hasattr(st.session_state, 'extracted_text'):
            st.subheader("Extracted Text")
            with st.expander("View Extracted Text", expanded=False):
                st.markdown(f'<div class="fade-in">{st.session_state.extracted_text}</div>', unsafe_allow_html=True)
        else:
            st.subheader("Extracted Text")
            with st.expander("View Extracted Text", expanded=False):
                st.markdown(f'<div class="fade-in">No extracted text available. Please run the analysis first.</div>', unsafe_allow_html=True)

        with st.expander("📈 Display CSV Results", expanded=False):
            try:
                df = pd.read_csv(get_csv_path())
                st.dataframe(df)
            except FileNotFoundError:
                st.markdown(f'<div class="fade-in">CSV file not found. Please run the analysis first.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())

        st.subheader("📁 Download Results")
        if hasattr(st.session_state, 'master_folder'):
            with st.expander("Available Files", expanded=False):
                display_file_download(st.session_state.master_folder)
        else:
            st.markdown(f'<div class="fade-in">No files available for download. Please run the analysis first.</div>', unsafe_allow_html=True)
        

def display_summary(analysis):
    st.subheader("Summary of Analysis")
    if analysis is not None:
      with st.spinner("Generating summary..."):

          summary = summarize_analysis(analysis)
          with st.expander("View Summary", expanded=False):
              st.markdown(f'<div class="fade-in">{summary}</div>', unsafe_allow_html=True)
    else:
      with st.expander("View Summary", expanded=False):
        st.markdown(f'<div class="fade-in">Click the "Summarize Results" button to generate the analysis summary.</div>', unsafe_allow_html=True)

        
def cleanup_previous_analysis():
    if hasattr(st.session_state, 'master_folder') and os.path.exists(st.session_state.master_folder):
        shutil.rmtree(st.session_state.master_folder)
        st.session_state.pop('master_folder', None)
        st.session_state.pop('csv_path', None)
        st.session_state.pop('extracted_text', None)
        st.session_state.pop('analysis_results', None)
    CSV_FILE = get_csv_path()
    # Reset the CSV file
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    initialize_csv()