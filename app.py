import streamlit as st
from components.utils import (
    MODELS,
    visualize_segmentation,
    generate_output_image,
    process_image,
    generate_analysis_string,
    display_file_download,
    display_analysis_results,
    cleanup_previous_analysis,
    display_summary,
)

def main():
    cleanup_previous_analysis()
    st.title("Multi-purpose Analysis Tool")

    col1, col2 = st.columns([2, 1])

    with col1:
        analysis_categories = list(MODELS.keys())
        selected_category = st.selectbox("Select model type:", analysis_categories, key="analysis_category")
        models = MODELS[selected_category]

        analysis_types = list({model_type for model_type in models})
        if len(analysis_types) > 1 and "Detection" in analysis_types and "Segmentation" in analysis_types:
            analysis_types.append("Full Analysis")
        analysis_type = st.selectbox("Select analysis type:", analysis_types, key="analysis_type")

    with col2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())

    if uploaded_file is not None and st.button("🚀 Analyze", key="analyze_button"):
        with st.spinner("🔍 Analyzing image..."):
            results={}
            if analysis_type == "Full Analysis":
                results.update(process_image({selected_category: ["Segmentation"]}, "Segmentation"))
                results.update(process_image({selected_category: ["Detection"]}, "Detection"))
            else:
              results = process_image({selected_category: [analysis_type]}, analysis_type)
            if results:
                st.session_state.analysis_results = results
                analysis_string = generate_analysis_string(st.session_state.analysis_results)
                display_summary(analysis_string)
                st.success("✅ Analysis complete. You can now summarize the results, view extracted text, or download the generated files.")

    display_analysis_results()


if __name__ == "__main__":
    main()