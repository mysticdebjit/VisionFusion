import os
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize the language model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ['GOOGLE_API_KEY'])

# Define the prompt template for summarization
summarize_template = """
You are an AI assistant specializing in summarizing analysis results from various models. 
Given the following analysis results, provide a concise and informative summary:

{analysis_results}

Focus on key details such as:
1. Types of objects or elements detected
2. Confidence levels of detections
3. Any notable features, characteristics, or patterns
4. Relevant metrics or measurements

Provide a clear and structured summary that captures the essence of the analysis:
"""

summarize_prompt = PromptTemplate(
    input_variables=["analysis_results"],
    template=summarize_template
)

# Initialize the LLMChain for summarization
summarize_chain = LLMChain(
    llm=llm,
    prompt=summarize_prompt,
    verbose=True
)

def summarize_analysis(analysis_results):
    try:
        summary = summarize_chain.predict(analysis_results=analysis_results)
        return summary
    except Exception as e:
        print(f"Error: {e}")
        return "I encountered an error while summarizing the analysis. Please check the input and try again."

# Function to generate a structured analysis string
def generate_analysis_string(results):
    analysis_string = "Analysis Results:\n"
    for model_name, model_results in results.items():
        analysis_string += f"\nModel: {model_name}\n"
        for result in model_results:
            analysis_string += f"- Detected: {result['name']} (Confidence: {result['confidence']:.2f})\n"
            if 'attributes' in result:
                for attr, value in result['attributes'].items():
                    analysis_string += f"  {attr}: {value}\n"
    return analysis_string

# Example usage
if __name__ == "__main__":
    # This would be replaced with actual analysis results in your application
    sample_results = {
        "Object Detection Model": [
            {"name": "Car", "confidence": 0.95, "attributes": {"color": "red", "model": "sedan"}},
            {"name": "Person", "confidence": 0.88, "attributes": {"pose": "standing"}}
        ],
        "Segmentation Model": [
            {"name": "Road", "confidence": 0.97, "attributes": {"condition": "good"}},
            {"name": "Sky", "confidence": 0.99, "attributes": {"weather": "clear"}}
        ]
    }
    
    analysis_string = generate_analysis_string(sample_results)
    summary = summarize_analysis(analysis_string)
    print("Summary of Analysis:")
    print(summary)