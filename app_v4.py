# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:43:13 2024

@author: BishwajitPrasadGond
"""

import os
import requests
import io
import PyPDF2
import gradio as gr
from fpdf import FPDF

# Function to extract text from a document
def extract_text_from_document(file_content: bytes, file_extension: str) -> str:
    text = None
    try:
        if file_extension == ".pdf":
            with io.BytesIO(file_content) as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
        elif file_extension == ".txt":
            text = file_content.decode('utf-8')
        else:
            raise ValueError("Unsupported file format. Only .pdf and .txt files are supported.")
        
        if not text:
            raise ValueError("The document is empty or unreadable.")
        
    except Exception as e:
        raise ValueError(f"Error reading document: {e}")
    
    return text

# Function to send request to Groq API
def send_request_to_groq(prompt_text: str) -> list:
    # Construct the Groq query
    query = """
    * | filter(text: contains({text})) | sort(by: ["text", "asc"])
    """.format(text=prompt_text)

    api_key = "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"
    index_name = "your_index_name"
    url = f"https://api.algolia.com/1/indexes/{index_name}/query"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"params": {"query": query}}

    response = requests.get(url, headers=headers, params=params)

    # Parse the response
    if response.status_code == 200:
        return response.json().get("hits", [])
    else:
        raise ValueError(f"Error: {response.status_code} - {response.text}")

# Function to perform agentic chunking and send the request to Groq
def agentic_chunking_and_groq(file):
    file_bytes = file.read()  # Read the file bytes from the file object
    file_extension = os.path.splitext(file.name)[1]  # Extract the file extension
    extracted_text = extract_text_from_document(file_bytes, file_extension)
    
    prompt = f"""
    Decompose the content into clear and simple propositions, ensuring they are interpretable out of context.
    Here is the content: {extracted_text}
    """

    # Send the prompt to Groq API and get response
    results = send_request_to_groq(prompt)
    return results

# Function to save text as PDF
def save_as_pdf(text, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(file_path)

# Gradio interface functions
def process_document(file, output_format):
    try:
        # Perform agentic chunking
        chunked_content = agentic_chunking_and_groq(file)
        
        # Format the output as plain text
        chunked_output = "\n".join([item['text'] for item in chunked_content])

        # Save the result if requested
        if output_format == "Text":
            with open("output.txt", "w") as f:
                f.write(chunked_output)
            return chunked_output, "Output saved as text file."
        elif output_format == "PDF":
            save_as_pdf(chunked_output, "output.pdf")
            return chunked_output, "Output saved as PDF file."
        else:
            return chunked_output, "No file saved."
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    demo = gr.Interface(
        fn=process_document,
        inputs=[gr.File(label="Upload a document"), gr.Radio(["Text", "PDF"], label="Save as")],
        outputs=[gr.Textbox(label="Agentic chunking output"), gr.Label(label="Status")],
        title="Agentic Chunking",
        description="Upload a document to perform agentic chunking and save the output as text or PDF.",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    create_gradio_interface()
