# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:52:30 2024

@author: BishwajitPrasadGond
"""


import os
from groq import Groq
import gradio as gr
import PyPDF2
import io
from fpdf import FPDF
from typing import Union
from docx import Document
import pandas as pd

client = Groq(api_key="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")

# Function to read text or PDF document
def extract_text_from_file(file: Union[gr.File, None]):
    if file is None:
        return "No file uploaded."
    
    file_ext = os.path.splitext(file.name)[1].lower()
    
    try:
        if file_ext == ".txt":
            # Read from text file
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_ext == ".pdf":
            # Read from PDF file
            pdf_reader = PyPDF2.PdfReader(file.name)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        elif file_ext == ".docx":
            # Read from DOCX file
            doc = Document(file.name)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_ext == ".csv":
            # Read from CSV file
            df = pd.read_csv(file.name)
            text = df.to_string(index=False)  # Convert DataFrame to string
        else:
            return f"Unsupported file type: {file_ext}. Please upload a .txt, .pdf, .docx, or .csv file."

        return text.strip()

    except Exception as e:
        return f"Error reading file: {str(e)}"

# Function to send the extracted text and prompt to Groq
def generate_response(doc_text: str, prompt: str):
    if not doc_text or not prompt:
        return "Document text or prompt is missing."
    
    try:
        # Combine the document text and prompt
        final_prompt = f"{prompt}\n\nDocument Text:\n{doc_text}"

        # Send the final prompt and text to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": doc_text
                }
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# Function to create a downloadable text file
def create_text_file(output_text: str):
    file = io.StringIO(output_text)
    return file.getvalue()

# Gradio Interface
def process_document(doc_file, prompt_text):
    doc_text = extract_text_from_file(doc_file)
    if "Error" in doc_text:
        return doc_text, "", None
    
    response = generate_response(doc_text, prompt_text)
    
    # Save the response to a text file
    text_file_content = create_text_file(response)
    file_path = "response.txt"
    
    # Write the response to the file
    with open(file_path, "w") as f:
        f.write(text_file_content)
    
    return response, response, file_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Agentic Chunking Prototype")

    with gr.Row():
        with gr.Column(scale=1):
            doc_upload = gr.File(label="Upload a Document", file_types=[".txt", ".pdf", ".docx", ".csv"])
            prompt_input = gr.Textbox(label="Enter a Custom Prompt", placeholder="Type your prompt here...")
            process_button = gr.Button("Process and Generate Response")
        
        with gr.Column(scale=1):
            output_display = gr.Textbox(label="Generated LLM Response", placeholder="Response will appear here...")
            error_display = gr.Textbox(label="Error Messages", placeholder="Errors will appear here...", interactive=False)
            text_download = gr.File(label="Download as Text File")  # File component for download
    
    process_button.click(
        process_document, 
        inputs=[doc_upload, prompt_input],
        outputs=[output_display, error_display, text_download]
    )

# Run the Gradio app
demo.launch()
