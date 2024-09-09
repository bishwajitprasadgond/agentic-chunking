# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:24:15 2024

@author: BishwajitPrasadGond
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:03:04 2024

@author: BishwajitPrasadGond
"""
import os
#from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr
import PyPDF2
import io
from fpdf import FPDF
from typing import Union
#from langchain_community.llms import OpenAI  # Updated import from langchain-community

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
        else:
            return f"Unsupported file type: {file_ext}. Please upload a .txt or .pdf file."

        return text.strip()

    except Exception as e:
        return f"Error reading file: {str(e)}"



#Function to send the extracted text and prompt to OpenAI LLM using Langchain
def generate_response(doc_text: str, prompt: str):
    if not doc_text or not prompt:
        return "Document text or prompt is missing."
    
    try:
        # Combine the document text and prompt
        final_prompt = f"{prompt}\n\nDocument Text:\n{doc_text}"

        # # Load OpenAI API key securely from environment variables
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        # if not openai_api_key:
        #     return "OpenAI API key is missing."

        # Initialize the LLM with the OpenAI API key
        
        llm = OpenAI(model="text-davinci-003", temperature=0.7)
        llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0,
            max_retries=2,
            #api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )

        # Generate the response from the LLM
        response = LLMChain(llm=llm,prompt=final_prompt)
        output = response.run()

        return output.strip()

    except Exception as e:
        return f"Error with LLM call: {str(e)}"

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
            doc_upload = gr.File(label="Upload a Text or PDF Document", file_types=[".txt", ".pdf"])
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
