# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:55:58 2024

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
import json

client = Groq(api_key="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")

# Predefined prompts
predefined_prompts = {
    "Agentic Chunking": """## You are an agentic chunker. You will be provided with content.

    Decompose the content into clear and simple propositions, ensuring they are interpretable out of context. 
    1. Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
    3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that").""",
    "Sentence Chunking": "Please break down the text into individual sentences and list them."
}

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

# Functions to create downloadable files
def create_text_file(output_text: str):
    return io.StringIO(output_text).getvalue()

def create_csv_file(output_text: str):
    df = pd.DataFrame([output_text], columns=["Response"])
    return df.to_csv(index=False)

def create_json_file(output_text: str):
    data = {"response": output_text}
    return json.dumps(data, indent=4)

# Gradio Interface
def process_document(doc_file, prompt_text, format_choice):
    doc_text = extract_text_from_file(doc_file)
    if "Error" in doc_text:
        return doc_text, "", None, None, None
    
    # Determine the actual prompt to use
    prompt = predefined_prompts.get(prompt_text, prompt_text)

    response = generate_response(doc_text, prompt)
    
    # Save the response to files
    if format_choice == "Text":
        file_content = create_text_file(response)
        file_path = "response.txt"
    elif format_choice == "CSV":
        file_content = create_csv_file(response)
        file_path = "response.csv"
    elif format_choice == "JSON":
        file_content = create_json_file(response)
        file_path = "response.json"
    else:
        return "Unsupported format choice.", "", None, None, None
    
    with open(file_path, "w") as f:
        f.write(file_content)
    
    return response, response, file_path, file_path, file_path

# Gradio UI
with gr.Blocks(css="""
    .container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .button {
        background-color: orange !important;  /* Ensure the background color is applied */
        color: white !important;  /* Ensure the text color is applied */
    }
    .header-logo {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    .header-logo img {
        max-width: 8%;
        height: auto;
    }
""") as demo:
    gr.Markdown("## Agentic Chunking Prototype")

    # Add Logo at the top
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <div class="header-logo">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" alt="IBM Logo">
                </div>
            """)

    with gr.Row():
        with gr.Column(scale=1):
            doc_upload = gr.File(label="Upload a Document ( Docx, Pdf, Txt, CSV )", file_types=[".txt", ".pdf", ".docx", ".csv"])
            prompt_dropdown = gr.Dropdown(
                choices=["Select a Prompt", "Agentic Chunking", "Sentence Chunking"],
                label="Select or Enter a Custom Prompt",
                value="Select a Prompt"
            )
            custom_prompt = gr.Textbox(label="Enter Custom Prompt (if needed)", placeholder="Type your prompt here...", visible=False)
            format_dropdown = gr.Dropdown(
                choices=["Text", "CSV", "JSON"],
                label="Select Output Format",
                value="Text"
            )
            process_button = gr.Button("Process and Generate Response", elem_id="process-button")
        
        with gr.Column(scale=1):
            output_display = gr.Textbox(label="Generated LLM Response", placeholder="Response will appear here...")
            error_display = gr.Textbox(label="Error Messages", placeholder="Errors will appear here...", interactive=False)
            text_download = gr.File(label="Download File")  # File component for download
    
    # Show custom prompt textbox only if "Custom Prompt" is selected
    def update_prompt_visibility(selected_prompt):
        return gr.update(visible=selected_prompt == "Enter Custom Prompt")
    
    prompt_dropdown.change(
        update_prompt_visibility,
        inputs=prompt_dropdown,
        outputs=custom_prompt
    )
    
    process_button.click(
        process_document, 
        inputs=[doc_upload, prompt_dropdown, format_dropdown],
        outputs=[output_display, error_display, text_download, text_download, text_download]
    )

# Run the Gradio app
demo.launch()
