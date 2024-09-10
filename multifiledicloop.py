# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:55:38 2024

@author: BishwajitPrasadGond
"""

import os
from groq import Groq
import gradio as gr
import PyPDF2
import io
from fpdf import FPDF
from typing import Union, List
from docx import Document
import pandas as pd
import json
import traceback

client = Groq(api_key="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")

# Predefined prompts
predefined_prompts = {
    "Agentic Chunking": """You are tasked with organizing content using "Agentic Chunking." Your goal is to format the content into three distinct sections: Heading, Metadata, and Propositions.

1. **Heading**: Create a short, descriptive heading for the content.
2. **Metadata**: Provide a brief summary of the content in 100 words. This summary should encapsulate the essence of the instructions related to decomposing content into simple propositions, handling compound sentences, and decontextualizing propositions.
3. **Propositions**: Decompose the content into clear and simple propositions, ensuring they are interpretable out of context. Follow these guidelines:
   - Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
   - For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
   - Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with explicit terms.

**Important:** The final output should be formatted as a Python dictionary with the following keys:
- "heading"
- "metadata"
- "propositions"

Here’s an example of how the output should be structured:
```python
{
    "heading": "",
    "metadata": "",
    "propositions": []
}
```

Ensure the format and structure are followed precisely and dont write "Here is the formatted output in a Python dictionary:" """,
    "Sentence Chunking": "Please break down the text into individual sentences and list them.",
    "Summarize": "Please summarize the content in 100 words."
}

# Global dictionary to store the LLM responses for each document
document_responses = {}

# Function to read text or PDF documents
def extract_text_from_files(files: List[gr.File]):
    combined_text = ""
    for file in files:
        file_ext = os.path.splitext(file.name)[1].lower()
        
        try:
            if file_ext == ".txt":
                with open(file.name, "r", encoding="utf-8") as f:
                    combined_text += f.read() + "\n"
            elif file_ext == ".pdf":
                pdf_reader = PyPDF2.PdfReader(file.name)
                for page_num in range(len(pdf_reader.pages)):
                    combined_text += pdf_reader.pages[page_num].extract_text() + "\n"
            elif file_ext == ".docx":
                doc = Document(file.name)
                combined_text += "\n".join(paragraph.text for paragraph in doc.paragraphs) + "\n"
            elif file_ext == ".csv":
                df = pd.read_csv(file.name)
                combined_text += df.to_string(index=False) + "\n"
            else:
                return f"Unsupported file type: {file_ext}. Please upload a .txt, .pdf, .docx, or .csv file."
        
        except Exception as e:
            return f"Error reading file {file.name}: {str(e)}"
    
    return combined_text.strip()

# Function to send the extracted text and prompt to Groq and store in dictionary
def generate_response(doc_text: str, prompt: str, doc_name: str):
    if not doc_text or not prompt:
        return "Document text or prompt is missing."
    
    try:
        final_prompt = f"{prompt}\n\nDocument Text:\n{doc_text}"

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

        response = chat_completion.choices[0].message.content

        # Store response in the global dictionary with the document name as the key
        document_responses[doc_name] = response

        return response

    except Exception as e:
        error_trace = traceback.format_exc()
        return f"An error occurred: {e}\nTraceback:\n{error_trace}"

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
def process_document(doc_files, prompt_text, format_choice):
    combined_responses = ""
    
    # Process each document and store the response
    for doc in doc_files:
        doc_text = extract_text_from_files([doc])
        if "Error" in doc_text:
            return doc_text, "", None, None, None
        
        prompt = predefined_prompts.get(prompt_text, prompt_text)
        
        response = generate_response(doc_text, prompt, doc.name)
        combined_responses += f"\nDocument: {doc.name}\nResponse:\n{response}\n"
    
    document_responses["All_Documents"] = combined_responses

    # Select file format for download
    if format_choice == "Text":
        file_content = create_text_file(combined_responses)
        file_path = "response.txt"
    elif format_choice == "CSV":
        file_content = create_csv_file(combined_responses)
        file_path = "response.csv"
    elif format_choice == "JSON":
        file_content = create_json_file(combined_responses)
        file_path = "response.json"
    else:
        return "Unsupported format choice.", "", None, None, None
    
    with open(file_path, "w") as f:
        f.write(file_content)
    
    return combined_responses, "Process completed successfully.", file_path, file_path, file_path

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
            doc_upload = gr.Files(label="Upload Multiple Documents ( Docx, Pdf, Txt, CSV )", file_types=[".txt", ".pdf", ".docx", ".csv"])
            prompt_dropdown = gr.Dropdown(
                choices=["Select a Prompt", "Agentic Chunking", "Sentence Chunking", "Summarize"],
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
