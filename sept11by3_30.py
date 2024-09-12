# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:25:26 2024

@author: BishwajitPrasadGond
"""

# Install necessary libraries
#!pip install chromadb openai PyPDF2 gradio docx fpdf pandas groq

# Import necessary modules
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
import gradio as gr
import PyPDF2
from fpdf import FPDF
from typing import List
from docx import Document
import pandas as pd
import json
import traceback
from groq import Groq

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db/"
))

# Create ChromaDB collection
collection = client.create_collection(name="agentic_corpus")

# Initialize embedding function using OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002"
)

# Initialize Groq client
groq_client = Groq(api_key="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")

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
all_documentsdic = {}

# Function to extract text from files
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

# Function to generate response using Groq
def generate_response(doc_text: str, prompt: str, doc_name: str):
    if not doc_text or not prompt:
        return "Document text or prompt is missing."
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": doc_text}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
        )

        response = chat_completion.choices[0].message.content
        document_responses[doc_name] = json.loads(response)

        # Push data to ChromaDB
        push_to_chromadb(doc_name, document_responses[doc_name])

        return response

    except Exception as e:
        return f"Error: {e}\nTraceback:\n{traceback.format_exc()}"

# Function to push data to ChromaDB
def push_to_chromadb(key, response_data):
    heading = response_data.get("heading", "")
    metadata = response_data.get("metadata", "")
    propositions = response_data.get("propositions", [])

    # Add documents to the collection
    collection.add(
        documents=[propositions],  # Propositions are added as documents
        metadatas=[{"metadata": metadata}],  # Metadata
        headings=[heading],  # Heading
        ids=[key]  # Unique key as ID
    )

    # Embed metadata using OpenAI embeddings
    metadata_embeddings = openai_ef([metadata])
    print(f"Stored metadata embeddings: {metadata_embeddings}")

# Gradio UI and Document Processing
def process_document(doc_files, prompt_text):
    combined_responses = ""

    for doc in doc_files:
        doc_text = extract_text_from_files([doc])
        if "Error" in doc_text:
            return doc_text, ""

        prompt = predefined_prompts.get(prompt_text, prompt_text)
        response = generate_response(doc_text, prompt, doc.name)
        combined_responses += f"\nDocument: {doc.name}\nResponse:\n{response}\n"

    return combined_responses, "Process completed successfully."

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Agentic Chunking Prototype")

    with gr.Row():
        with gr.Column():
            doc_upload = gr.Files(label="Upload Documents", file_types=[".txt", ".pdf", ".docx", ".csv"])
            prompt_dropdown = gr.Dropdown(choices=["Agentic Chunking"], label="Select Prompt")
            process_button = gr.Button("Process Documents")

        with gr.Column():
            output_display = gr.Textbox(label="Generated Response", placeholder="Response will appear here...")

    # Process button functionality
    process_button.click(
        process_document, 
        inputs=[doc_upload, prompt_dropdown], 
        outputs=[output_display]
    )

# Run the Gradio app
demo.launch()
