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
import chromadb  # Using ChromaDB for vector storage
from chromadb.utils import embedding_functions

# Initialize Groq API client
client = Groq(api_key="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")

# Initialize Chroma vector database
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="groq_embeddings")

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extract_text()
        return text
    except Exception as e:
        return str(e)

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return str(e)

# Function to create embeddings using Groq
def create_embeddings(text: str) -> List[float]:
    try:
        response = client.create_embedding(text)
        return response['embedding']
    except Exception as e:
        traceback.print_exc()
        return []

# Function to save embeddings in ChromaDB
def store_in_vectordb(embedding: List[float], metadata: dict):
    collection.add(
        documents=[metadata['text']],
        embeddings=[embedding],
        metadatas=[metadata]
    )

# Function to search embeddings in ChromaDB using a selected similarity method
def search_embeddings(query: str, method: str):
    try:
        # Generate embedding for the query
        query_embedding = create_embeddings(query)

        # Search based on the selected method
        if method == "Cosine Similarity":
            results = collection.query(query_embeddings=[query_embedding], n_results=5, distance_metric="cosine")
        elif method == "Euclidean Distance":
            results = collection.query(query_embeddings=[query_embedding], n_results=5, distance_metric="euclidean")
        elif method == "Manhattan Distance":
            results = collection.query(query_embeddings=[query_embedding], n_results=5, distance_metric="l1")
        else:
            return "Invalid search method."

        # Format the results
        output = ""
        for i, result in enumerate(results['documents'][0]):
            output += f"Result {i+1}:\nDocument: {result}\n\n"
        return output
    except Exception as e:
        return str(e)

# Main function to handle file input and process
def process_file(file: Union[io.BytesIO, str], file_type: str):
    if file_type == "pdf":
        text = extract_text_from_pdf(file)
    elif file_type == "docx":
        text = extract_text_from_docx(file)
    else:
        return "Unsupported file format"

    if not text:
        return "Failed to extract text"

    # Create embeddings
    embeddings = create_embeddings(text)

    if embeddings:
        # Store in vector database
        store_in_vectordb(embeddings, {"text": text, "file_type": file_type})
        return f"Embeddings created and stored for file."
    else:
        return "Failed to create embeddings"

# Gradio interface for file uploads and searching
def gradio_upload(file):
    file_type = file.name.split(".")[-1].lower()
    result = process_file(file.name, file_type)
    return result

def gradio_search(query, method):
    return search_embeddings(query, method)

# Gradio UI with upload and search functionality
with gr.Blocks() as iface:
    # Upload section
    with gr.Row():
        file_input = gr.File(label="Upload a file (PDF/DOCX)")
        file_output = gr.Textbox(label="Upload Status")

    upload_button = gr.Button("Upload File")
    upload_button.click(fn=gradio_upload, inputs=file_input, outputs=file_output)

    # Search section
    with gr.Row():
        query_input = gr.Textbox(label="Enter Query for Search")
        method_dropdown = gr.Dropdown(
            choices=["Cosine Similarity", "Euclidean Distance", "Manhattan Distance"],
            label="Choose Similarity Search Method"
        )
        search_output = gr.Textbox(label="Search Results")

    search_button = gr.Button("Search Embeddings")
    search_button.click(fn=gradio_search, inputs=[query_input, method_dropdown], outputs=search_output)

# Launch Gradio interface
if __name__ == "__main__":
    iface.launch()
