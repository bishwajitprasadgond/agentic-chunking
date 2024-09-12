import os
from transformers import AutoModel, AutoTokenizer
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
from chromadb.config import Settings

# Initialize the model and tokenizer from transformers
model_name = 'jinaai/jina-embeddings-v2-base-en'
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb"  # Ensure this directory exists
))

# Ensure the collection does not already exist
collection_name = "groq_embeddings"
if collection_name not in [coll['name'] for coll in chroma_client.list_collections()]:
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

# Function to create embeddings using Transformers model
def create_embeddings(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = embedding_model(**inputs)
        # Typically, you'd use the last hidden state or pooler output for embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Averaging over token embeddings
        return embedding
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        return None

# Function to add document embeddings to ChromaDB
def add_to_chromadb(text: str, doc_id: str):
    embedding = create_embeddings(text)
    if embedding:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id]
        )
        return "Embedding added successfully."
    else:
        return "Failed to create embedding."

# Function to search embeddings with different similarity methods
def search_in_chromadb(query: str, similarity_method: str):
    embedding = create_embeddings(query)
    if embedding:
        search_result = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            similarity=similarity_method  # Support different similarity methods
        )
        matched_info = ' '.join(item['metadata']['text'] for item in search_result['matches'])
        sources = [item['metadata']['source'] for item in search_result['matches']]
        context = f"Information: {matched_info} and the sources: {sources}"
        sys_prompt = f"""
        Instructions:
        - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
        - Utilize the context provided for accurate and specific information.
        - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
        - Cite your sources
        Context: {context}
        """
        return sys_prompt
    else:
        return "Failed to create query embedding."

# Gradio UI for embedding addition and search
def add_document_to_collection(file: Union[str, io.BytesIO], doc_type: str):
    try:
        text = ""
        if doc_type == "pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        elif doc_type == "docx":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = file.read().decode("utf-8")
        
        result = add_to_chromadb(text, doc_type + "_doc")
        return result
    except Exception as e:
        traceback.print_exc()
        return str(e)

# Gradio UI for searching
def search(query: str, similarity_method: str):
    response = search_in_chromadb(query, similarity_method)
    return response

# Dropdown options for similarity methods
similarity_methods = ["cosine", "dot_product", "euclidean"]

# Gradio Interface
with gr.Blocks() as app:
    with gr.Tab("Add Document"):
        file_input = gr.File(label="Upload Document", type="file")
        doc_type = gr.Radio(choices=["pdf", "docx", "txt"], label="Document Type")
        add_button = gr.Button("Add to ChromaDB")
        result_text = gr.Textbox(label="Result")

        add_button.click(
            fn=add_document_to_collection, 
            inputs=[file_input, doc_type], 
            outputs=[result_text]
        )
    
    with gr.Tab("Search"):
        query_input = gr.Textbox(label="Search Query")
        similarity_method_input = gr.Dropdown(choices=similarity_methods, label="Similarity Method")
        search_button = gr.Button("Search")
        search_results = gr.Textbox(label="Search Results")

        search_button.click(
            fn=search, 
            inputs=[query_input, similarity_method_input], 
            outputs=[search_results]
        )

app.launch()
