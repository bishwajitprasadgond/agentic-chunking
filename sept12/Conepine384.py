# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:25:26 2024

@author: BishwajitPrasadGond
"""
import os
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
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
#%%
# Initialize Pinecone with API key
pc = Pinecone(api_key="b0b34b7c-6454-481f-bde9-9f9bd1b008e2")
index_name = 'agentic-app'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
#%%
load_dotenv()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
groq_client = Groq(api_key="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")
#%%
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
#%%
document_responses = {}

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
#%%
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
        try:
            document_responses[doc_name] = json.loads(response)
        except json.JSONDecodeError:
            return f"Error: Response is not valid JSON. Here is the raw response:\n{response}"
        push_to_pinecone(doc_name, document_responses[doc_name])
        return response
    except Exception as e:
        return f"Error: {e}\nTraceback:\n{traceback.format_exc()}"
#%%
def push_to_pinecone(key, response_data):
    heading = response_data.get("heading", "")
    metadata = response_data.get("metadata", "")
    propositions = response_data.get("propositions", [])
    metadata_embedding = model.encode([metadata])
    propositions_str = "\n".join(propositions)
    index.upsert([
        {
            'id': key,
            'values': metadata_embedding[0].tolist(),
            'metadata': {
                'heading': heading,
                'metadata': metadata,
                'propositions': propositions_str
            }
        }
    ])
    print(f"Current metadata embedding added: {metadata_embedding.tolist()}")

def search_pinecone(query):
    try:
        result = index.query(
            vector=model.encode([query])[0],
            top_k=5,
            include_metadata=True
        )
        if result['matches']:
            search_results = "\n".join([f"ID: {match['id']}\nScore: {match['score']}\nMetadata: {match['metadata']}" for match in result['matches']])
        else:
            search_results = "No matches found."
        return search_results
    except Exception as e:
        return f"Error searching Pinecone: {e}"
#%%
def process_document(doc_files, prompt_text):
    combined_responses = ""
    for doc in doc_files:
        doc_text = extract_text_from_files([doc])
        if "Error" in doc_text:
            return doc_text, "", status_dashboard("Error processing document")
        prompt = predefined_prompts.get(prompt_text, prompt_text)
        response = generate_response(doc_text, prompt, doc.name)
        combined_responses += f"\nDocument: {doc.name}\nResponse:\n{response}\n"
    return combined_responses, "Process completed successfully.", status_dashboard("Process completed successfully")

def status_dashboard(status_message: str):
    return f"Status: {status_message}"

with gr.Blocks() as demo:
    gr.Markdown("## Agentic Chunking Prototype")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload and Process Documents")
            doc_upload = gr.Files(label="Upload Documents", file_types=[".txt", ".pdf", ".docx", ".csv"])
            prompt_dropdown = gr.Dropdown(choices=["Agentic Chunking"], label="Select Prompt")
            process_button = gr.Button("Process Documents")
        
        with gr.Column():
            gr.Markdown("### Search Pinecone")
            search_box = gr.Textbox(label="Search Query", placeholder="Enter search query here...")
            search_button = gr.Button("Search Pinecone")

    with gr.Row():
        output_display = gr.Textbox(label="Generated Response", placeholder="Response will appear here...")
        status_display = gr.Textbox(label="Status", placeholder="Status will appear here...")

    process_button.click(
        process_document,
        inputs=[doc_upload, prompt_dropdown],
        outputs=[output_display, status_display]
    )

    search_button.click(
        search_pinecone,
        inputs=[search_box],
        outputs=[output_display]
    )

demo.launch()
#%%
index = pc.Index(index_name)
index.describe_index_stats()
