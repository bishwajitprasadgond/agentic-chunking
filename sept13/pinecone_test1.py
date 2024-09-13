# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:40:19 2024

@author: BishwajitPrasadGond
"""
#import pdfminer
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
from PyPDF2.errors import PdfReadError
import re
import requests
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

Ensure the format and structure are followed precisely and do not write "Here is the formatted output in a Python dictionary:" """,
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
                with open(file.name, "rb") as pdf_file:
                    try:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        # Check if the PDF is encrypted
                        if pdf_reader.is_encrypted:
                            # Attempt to decrypt (empty string assumes no password)
                            pdf_reader.decrypt("")
                        for page_num in range(len(pdf_reader.pages)):
                            combined_text += pdf_reader.pages[page_num].extract_text() + "\n"
                    except PdfReadError as e:
                        return f"Error reading PDF file {file.name}: {str(e)}"
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


def extract_json_from_response(response: str):
    # Use regex to extract the first JSON-like structure from the response
    try:
        match = re.search(r'{.*}', response, re.DOTALL)
        if match:
            return match.group(0)  # Return the matched string
        else:
            return None
    except Exception as e:
        return f"Error extracting JSON: {e}"

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
        
        # Extract JSON part from response
        extracted_json = extract_json_from_response(response)
        
        if extracted_json:
            try:
                document_responses[doc_name] = json.loads(extracted_json)
                push_to_pinecone(doc_name, document_responses[doc_name])
                return extracted_json  # Return the extracted JSON
            except json.JSONDecodeError:
                return f"Error: Extracted part is not valid JSON. Here is the raw extracted part:\n{extracted_json}"
        else:
            return f"Error: No valid JSON found in the response. Here is the raw response:\n{response}"

    except Exception as e:
        return f"Error: {e}\nTraceback:\n{traceback.format_exc()}"

#%%
def push_to_pinecone(key, response_data):
    heading = response_data.get("heading", "")
    metadata = response_data.get("metadata", "")
    propositions = response_data.get("propositions", [])
    
    # Convert embeddings and propositions to standard Python types
    metadata_embedding = model.encode([metadata])[0].tolist()  # Convert ndarray to list
    propositions_str = "\n".join(propositions)  # Ensure propositions is a single string
    
    index.upsert([
        {
            'id': key,
            'values': metadata_embedding,  # Pass as a list
            'metadata': {
                'heading': heading,
                'metadata': metadata,
                'propositions': propositions_str
            }
        }
    ])
    print(f"Current metadata embedding added: {metadata_embedding}")
#%%

# def search_pinecone(query):
#     try:
#         # Convert query to vector
#         query_vector = model.encode([query])[0].tolist()
        
#         # Search in Pinecone
#         result = index.query(
#             vector=query_vector,
#             top_k=1,
#             include_metadata=True
#         )
        
#         #Check if matches were found
#         if result['matches']:
#             # Extract search results with metadata
#             search_results = [
#                 {
#                     'id': match['id'],
#                     'score': match['score'],
#                     'metadata': match['metadata'].get('propositions', 'No proposition found')  # Safely get the 'propositions' field
#                 }
#                 for match in result['matches']
#             ]
#         else:
#             return "No matches found."
#         # if result['matches']:
#         #     search_results = "\n".join([f"ID: {match['id']}\nScore: {match['score']}\nMetadata: {match['metadata']}" for match in result['matches']])
#         # else:
#         #     search_results = "No matches found."
        
#         # Return the search results
#         return search_results

#     except Exception as e:
#         return f"Error searching Pinecone: {e}"

# #%%
# # Function to send the search results and query to Groq
# def send_to_groq(query, propositions):
#     prompt = (
#         f"Based on the following propositions, answer the query concisely. If the answer is not present, "
#         f"reply 'Answer is not present.'\n\n"
#         f"Propositions:\n{propositions}\n\nQuery: {query}"
#     )
    
#     try:
#         # API call to Groq (adjust the URL and headers according to your Groq API configuration)
#         response = requests.post(
#             'https://api.groq.com/query',  # Replace with actual Groq API endpoint
#             json={'prompt': prompt},
#             headers={'Content-Type': 'application/json'}
#         )
        
#         if response.status_code == 200:
#             return response.json().get('response', 'Answer is not present.')
#         else:
#             return f"Error: Groq API returned status code {response.status_code}"

#     except Exception as e:
#         return f"Error calling Groq API: {e}"
#%%
def search_pinecone(query):
    try:
        # Convert query to vector
        query_vector = model.encode([query])[0].tolist()

        # Search in Pinecone
        result = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )

        # Check if matches were found
        if result['matches']:
            # Extract search results with metadata
            search_results = [
                {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata'].get('propositions', 'No proposition found')  # Safely get the 'propositions' field
                }
                for match in result['matches']
            ]
            # Convert search results to string for later use
            search_results_str = "\n".join(
                [f"ID: {res['id']}\nScore: {res['score']}\nPropositions: {res['metadata']}" for res in search_results]
            )
            return search_results_str
        else:
            return "No matches found."

    except Exception as e:
        return f"Error searching Pinecone: {e}"

# Function to send the search results and query to Groq
def send_and_get_reply_groq(query, search_results_str):
    if not search_results_str or not query:
        return "Document text or query is missing."
    
    # Formulate the prompt using search results
    prompt = (
        f"Based on the following propositions, answer the query concisely. If the answer is not present, "
        f"reply 'Answer is not present.'\n\n"
        f"Propositions:\n{search_results_str}\n\nQuery: {query}"
    )
    
    try:
        # Send the prompt to the Groq model for a response
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": search_results_str}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
        )
        
        response = chat_completion.choices[0].message.content
        
        # Extract JSON part from response (if applicable)
        extracted_json = extract_json_from_response(response)
        
        if extracted_json:
            try:
                document_responses[query] = json.loads(extracted_json)
                push_to_pinecone(query, document_responses[query])
                return extracted_json  # Return the extracted JSON
            except json.JSONDecodeError:
                return f"Error: Extracted part is not valid JSON. Here is the raw extracted part:\n{extracted_json}"
        else:
            return f"Error: No valid JSON found in the response. Here is the raw response:\n{response}"

    except Exception as e:
        return f"Error: {e}\nTraceback:\n{traceback.format_exc()}"


#%%
# Main function to execute the search and send propositions to Groq
def process_query(query):
    search_results = search_pinecone(query)
    
    if isinstance(search_results, str):
        return search_results
    
    propositions = "\n".join([f"{result['proposition']}" for result in search_results])
    groq_response = send_and_get_reply_groq(query, propositions)
    
    return groq_response
#%%
def process_document(doc_files, prompt_text):
    combined_responses = ""
    for doc in doc_files:
        doc_text = extract_text_from_files([doc])
        if "Error" in doc_text:
            return doc_text, "", "", status_dashboard("Error processing document")
        prompt = predefined_prompts.get(prompt_text, prompt_text)
        response = generate_response(doc_text, prompt, doc.name)
        combined_responses += f"\nDocument: {doc.name}\nResponse:\n{response}\n"
    return combined_responses, "", "Process completed successfully.", status_dashboard("Process completed successfully")
#%%
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
        answer_box = gr.Textbox(label="Answer to your query", placeholder="Answer will appear here...")
        status_display = gr.Textbox(label="Status", placeholder="Status will appear here...")

    process_button.click(
        process_document,
        inputs=[doc_upload, prompt_dropdown],
        outputs=[output_display, answer_box, status_display]
    )

    search_button.click(
        process_query,
        inputs=[search_box],
        outputs=[answer_box]
    )

demo.launch()

