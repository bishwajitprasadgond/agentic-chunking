

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:14:29 2024

@author: BishwajitPrasadGond
"""
import os
import requests
import json
import io
import PyPDF2
from transformers import pipeline
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

app = FastAPI()

class ChunkedResponse(BaseModel):
    chunks: list

# Async document chunking function using Langchain
async def agentic_chunking(document_text: str) -> list:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512)  # Adjust chunk size as needed
        chunks = splitter.split_text(document_text)
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chunking: {str(e)}")

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

# Function to perform agentic chunking
def agentic_chunking_function(text):
    # Constructing the prompt
    prompt = f"""
<begin_of_text]></start_header_id/>system</end_header_id/>

## You are an agentic chunker. You will be provided with content.

Decompose the content into clear and simple propositions, ensuring they are interpretable out of context. 
1. Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that").
4. Present the results as a list of strings, formatted in JSON. </eot_id/></start_header_id|>user</end_header_id]>

Here is the content: {text}

Strictly follow the instructions provided and output in the desired format only.

<leot_id]></start_header_id]>
"""

    # Load a pre-trained LLM from Hugging Face (like GPT-Neo)
    generator = pipeline('text-generation', model="EleutherAI/gpt-neo-2.7B")

    # Generate the response with specific settings for warnings
    response = generator(
        prompt, 
        max_length=2000, 
        do_sample=True, 
        temperature=0.7, 
        truncation=True,  # Explicitly setting truncation
        clean_up_tokenization_spaces=False  # To avoid future warning
    )

    # Extracting generated text
    generated_text = response[0]['generated_text']

    # Save the response in a list and JSON format
    propositions = generated_text.split("\n")  # Assuming each proposition is separated by newlines

    # Save to JSON file
    with open('agentic_chunking_output.json', 'w') as json_file:
        json.dump(propositions, json_file, indent=4)

    return propositions

@app.post("/chunk-document", response_model=ChunkedResponse)
async def chunk_document(file: UploadFile = File(...)):
    try:
        # Read the file contents as bytes
        file_bytes = await file.read()
        
        # Extract text from document
        file_extension = os.path.splitext(file.filename)[1]
        extracted_text = extract_text_from_document(file_bytes, file_extension)
        
        # Perform agentic chunking
        response = agentic_chunking_function(extracted_text)
        
        return JSONResponse(content={"chunks": response})
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file content. Ensure the file is in UTF-8 format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def create_gradio_interface():
    # Create Gradio interface
    demo = gr.Interface(
        fn=lambda file: requests.post("http://127.0.0.6:8880/chunk-document", files={"file": file}).json(),
        inputs=gr.File(label="Upload a document"),
        outputs=[gr.Textbox(label="Agentic chunking output"), gr.Label(label="Error")],
        title="Agentic Chunking",
        description="Upload a document to perform agentic chunking",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    import threading
    
    # Start FastAPI server
    def start_uvicorn():
        uvicorn.run(app, host="127.0.0.1", port=8880)

    # Run FastAPI server in a separate thread
    threading.Thread(target=start_uvicorn).start()
    
    # Launch Gradio interface
    create_gradio_interface()
