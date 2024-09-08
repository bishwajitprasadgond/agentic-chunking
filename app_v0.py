# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:25:59 2024

@author: BishwajitPrasadGond
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:14:29 2024

@author: Bishwajit Prasad Gond
"""

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
    prompt = f"""
    You are an agentic chunker. Decompose the content into simple propositions. Here is the content: {text}
    """
    
    # Load a pre-trained LLM from Hugging Face
    generator = pipeline('text-generation', model="EleutherAI/gpt-neo-2.7B")
    
    # Generate the response
    response = generator(prompt, max_length=2000, do_sample=True, temperature=0.7, truncation=True)
    
    # Extracting generated text
    generated_text = response[0]['generated_text']
    
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
        
        # Get the file extension
        file_extension = file.filename.split(".")[-1]
        
        # Extract text from document
        extracted_text = extract_text_from_document(file_bytes, f".{file_extension}")
        
        # Perform agentic chunking
        response = agentic_chunking_function(extracted_text)
        
        return JSONResponse(content={"chunks": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Gradio Interface
def create_gradio_interface():
    demo = gr.Interface(
        fn=chunk_document,
        inputs=gr.File(label="Upload a document"),
        outputs=gr.JSON(label="Agentic chunking output"),
        title="Agentic Chunking",
        description="Upload a document to perform agentic chunking",
    )
    demo.launch()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.6", port=8000)
