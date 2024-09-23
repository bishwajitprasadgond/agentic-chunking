# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:41:15 2024

@author: BishwajitPrasadGond
"""

#%%
import os
import json
from typing import List, Dict
import PyPDF2
import docx
from PIL import Image
import pytesseract
import csv
import requests  # Assuming we're using an HTTP API for the LLM

# Step 1: Parsing different document types
def parse_document(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return parse_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return parse_docx(file_path)
    elif file_extension.lower() == '.txt':
        return parse_txt(file_path)
    elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
        return parse_image(file_path)
    elif file_extension.lower() == '.csv':
        return parse_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def parse_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return ' '.join(page.extract_text() for page in reader.pages)

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse_image(file_path: str) -> str:
    return pytesseract.image_to_string(Image.open(file_path))

def parse_csv(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return ' '.join(' '.join(row) for row in reader)

# Step 2: Chunking the text
def chunk_text(text: str, max_words: int = 2000) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Step 3: Processing chunks with LLM (mock implementation)
def process_chunk_with_llm(chunk: str) -> Dict[str, List[str]]:
    # This is a mock implementation. Replace with actual API call to Groq's LLM
    # The actual implementation would depend on the specific API and its requirements
    response = requests.post(
        "https://api.groq.com/llm/analyze",
        json={"text": chunk},
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    return response.json()

# Step 4: Sorting and storing results
def sort_and_store_results(results: List[Dict[str, List[str]]]) -> Dict[str, List[Dict[str, str]]]:
    sorted_results = {
        "Date": [],
        "Person": [],
        "Money": [],
        "Other": []
    }
    
    for chunk_id, chunk_result in enumerate(results):
        for topic, sentences in chunk_result.items():
            for sentence in sentences:
                sorted_results[topic].append({
                    "chunk_id": chunk_id,
                    "content": sentence
                })
    
    return sorted_results

# Main processing pipeline
def process_document(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    # Step 1: Parse the document
    text = parse_document(file_path)
    
    # Step 2: Chunk the text
    chunks = chunk_text(text)
    
    # Step 3: Process chunks with LLM
    llm_results = [process_chunk_with_llm(chunk) for chunk in chunks]
    
    # Step 4: Sort and store results
    final_results = sort_and_store_results(llm_results)
    
    # Store results in JSON file
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    return final_results

# Example usage
if __name__ == "__main__":
    file_path = "path/to/your/document.pdf"  # Replace with actual file path
    results = process_document(file_path)
    print("Processing complete. Results stored in 'results.json'.")