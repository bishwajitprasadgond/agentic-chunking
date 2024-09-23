# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:36:01 2024

@author: BishwajitPrasadGond
"""

#%%

import os
import json
from docx import Document
import PyPDF2
import uuid
import requests
#%%
# Helper function to read DOCX
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Helper function to read PDF
def read_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    full_text = []
    for page in range(len(reader.pages)):
        full_text.append(reader.pages[page].extract_text())
    return "\n".join(full_text)

# Function to create chunks of 2000 words
def chunk_text(text, chunk_size=2000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# Placeholder for sending text to Groq's LLM API (modify according to Groq API)
def process_with_llm(chunk):
    # Example API call to LLM (replace with actual API endpoint)
    url = 'https://api.groq.com/llm'
    response = requests.post(url, json={'text': chunk})
    
    if response.status_code == 200:
        return response.json()  # Assuming API returns topics
    else:
        return {'topics': {}}

# Directory to scan for documents
directory_path = './documents'

# Process each file in the directory
processed_data = []
for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    
    # Extract text based on file type
    if file_name.endswith('.docx'):
        text = read_docx(file_path)
    elif file_name.endswith('.pdf'):
        text = read_pdf(file_path)
    elif file_name.endswith('.txt'):
        with open(file_path, 'r') as file:
            text = file.read()
    else:
        continue  # Skip unsupported file formats
    
    # Create chunks and process each chunk with LLM
    for chunk in chunk_text(text):
        chunk_id = str(uuid.uuid4())  # Generate a unique ID for each chunk
        topics = process_with_llm(chunk)  # Get topics from LLM
        
        processed_data.append({
            'chunk_id': chunk_id,
            'chunk_text': chunk,
            'topics': topics
        })

# Sorting topics based on category
sorted_data = {}
for item in processed_data:
    for topic, value in item['topics'].items():
        if topic not in sorted_data:
            sorted_data[topic] = []
        sorted_data[topic].append({
            'chunk_id': item['chunk_id'],
            'value': value
        })

# Storing the results in JSON format
output_file = 'processed_topics.json'
with open(output_file, 'w') as json_file:
    json.dump(sorted_data, json_file, indent=4)

print(f"Data stored in {output_file}")


#%%


#%%

#%%