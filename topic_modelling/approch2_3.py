# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:25:44 2024

@author: BishwajitPrasadGond
"""

import os
import json
from docx import Document
import PyPDF2
import uuid
import spacy
from datetime import datetime

# Load spaCy model (English in this case, can be changed based on language)
nlp = spacy.load("en_core_web_sm")
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

# Function to extract entities using spaCy
def extract_entities(chunk_text):
    doc = nlp(chunk_text)
    entities = {
        "date": [],
        "person": [],
        "money": []
    }

    # Extract specific entities and append to lists
    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["date"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["person"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text)
    
    return entities

# Function to split text into sentences based on periods (".")
def split_sentences_by_period(text):
    doc = nlp(text)
    sentences = []
    sentence_buffer = []
    
    for token in doc:
        sentence_buffer.append(token.text)
        if token.text == ".":  # Check if the token is a period
            sentences.append(" ".join(sentence_buffer).strip())
            sentence_buffer = []
    
    if sentence_buffer:
        sentences.append(" ".join(sentence_buffer).strip())  # Add the last part if not empty
    
    return sentences
#%%
# Directory to scan for documents
directory_path = r'C:\Users\BishwajitPrasadGond\bishu'

# Process each file in the directory
processed_data = {}
chunk_counter = 1
#%%
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
    
    # Create chunks and process each chunk with NLP
    for chunk in chunk_text(text):
        chunk_id = str(uuid.uuid4())  # Generate a unique ID for each chunk
        entities = extract_entities(chunk)  # Extract entities with spaCy
        
        # Split the chunk into sentences based on periods (".")
        sentences = split_sentences_by_period(chunk)
        sentence_dict = {f"Sentence {i+1}": sentence for i, sentence in enumerate(sentences)}

        # Metadata
        metadata = {
            "docid": file_path,
            "timestamp": str(datetime.now())
        }

        # Storing each chunk with its own chunk_id
        processed_data[chunk_counter] = {
            "chunk_id": chunk_id,
            "date": entities["date"],  # Store dates as list
            "person": entities["person"],  # Store persons as list
            "money": entities["money"],  # Store money as list
            "metadata": metadata,
            "sentence": sentence_dict,
            "original": chunk  # Store original chunk text
        }
        
        chunk_counter += 1

# Storing the results in JSON format
output_file = 'processed_chunks.json'
with open(output_file, 'w') as json_file:
    json.dump(processed_data, json_file, indent=4)

print(f"Data stored in {output_file}")
