# Asynchronous Agentic Document Chunking API

## Overview

This repository contains the implementation of an asynchronous API for Agentic Chunking of user-provided documents. The API is built with FastAPI and integrates several key components to parse, process, and visualize documents. The goal is to efficiently handle document chunking operations and provide a user-friendly interface for testing and previewing the results.

## Key Components

1. **FastAPI**: Provides the core asynchronous API framework for handling document uploads and processing.
2. **Langchain**: Facilitates the chunking of documents and interactions with language models.
3. **Groq**: Assists with understanding and processing document chunks in a meaningful way.
4. **Gradio**: Offers a user interface for testing the API and visualizing the chunking results.

## Solution Outline

### 1. FastAPI Microservice

- **Objective**: Create an asynchronous API endpoint to handle document uploads and chunking operations.
- **Details**: The API will accept files and return the processed chunks asynchronously.

### 2. Agentic Chunking with Langchain

- **Objective**: Implement a chunking mechanism using Langchain to split documents into logical chunks.
- **Details**: Define the chunking logic and integrate with Llava for deeper understanding and processing of these chunks.

### 3. Gradio UI for Testing

- **Objective**: Develop a Gradio-based user interface to upload documents and visualize chunking results.
- **Details**: The UI will interact with the FastAPI endpoint to enable users to test document processing and view the results.

### 4. Asynchronous Processing

- **Objective**: Ensure all operations and API responses are handled asynchronously using FastAPI's async capabilities.
- **Details**: Implement async processing for all document chunking tasks to ensure efficient handling and responsiveness.

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Langchain
- Llava
- Gradio
- Uvicorn (for running the FastAPI server)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/async-document-chunking-api.git
   cd async-document-chunking-api
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the FastAPI server:**

   ```bash
   uvicorn main:app --reload
   ```

2. **Launch the Gradio UI:**

   ```bash
   python gradio_ui.py
   ```

3. **Navigate to the Gradio interface URL** (usually `http://127.0.0.1:7860`) to upload documents and test the API.



## Contribution

Feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2 License - see the [LICENSE](LICENSE) file for details.



