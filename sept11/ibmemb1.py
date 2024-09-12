import os
from dotenv import load_dotenv
from genai.schema import (
    DecodingMethod,
    GenerateParams,
    Credentials,
    ModerationParameters,
    ModerationSocialBias,
    ModerationSocialBiasInput,
    ModerationSocialBiasOutput
)
from genai.extensions.langchain import LangChainInterface
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import chromadb
from chromadb.config import Settings

load_dotenv()

# Get the API key and URL from the environment variables
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)

# Create a Credentials object to pass to the LangChainInterface
creds = Credentials(api_key, api_endpoint=api_url)

# Create a GenerateParams object to pass to the LangChainInterface
params = GenerateParams(
    decoding_method=DecodingMethod.GREEDY,
    max_new_tokens=1000,
    min_new_tokens=200,
    temperature=0.7,
    moderations=ModerationParameters(
        social_bias=ModerationSocialBias(
            input=ModerationSocialBiasInput(enabled=True, threshold=0.8),
            output=ModerationSocialBiasOutput(enabled=True, threshold=0.8)
        )
    )
)

# Create a LangChainInterface object to use for generating text
llm = LangChainInterface(model="meta-llama/llama-2-70b-chat", params=params, credentials=creds)

# Function to create embeddings using LangChainInterface
def create_embeddings(text: str):
    try:
        # Assuming `LangChainInterface` has a method to generate embeddings
        # Adjust method as necessary based on actual API
        response = llm.embed_text(text)  # Hypothetical method
        return response  # Adjust based on the actual response format
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        return None

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb"  # Ensure this directory exists
))

# Ensure the collection does not already exist
collection_name = "ibm_embeddings"
if collection_name not in [coll['name'] for coll in chroma_client.list_collections()]:
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

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
        return search_result
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
    results = search_in_chromadb(query, similarity_method)
    if isinstance(results, str):
        return results
    output = ""
    for doc_id, score in zip(results["ids"][0], results["distances"][0]):
        output += f"Document ID: {doc_id}, Similarity Score: {score}\n"
    return output

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
