import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_EMBED_NAME = os.getenv("OPENAI_EMBED_NAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Validate environment variables
if not OPENAI_EMBED_NAME or not OPENAI_BASE_URL:
    raise ValueError("Environment variables OPENAI_EMBED_NAME or OPENAI_BASE_URL are not set.")

try:
    # Initialize embedding model
    embed_model = OllamaEmbeddings(
        model=OPENAI_EMBED_NAME,
        base_url=OPENAI_BASE_URL
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize Ollama embeddings: {e}")

def create_and_persist_vector_store(file_path: str, persist_directory: str):
    """
    Reads a file, splits the text into chunks, embeds it, and persists the vector store.
    """
    try:
        # Read the file content
        with open(file_path, "r") as f:
            text = f.read()

        # Initialize the text splitter
        text_splitter = SemanticChunker(embed_model)
        chunks = text_splitter.split_text(text)

        # Create the vector store and persist it
        vector_store = Chroma.from_texts(chunks, embed_model, persist_directory=persist_directory)

        print(f"Vector store created and persisted at: {persist_directory}")
        return vector_store

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the vector store: {e}")

if __name__ == "__main__":
    file_path = "gcp_architect_guide.txt"
    persist_directory = "./chroma_db"

    # Validate file path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Create and persist vector store
    vector_store = create_and_persist_vector_store(file_path, persist_directory)