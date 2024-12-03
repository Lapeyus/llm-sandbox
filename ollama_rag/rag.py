import os
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME_LARGE = os.getenv('OPENAI_MODEL_NAME_LARGE')
OPENAI_EMBED_NAME = os.getenv('OPENAI_EMBED_NAME')
OPENAI_BASE_URL_RAG = os.getenv('OPENAI_BASE_URL_RAG')

# Initialize embedding model
embed_model = OllamaEmbeddings(
        model=OPENAI_EMBED_NAME,
        base_url=OPENAI_BASE_URL_RAG
)

# Initialize language model
llm = OllamaLLM(
    verbose=True,
    model=OPENAI_MODEL_NAME,
    base_url=OPENAI_BASE_URL_RAG
)

def load_persisted_vector_store(persist_directory: str):
    """
    Loads a persisted vector store from the given directory.
    """
    vector_store = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embed_model
    )
    
    return vector_store.as_retriever(
        # search_type="similarity_score_threshold",
        # search_kwargs={
        #     "k": 3, 
        #    "score_threshold": 0.5
        # },
    )

def create_retrieval_qa_chain(persist_directory: str):
    """
    Creates a retrieval QA chain from a persisted vector store.
    """
    retriever = load_persisted_vector_store(persist_directory)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt,
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

if __name__ == "__main__":
    persist_directory = "./chroma_db"

    try:
        # Create the retrieval chain
        retrieval_chain = create_retrieval_qa_chain(persist_directory)

        # Invoke the chain
        response = retrieval_chain.invoke(
            {"input": "what Dataprep is used for"},
            filter={"source_file": "state_of_the_union.txt"} 
        )
        print(response['answer'])
        print(response['context'])

    except Exception as e:
        print(f"An error occurred: {e}")
        if "404" in str(e):
            print("The API endpoint was not found. Check the base URL and ensure the server is running.")
        raise