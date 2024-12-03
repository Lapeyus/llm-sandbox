import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from swarm import Swarm, Agent

# Load environment variables
load_dotenv()
client = Swarm()

# Get environment variables
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME_LARGE = os.getenv('OPENAI_MODEL_NAME_LARGE')
OPENAI_EMBED_NAME = os.getenv('OPENAI_EMBED_NAME')
OPENAI_BASE_URL_RAG = os.getenv('OPENAI_BASE_URL_RAG')

# Validate environment variables
required_env_vars = [OPENAI_MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EMBED_NAME, OPENAI_BASE_URL_RAG]
if any(env_var is None for env_var in required_env_vars):
    raise ValueError("Some required environment variables are not set.")

# Create an Agent for rephrasing
rephrase_agent = Agent(
    name="rephrase_assistant",
    instructions=(
        "You're a specialist in 'text normalization' with skills in 'text decomposition',"
        "Your task is to apply 'semantic segmentation' and 'atomic statement extraction', "
        "breaking down complex or compound statements into discrete, self-contained units "
        "of meaning to enhance clarity and facilitate downstream processing."
    ),
    model=OPENAI_MODEL_NAME_LARGE
)

PROMPT='''
Take the provided question and transform it into multiple standalone statements. Each statement should be self-contained, meaning it conveys a complete idea on its own and is understandable without requiring additional context. 
For example: 

Input: Take the provided text and transform it into multiple standalone statements. Each statement should be self-contained, meaning it conveys You want to configure autohealing for network load balancing for a group of Compute Engine instances that run in multiple zones, using the fewest possible steps.You need to configure re-creation of VMs if they are unresponsive after 3 attempts of 10 seconds each.What should you do?a complete idea on its own and is understandable without requiring additional context. For example: Input: Virtual Private Cloud (VPC) provides networking functionality to Compute Engine virtual machine (VM) instances, Google Kubernetes Engine (GKE) clusters. It also works with serverless workloads. Output: •	Virtual Private Cloud (VPC) provides networking functionality to Compute Engine virtual machine (VM) instances. •	Virtual Private Cloud (VPC) provides networking functionality to Google Kubernetes Engine (GKE) clusters. •	Virtual Private Cloud (VPC) provides networking functionality to serverless workloads. 

Output:
- You want to configure autohealing for network load balancing for a group of Compute Engine instances
- The group of Compute Engine instances runs in multiple zones
- You want to using the fewest possible steps
- You need to configure re-creation of VMs if they are unresponsive after 3 attempts of 10 seconds each
- What should you do?
'''


# Initialize embedding model
try:
    embed_model = OllamaEmbeddings(
        model=OPENAI_EMBED_NAME,
        base_url=OPENAI_BASE_URL_RAG
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize Ollama embeddings: {e}")

def create_and_persist_vector_store(file_path: str, persist_directory: str, decomposition_output_path: str):
    """
    Reads a file, splits the text into chunks, decomposes the text into standalone statements, 
    embeds it, and persists the vector store. Also saves the decomposition to a text file.
    """
    try:
        # Read the file content
        with open(file_path, "r") as f:
            text = f.read()

        # Initialize the text splitter
        text_splitter = SemanticChunker(
            embed_model, 
            breakpoint_threshold_type="gradient"
        )
        chunks = text_splitter.split_text(text)

        # Log the initial chunks for debugging
        print(f"Initial chunks: {chunks[:5]}...")  # Print first 5 chunks for brevity

        # Aggregate all decomposed statements
        all_decompositions = []

        for chunk in chunks:
            response = client.run(
                agent=rephrase_agent,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Take the provided text and transform each sentence into one or multiple standalone statements. "
                            f"Each statement should be self-contained, meaning it conveys a complete idea on its own and is "
                            f"understandable without requiring additional context or modifying the original idea"
                            f"example: Input: Virtual Private Cloud (VPC) provides networking functionality to Compute Engine virtual machine (VM) instances, Google Kubernetes Engine (GKE) clusters. It also works with serverless workloads. Output: - Virtual Private Cloud (VPC) provides networking functionality to Compute Engine virtual machine (VM) instances. - Virtual Private Cloud (VPC) provides networking functionality to Google Kubernetes Engine (GKE) clusters. - Virtual Private Cloud (VPC) provides networking functionality to serverless workloads."
                            f"Reply with the output only, don't add intros or explanations:\n{chunk}"
                        )
                    }
                ],
            )

            # Parse the rephrased content
            content = response.messages[-1]["content"].strip()
            decomposition = [line.strip() for line in content.split("\n- ") if line.strip()]
            print(f"Decomposition for the current chunk: {decomposition}...")   

            # Append to the aggregated list
            all_decompositions.extend(decomposition)

            # Add metadata to each statement
            metadata_statements = [
                {
                    "context": chunk,
                    "length": len(statement),
                    "source_file": file_path
                }
                for statement in decomposition
            ]

            # Create the vector store and persist it
            vector_store = Chroma.from_texts(
                texts=decomposition,
                embedding=embed_model,
                persist_directory=persist_directory,
                metadatas=metadata_statements
            )
            print(f"--" * 80) 
            print(f"Chunk being processed: {chunk}...")
            print(f" -- " * 80)
            print(f"decomposition being processed: {decomposition}...")
            print(f" --| " * 80)
            print(f"decomposition being processed: {metadata_statements}...")

        # Save all decompositions to a text file
        with open(decomposition_output_path, "w") as output_file:
            for statement in all_decompositions:
                output_file.write(statement + "\n")

        print(f"Decomposed statements saved to: {decomposition_output_path}")
        print(f"Vector store created and persisted at: {persist_directory}")

        return vector_store

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the vector store: {e}")

if __name__ == "__main__":
    file_path = "state_of_the_union.txt"
    persist_directory = "./chroma_db"
    decomposition_output_path = "decomposed_statements.txt"

    # Validate file path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Create and persist vector store
    vector_store = create_and_persist_vector_store(file_path, persist_directory, decomposition_output_path)