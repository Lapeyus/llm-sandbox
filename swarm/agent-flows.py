from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize Swarm client
client = Swarm()

# Updated Modes configuration with two agents
modes = {
    "questions": {
        "prompt": "Develop a set of thought-provoking questions based on the text. Your response should include: 1. Comprehension Questions: Ensure understanding of the key ideas. 2. Analytical Questions: Encourage deeper thinking about the content. 3. Application Questions: Explore how the ideas could be applied in real-world scenarios. Provide your questions in markdown format:\n",
        "file_extension": "questions"
    },
    "answers": {
        "prompt": "Answer the following questions based strictly on the provided text. Your answers should be clear, concise, and relevant to the content. Include explanations where needed. Provide your responses in markdown format:\n",
        "file_extension": "answers"
    }
}

# Generic function to create agents dynamically
def create_agent(name, instructions):
    return Agent(
        name=name,
        instructions=instructions,
        model="llama3.2"
    )

# Process text using a specific mode (agent)
def process_with_agent(mode, text):
    mode_config = modes[mode]
    agent = create_agent(name=f"{mode.upper()} Agent", instructions=mode_config["prompt"])
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": text}],
    )
    return response.messages[-1]["content"]

# Workflow: generate questions and then answer them
def question_answer_workflow(text):
    # Step 1: Generate questions using the first agent
    print("Generating questions...")
    questions = process_with_agent("questions", text)
    print("\n--- QUESTIONS GENERATED ---\n")
    print(questions)
    
    # Step 2: Answer the questions using the second agent
    print("Answering questions...")
    answers = process_with_agent("answers", f"Text: {text}\n\nQuestions:\n{questions}")
    print("\n--- ANSWERS GENERATED ---\n")
    print(answers)
    
    return {"questions": questions, "answers": answers}

# Example input text
if __name__ == "__main__":
    input_text = (
"""
The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks.

Supported Languages: For text only tasks, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Note for image+text applications, English is the only language supported.
"""
    )
    
    # Run the workflow
    processed_results = question_answer_workflow(input_text)
    
    # Output results
    print("\n--- FINAL OUTPUT ---\n")
    print("Questions:\n", processed_results["questions"])
    print("\nAnswers:\n", processed_results["answers"])