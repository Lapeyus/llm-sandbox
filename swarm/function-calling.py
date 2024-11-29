from swarm import Swarm, Agent
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Define agent functions
def get_weather(location, time="now"):
    # Function to get weather information
    return f"The weather in {location} at {time} is sunny."

def send_email(recipient, subject, body):
    # Function to send an email
    return "Email sent successfully."

# Define agents
weather_agent = Agent(
    name="Weather Agent",
    model=OPENAI_MODEL_NAME,
    instructions="Provide weather updates using your tools.",
    # tool_choice=
    functions=[get_weather, send_email],
    parallel_tool_calls=True
)

# Initialise Swarm client and run conversation
client = Swarm()

response = client.run(
    agent=weather_agent,
    messages=[{"role": "user", "content": "What's the weather in New York?"}],
    # model_override='qwen2.5-coder:32b'
)

print(response.messages[-1]["content"])