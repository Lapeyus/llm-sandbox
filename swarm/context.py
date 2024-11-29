from swarm import Swarm, Agent
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME_LARGE = os.getenv('OPENAI_MODEL_NAME_LARGE')

client = Swarm()

def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."

def print_account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"

agent = Agent(
    name="Agent",
    instructions=instructions,
    functions=[print_account_details],
    model=OPENAI_MODEL_NAME_LARGE
)

context_variables = {"name": "James", "user_id": 123}
# context_variables = {
# "customer_context": """Here is what you know about the customer's details:
# 1. CUSTOMER_ID: customer_12345
# 2. NAME: John Doe
# 3. PHONE_NUMBER: (123) 456-7890
# 4. EMAIL: johndoe@example.com
# 5. STATUS: Premium
# 6. ACCOUNT_STATUS: Active
# 7. BALANCE: $0.00
# 8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
# """,
#     "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
# The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024.""",
# }
# response = client.run(
#     messages=[{"role": "user", "content": "Hi!"}],
#     agent=agent,
#     context_variables=context_variables,
# )
# print(response.messages[-1]["content"])

# response = client.run(
#     messages=[{"role": "user", "content": "Print my account details!"}],
#     agent=agent,
#     context_variables=context_variables,
# )
# print(response.messages[-1]["content"])

 

def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")


messages = []
agent = agent
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(
        agent=agent, 
        messages=messages,
        context_variables=context_variables,
    )
    messages = response.messages
    agent = response.agent
    pretty_print_messages(messages)