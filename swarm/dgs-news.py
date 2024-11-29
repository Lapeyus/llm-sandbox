from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv

current_date = datetime.now().strftime("%Y-%m")

load_dotenv()

# Initialize Swarm client
client = Swarm()

# 1. Create Internet Search Tool

def get_news_articles(topic):
    print(f"Running DuckDuckGo news search for {topic}...")
    
    # DuckDuckGo search
    ddg_api = DDGS()
    results = ddg_api.text(f"{topic} {current_date}", max_results=5)
    if results:
        news_results = "\n\n".join([f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}" for result in results])
        return news_results
    else:
        return f"Could not find news results for {topic}."
    
# 2. Create AI Agents

# News Agent to fetch news
news_agent = Agent(
    name="News Assistant",
    instructions="You provide the latest news articles for a given topic using DuckDuckGo search.",
    functions=[get_news_articles],
    model="llama3.2"
)

# Editor Agent to edit news
editor_agent = Agent(
    name="Editor Assistant",
    instructions="Rewrite and give me as news article ready for publishing. Each News story in separate section.",
    model="llama3.2"
)

# 3. Create workflow

def run_news_workflow(topic):
    print("Running news Agent workflow...")
    
    # Step 1: Fetch news
    news_response = client.run(
        agent=news_agent,
        messages=[{"role": "user", "content": f"Get me the news about {topic} on {current_date}"}],
    )
    
    raw_news = news_response.messages[-1]["content"]
    
    # Step 2: Pass news to editor for final review
    edited_news_response = client.run(
        agent=editor_agent,
        messages=[{"role": "user", "content": raw_news }],
    )
    
    return edited_news_response.messages[-1]["content"]

# Example of running the news workflow for a given topic
print(run_news_workflow("openai operator"))
