
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from datetime import datetime


def get_date():
    """Get the current date and time"""
    return datetime.now().strftime("%d-%m-%Y")

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
You are a helpful assistant.
Use the get_date tool if the user is asking about today's date.
"""

agent = create_agent(model=llm,
                     tools = [get_date],
                     system_prompt = system_prompt
                     )

user_query = input("Enter your query: ")
response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
response = response["messages"][-1].content

print(response)
