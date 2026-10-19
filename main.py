import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """Get the current date and time"""
    return datetime.now().strftime("%d-%m-%Y")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

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
response = response["messages"][-1].content[0]["text"]

print(response)
