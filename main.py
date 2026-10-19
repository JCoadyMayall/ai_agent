from dotenv import load_dotenv

import uuid

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from datetime import datetime

import gradio as gr

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_tavily import TavilySearch


load_dotenv()

def get_date():
    """Get the current date and time"""
    return datetime.now().strftime("%d-%m-%Y")


search_tool = TavilySearch()

conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
You are a helpful assistant.
Answer all user's queries.
Use the get_date tool ONLY when the user is explicitly asking about today's date.
Use the search_tool for answering questions requiring up-to-date information.
"""

agent = create_agent(
    model=llm,
    tools = [get_date, search_tool],
    system_prompt = system_prompt,
    checkpointer=checkpointer
)


def chat(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    last_response = response["messages"][-1].content
    return last_response

with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

demo.launch()
