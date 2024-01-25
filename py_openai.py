import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ___ setting ___
# set openai key in .env file as OPENAI_API_KEY'
MODEL = "gpt-3.5-turbo"
MAX = 50
TEMP = 1.5
# _______________

api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)


# from langchain_community.chat_models import OpenAI


# # ___________ Settings _____________
# prompt = "Hello, I'm your friendly OpenAI chatbot! How can I assist you today?"


# response = openai.complete(prompt)
# print(f"Chatbot: {response.text}")

# while True:
#     user_input = input("You: ")
#     response = openai.complete(f"{prompt}\n\nYou: {user_input}")
#     print(f"Chatbot: {response.text}")
#     if user_input.lower() == "exit":
#         break

# openai.close()
