# Set damn keys
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAI  # for LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# init tuned llm
MODEL = "gpt-3.5-turbo"
MAX = 50
TEMP = 1.5

llm = OpenAI(api_key=openai_api_key, temperature=TEMP, max_tokens=MAX)
chat_model = ChatOpenAI(api_key=openai_api_key, temperature=TEMP, max_tokens=MAX)

# Prompt - chat
from langchain_core.prompts import ChatPromptTemplate

system_template = "You are a {time} friend of mine. and live in {place} and now, We are a {relationship}. My friend would respond to me in one sentence, no more."
human_template = "\n\nI want to go to {user_input}"

chat_template = ChatPromptTemplate.from_messages(  # v.[ from_template ] for llm.
    [
        ("system", system_template),
        ("human", "Where shall we go, today?"),  # 이건 아마도 샘플일 것 같은데...
        ("ai", "As you go, my darling"),
        ("human", human_template),
    ]
)

# user_input = input('where shall we go? : ')
user_input = "library"

prompt = chat_template.format_messages(  # v. [ prompt.format ] for llm
    time="old", place="New york", relationship="good mood", user_input=user_input
)

print(f"me : {prompt}")
print(f"Chat answer : {chat_model.invoke(prompt).content}")
