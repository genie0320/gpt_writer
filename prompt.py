# TODO: ChatPromptTemplate 와 ChatMessagePromptTemplate 의 차이점.
# TODO: SystemMessagePromptTemplate ~ 등은 대체 어디에 왜 쓰는 물건인가?
# call()과 함께 논의되는 것을 봤을 때... 아무래도 deprecated 된 기능이 아닐까?
# TODO: schema는 당췌 언제, 어떤 이유로 쓰는 물건인가?
# from langchain.schema import HumanMessage # for chat
# from langchain.schema import AIMessage # for chat
# from langchain.schema import SystemMessage # for chat
# TODO: what is templates for?
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
)

# Prompt _______________________________________
"""
Prompt는 크게 단순텍스트(llm)와 대화형식의 텍스트(chat)로 나뉘지만 만드는 방법은 같다.
1. 템플릿만들기 : PromptTemplate.from_template('str {variable}')
예) input_variables=['activity', 'name_of_person'] template='My {name_of_person} {activity}'
2. 조립하기 : prompt.format >> plain text 상태가 된다.
예) hi! Nice to see you.what are you doing?
3. invoke()로 보내기
"""


import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ___ setting ___
MODEL = "gpt-3.5-turbo"
MAX = 50
TEMP = 1.5
# _______________

# LLM Model _______________________________________
# from langchain.prompts import PromptTemplate  # for LLM, older way
# from langchain.prompts.chat import ChatPromptTemplate  # for chat # older way
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # Use this.

from langchain_openai import OpenAI  # for LLM
from langchain_openai import ChatOpenAI

llm = OpenAI(api_key=openai_api_key, temperature=TEMP, max_tokens=MAX)

# LLM_Case 01 : with just a simple text
# text = "Hi?"
# res_llm = llm.invoke(text)
# print(res_llm)

# LLM_Case 02 : with variables
# prompt = PromptTemplate.from_template(  # template를 쓰는 것이 포인트지만... 안써도 되던데?
#     "My {name_of_person} {activity}"
#     + ". It makes me {feeling}."
#     + "\n\n translate this sentence in {language}"
# )
# user_input = input("language? : ")
# message = prompt.format(
#     name_of_person="husband",
#     activity="reading a book",
#     feeling="happy",
#     language=user_input,
# )
# print(llm.invoke(message))


# Chat_Case 01 : Simple way
# chat_model = ChatOpenAI(api_key=openai_api_key, temperature=TEMP, max_tokens=MAX)

# system = "You are a killing developer who work for google core department for future vision. And you love Japanese anime so much. Answer in {number} sentence and no more."
# human = "Could you help me to develop a mini project with {language}?"

# user_input01 = input("Question? : ")
# user_input02 = input("How many? : ")

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         ("human", human),
#         ("ai", "Sure, why not?"),
#         ("human", user_input01),
#     ]
# )

# message = chat_prompt.format_messages(language="python", number=user_input02)

# chat_model_res = chat_model.invoke(message)  # chat model로 부름
# print(chat_model_res.content)  # chat_model로 돌아오는 대답은 .content로 꺼내줘야함.

# # Chat_Case 02 : Complicated way. But bard says which can use for finer setting.
# template = (
#     "You are a helpful assistant that translates {input_language} to {output_language}."
# )
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_message_prompt, human_message_prompt]
# )

# get a chat completion from the formatted messages
# res = chat_model.invoke(
#     chat_prompt.format_prompt(
#         input_language="English", output_language="French", text="I love programming."
#     ).to_messages()
# )
# print(res)

# When used in chain.
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat_model = ChatOpenAI(api_key=openai_api_key, temperature=TEMP, max_tokens=MAX)

# system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
# human_template = "{text}"

# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_message_prompt, human_message_prompt]
# )

system_prompt = SystemMessage(
    content="You are a helpful assistant that translates {input_language} to {output_language}."
)
# human_prompt = HumanMessage(content="hi, {calling}")
# ai_prompt = AIMessage(content="What? What did you say?")
prompt = (
    system_prompt
    + HumanMessage(content="hi, buddy")
    + AIMessage(content="What? What did you say?")
    + "{input}"
)

# prompt.format_messages(
#     input_language="English", output_language="Korean", calling="buddy"
# )
chain = LLMChain(llm=chat_model, prompt=prompt)
res = chain.invoke("what language can you handle?")  # _.run() is deprecated
# print(prompt)  # _.run() is deprecated
print(res["text"])
