# -*- coding: utf-8 -*-
"""main_langchain_RAG.ipynb의 사본
# Project setting
pip install --quiet langchain langchain-openai
pip install --quiet pypdf chromadb tiktoken
pip install --quiet icecream
"""
# %%
import icecream as ic

# Load Keys
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# ___ setting ___
LLM_MODEL = "gpt-3.5-turbo"
MAX = 50
TEMP = 1.5

EMBED_MODEL = "text-embedding-ada-002"
SPLIT = 500
OVERRAP = 50
# _______________

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

llm_model = OpenAI(
    api_key=openai_api_key,
    # model_kwargs= {'model':LLM_MODEL},
    model=LLM_MODEL,
    max_tokens=MAX,
    temperature=TEMP,
)
chat_model = ChatOpenAI(
    api_key=openai_api_key, model=LLM_MODEL, max_tokens=MAX, temperature=TEMP
)
embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    embeddings_model=EMBED_MODEL,
)

# Chroma - persistent
client = chromadb.PersistentClient(path="/embed_chroma")

print("* Got the key" if openai_api_key else "Something goes wrong")
print(
    f"* LLM model set : - model : {LLM_MODEL} - max token : {MAX} - temperature : {TEMP}"
)
print(
    f"* Embed model set : - model : {EMBED_MODEL} - max token : {MAX} - temperature : {TEMP}"
)
# %%
"""# LLM setting"""

# # Completion
# prompt = PromptTemplate.from_template( '{time} + \n\n text{name} +  {action}')
# user_input = 'Where shall we go today?'
# new_prompt = prompt.format(
#     time = "old",
#     name = "Genie",
#     action = user_input
# )
# res_llm = llm.invoke(new_prompt)

# # Chat
# template = "You are my new friend. We met {place} for {activity}."
# human_template = "{text}"
# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", template), # 값을 tuple로 전달.
#     ("human", human_template),
# ])

# prompt = chat_prompt.format_messages(place="in library", activity="being a study friend", text="Hey, Sweety!")
# res_chat = chat_model.invoke(prompt).content

# print(prompt)
# print(res_chat)
# %%
"""# RAG"""

## loader
# TODO: 특정 폴더 안의 모든 파일을 불러올 수 있도록 구성.
# from langchain_community.document_loaders import DirectoryLoader > 이건 다 불러올 수 있는건가..?

# with open("../../state_of_the_union.txt") as f: # 이건 로컬파일일 때...이야기인가. loader와 무슨 관계?
#     state_of_the_union = f.read()

# Text loader
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader("resources/books_content.txt", encoding="UTF-8")
text_pages = text_loader.load()

# from langchain_community.document_loaders import PyPDFLoader
# pdf_loader = PyPDFLoader("resources/books_content.txt")
# pdf_pages = loader.load_and_split()
# pages[0]

## Transform (chunking)
# TODO:from langchain_experimental.text_splitter import SemanticChunker > 이건 openai의 실험적 기능이라는데... 한번 써보고 싶음.
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=30,
    # length_function=len, > tiktoken 사용시에는 왠지... 비활성해야 함.
    is_separator_regex=False,
)

# texts = text_splitter.create_documents([state_of_the_union])
texts = text_splitter.split_documents(text_pages)
# print(texts[0])
print(texts[0])
print(texts[1])
# %%
# Embedding & Store to vectorDB
from langchain_openai import OpenAIEmbeddings

# chroma_client = chromadb.Client() > 이건 뭐에 필요한 것인지...
embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)

db = Chroma.from_documents(texts, embeddings_model)

# user_input_embedding = input("검색할 키워드 입력 :")
user_input_embedding = "기획이란"
query = user_input_embedding
docs = db.similarity_search(query)
print(docs[0].page_content)

## Store
# Done with embedding
# %%
## Retrieve from DB
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

## Make answer with LLM
from langchain.chains import RetrievalQA

# user_input_multiquery = input("검색할 키워드 입력 :")
user_input_multiquery = "입력된 자료를 종합해서 새로운 책의 목차를 생성해줘."
question = user_input_multiquery
llm = chat_model
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
result = qa_chain.invoke({"query": question})
# print(result)
print(result["result"])

# %%
