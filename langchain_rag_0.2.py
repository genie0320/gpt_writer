"""# Project setting
pip install --quiet langchain langchain-openai
pip install --quiet pypdf chromadb tiktoken
pip install --quiet icecream
"""
# %%
import os
import icecream as ic
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import chromadb

# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_model = OpenAI(
    api_key=openai_api_key, model="gpt-3.5-turbo", max_tokens=50, temperature=2
)
chat_model = ChatOpenAI(
    api_key=openai_api_key, model="gpt-3.5-turbo", max_tokens=50, temperature=2
)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Chroma - persistent
# client = chromadb.PersistentClient(path="/embed_chroma/")
# client.heartbeat()
# client.reset()  # Empties and completely resets the database.

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
from langchain_community.document_loaders import PyPDFLoader

text_loader = TextLoader("resources/books_content.txt", encoding="UTF-8")
text_pages = text_loader.load()

pdf_loader = PyPDFLoader("resources/plan01.pdf")
pdf_pages = pdf_loader.load_and_split()

# TODO:from langchain_experimental.text_splitter import SemanticChunker > 이건 openai의 실험적 기능이라는데... 한번 써보고 싶음.
## Transform (chunking)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=30,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(text_pages)
print(texts[0])
# %%
# Embedding & Store to vectorDB
from langchain_openai import OpenAIEmbeddings

user_query = "기획이란"

# save to disk
db = Chroma.from_documents(texts, embeddings, persist_directory="./embed_chroma")
docs = db.similarity_search(user_query)

# # load from disk
# db = Chroma(persist_directory="./embed_chroma", embedding_function=embeddings)
# docs = db.similarity_search(user_query)
# print(f"from db load : {docs[0].page_content}")

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
