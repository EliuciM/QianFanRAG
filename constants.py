ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'xlsx', 'xls'}

from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)

DOCUMENT_MAP = {
    ".csv": CSVLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader
}

MODEL_TOKEN_MAP = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16000,
    "ernie-4.0-8k-preview": 8000,
    "ernie-3.5-128k": 128000,
    "ernie-3.5-8k": 8000
}

CHAT_MODEL_NAME = "ernie-4.0-8k-preview"

MAX_TOKENS = MODEL_TOKEN_MAP.get(CHAT_MODEL_NAME)

MIN_BACKUP_TOKENS = 250

import os

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

BASE_SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY,"upload")

FAISS_PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY,"faissdb")

if not os.path.exists(BASE_SOURCE_DIRECTORY):
    os.makedirs(BASE_SOURCE_DIRECTORY)

if not os.path.exists(FAISS_PERSIST_DIRECTORY):
    os.makedirs(FAISS_PERSIST_DIRECTORY)

# xiaoyi 的 key 
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""

# qianfan 的 key
os.environ["QIANFAN_ACCESS_KEY"] = ""
os.environ["QIANFAN_SECRET_KEY"] = ""

import openai
openai.api_key = os.getenv("OPENAI_API_KEY", default = None)
openai.api_base = os.getenv("OPENAI_API_BASE", default = None)

PROMPT_TEMPLATE= """
你是为我总结和整理文档的私人助手，如果Question十分日常，比如：你好！你可以做什么？请直接回答。
如果Question是针对Context的，请谨慎评估两者的相关性，仅从Context中总结答案，不要使用其他的外部知识：
Question: {question}
Context: {context}
Answer:"""
