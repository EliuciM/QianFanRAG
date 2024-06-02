import os
import shutil
from tqdm import tqdm
import json

import openai
import qianfan
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from constants import DOCUMENT_MAP, BASE_SOURCE_DIRECTORY, FAISS_PERSIST_DIRECTORY, ALLOWED_EXTENSIONS

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class and file_extension == ".txt":
        loader = loader_class(file_path, encoding="utf-8")
    elif loader_class and file_extension == ".csv":
        loader = loader_class(file_path, encoding="utf-8")
    elif loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()

def load_document_batch(filepaths: str):
    print("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)

def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # # Have at least one worker and at most INGEST_THREADS workers
    # n_workers = min(INGEST_THREADS, max(len(paths), 1))
    # chunksize = round(len(paths) / n_workers)
    # docs = []
    # with ProcessPoolExecutor(n_workers) as executor:
    #     futures = []
    #     # split the load operations into chunks
    #     for i in range(0, len(paths), chunksize):
    #         # select a chunk of filenames
    #         filepaths = paths[i:(i + chunksize)]
    #         # submit the task
    #         future = executor.submit(load_document_batch, filepaths)
    #         futures.append(future)
    #     # process all results
    #     for future in as_completed(futures):
    #         # open the file and load the data
    #         contents, _ = future.result()
    #         docs.extend(contents)
    docs = []
    for i in paths:
        docs.append(load_single_document(i))
    return docs

def ingest_documents(fileIdentifier: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    SOURCE_DIRECTORY = os.path.join(BASE_SOURCE_DIRECTORY, fileIdentifier)
    PERSIST_DIRECTORY = os.path.join(FAISS_PERSIST_DIRECTORY, fileIdentifier)
    PROGRESS_PATH = os.path.join(PERSIST_DIRECTORY, "progress.json")

    try:
        # Check if the SOURCE_DIRECTORY exists and is not empty
        if not os.path.exists(SOURCE_DIRECTORY) or not os.listdir(SOURCE_DIRECTORY):
            return f"There are no documents in {fileIdentifier}. Please check your expert file arrangement", False

        # Check if the PERSIST_DIRECTORY exists
        if os.path.exists(PERSIST_DIRECTORY):
            # Check if the directory is empty
            if not os.listdir(PERSIST_DIRECTORY):
                # If empty, remove and recreate the directory
                shutil.rmtree(PERSIST_DIRECTORY)
                os.makedirs(PERSIST_DIRECTORY)
            elif os.path.exists(os.path.join(PERSIST_DIRECTORY, "index.faiss")) and os.path.exists(os.path.join(PERSIST_DIRECTORY, "index.pkl")):
                # If not empty, notify the user that the vector database already exists
                return f"The vector database has been created. Location is: {PERSIST_DIRECTORY}. If you want to create a new one, please delete it entirely", True
        else:
            # If the persistent directory does not exist, create it
            os.makedirs(PERSIST_DIRECTORY)

        # Retrieve the list of document names from the source directory
        documents_name = os.listdir(SOURCE_DIRECTORY)
        print(f"Loaded {len(documents_name)} documents from {SOURCE_DIRECTORY}")

        # Load documents based on the names retrieved
        documents = load_documents(SOURCE_DIRECTORY)[0]

        # Calculate and print number of tokens per document
        num_tokens = num_tokens_from_documents(documents)
        for doc, num_token in zip(documents, num_tokens):
            doc.metadata["num_tokens"] = num_token 

        short_documents = [doc for doc in documents if doc.metadata["num_tokens"] < 512]
        documents = [doc for doc in documents if doc.metadata["num_tokens"] >= 512]

        print(f"Total number of tokens is {sum(num_tokens)}")

        # Initialize text splitter and split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        # Calculate and print number of tokens per document
        num_tokens = num_tokens_from_documents(texts)
        for doc, num_token in zip(texts, num_tokens):
            doc.metadata["num_tokens"] = num_token 

        texts = [doc for doc in texts if doc.metadata["num_tokens"] >= 30]

        texts.extend(short_documents)

        print(f"Split into {len(texts)} chunks of text")

        # Use OpenAI embedding
        # embeddings = OpenAIEmbeddings(
        #     openai_api_base=os.getenv("OPENAI_API_BASE", default=None),
        #     openai_api_key=os.getenv("OPENAI_API_KEY", default=None)
        # )
        embeddings = QianfanEmbeddingsEndpoint(model='bge-large-zh')

        text_embeddings_info = embedding_documents(texts, embeddings, PROGRESS_PATH)
        text_embeddings = [i["embeddings"] for i in text_embeddings_info]
        text_content = [i["content"] for i in text_embeddings_info]
        text_metadata = [i["metadata"] for i in text_embeddings_info]

        text_embedding_pairs = list(zip(text_content, text_embeddings))
        faiss_db = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=text_metadata)
        faiss_db.save_local(PERSIST_DIRECTORY)

        return f"The vector database is created successfully. Location is: {PERSIST_DIRECTORY}", True
    
    except KeyError as key_error:
        print("Caught KeyError:", key_error)
        # Remove the newly created directory if an exception occurred
        if os.path.exists(PERSIST_DIRECTORY) and not os.listdir(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return "Data not found when embedding. Please check your secret key.", False

    except AttributeError as attr_error:
        print("Caught AttributeError:", attr_error)
        # Remove the newly created directory if an exception occurred
        if os.path.exists(PERSIST_DIRECTORY) and not os.listdir(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return "Data not found when embedding. Please check your secret key.", False
    
    except Exception as e:
        print("An exception occurred:", e)
        # Remove the newly created directory if an exception occurred
        if os.path.exists(PERSIST_DIRECTORY) and not os.listdir(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return f"An error occurred while creating the vector database: {str(e)}. Please contact the customer service.", False

import tiktoken
from qianfan.resources.tools import tokenizer
QFTokenizer = tokenizer.Tokenizer()

def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:

    return QFTokenizer.count_tokens(text=string, mode='local')

def num_tokens_from_string_tiktoken(string: str, encoding_name: str="cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    cl100k_base 对应 gpt-4, gpt-3.5-turbo, text-embedding-ada-002的tokenizer, reference https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_documents(documents: List[Document], encoding_name: str="cl100k_base") -> int:
    """
    Returns the number of tokens in a list of documents.
    cl100k_base 对应 gpt-4, gpt-3.5-turbo, text-embedding-ada-002的tokenizer, reference
    """
    num_tokens = []
    for doc in documents:
        doc_tokens = num_tokens_from_string(doc.page_content, encoding_name=encoding_name)
        num_tokens.append(doc_tokens)
    return num_tokens

def embedding_documents(documents: List[Document], embeddings: OpenAIEmbeddings, PROGRESS_PATH: str) -> List[dict]:
    """
    Returns the embedding of a list of documents.
    根据提供的文档生成embeddings，并处理中断续传。当遇到异常时，使用1024维的全零向量作为备份，并在metadata中记录是否成功生成embedding。
    """
    # 尝试加载之前的进度
    doc_embeddings = load_progress(PROGRESS_PATH)
    start_index = len(doc_embeddings)  # 确定开始处理的索引

    for doc in tqdm(documents[start_index:], desc="Embedding documents"):
        doc_info = {
            "content": doc.page_content,
            "metadata": doc.metadata.copy()  # 使用副本以防修改原始metadata
        }

        try:
            # 正常生成embeddings
            doc_info["embeddings"] = embeddings.embed_query(doc.page_content)
            doc_info["metadata"]["embedding_success"] = True  # 标记成功生成embedding
        except Exception as e:
            # 处理异常，使用全零向量
            print(f"An error occurred: {e}. Using a zero vector as a placeholder.")
            doc_info["embeddings"] = [0.0] * 1024  # 生成1024维的全零向量
            doc_info["metadata"]["embedding_success"] = False  # 标记embedding生成失败

        # 计算tokens数量
        doc_info["metadata"]["num_tokens"] = num_tokens_from_string(doc.page_content)

        # 添加到文档embeddings列表中
        doc_embeddings.append(doc_info)
        
        # 定期保存进度
        if len(doc_embeddings) % 100 == 0:
            print(f"Saving progress at index {len(doc_embeddings)}")
            save_progress(doc_embeddings, PROGRESS_PATH)

    # 处理完成后再次保存进度
    save_progress(doc_embeddings, PROGRESS_PATH)
    return doc_embeddings

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_embeddings_openai(query:str) -> List[float]:
    embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")

    return embedding.data[0].embedding

def get_embedding_qianfan(query_list:List[str]) -> List[float]:
    embedding_model = qianfan.Embedding()
    embedding = embedding_model.do(texts=query_list, model="bge-large-zh")

    return embedding["body"]["data"][0]["embedding"]

def save_progress(embeddings_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(embeddings_list, f, ensure_ascii=False, indent=4)

def load_progress(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    ingest_documents('5000_csv', chunk_size=1000, chunk_overlap=200)