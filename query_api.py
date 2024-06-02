import os
import tiktoken
import openai
import qianfan
import argparse

from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

from ingest import ingest_documents, num_tokens_from_string
from constants import FAISS_PERSIST_DIRECTORY, PROMPT_TEMPLATE, MAX_TOKENS, MIN_BACKUP_TOKENS, CHAT_MODEL_NAME

# 使用 Qianfan 的 embedding
embeddings = QianfanEmbeddingsEndpoint(model='bge-large-zh')

# embeddings = OpenAIEmbeddings(
#     openai_api_base=os.getenv("OPENAI_API_BASE", default=None),
#     openai_api_key=os.getenv("OPENAI_API_KEY", default=None)
# )

if os.path.exists(os.path.join(FAISS_PERSIST_DIRECTORY, "5000")) and len(os.listdir(os.path.join(FAISS_PERSIST_DIRECTORY, "5000"))) > 0:
    vector_store_5000 = FAISS.load_local(os.path.join(FAISS_PERSIST_DIRECTORY, "5000"), embeddings, allow_dangerous_deserialization=True)

app = Flask(__name__)

def chat_with_file(query: str, fileIdentifier: str) -> dict:
    '''
    This function implements the information retreival task.

    1. Loads OpenAIEmbeddings
    2. Loads the existing vectorestore that was created by ingest.py
    3. Loads OpenAI llm
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    '''

    print(f'The file identifier is {fileIdentifier}')

    msg, state = ingest_documents(fileIdentifier, chunk_size=1000, chunk_overlap=200)

    print('Mark')

    if not state:
        return msg

    if fileIdentifier == "5000":
        vector_store = vector_store_5000
        
    else:
        PERSIST_DIRECTORY = os.path.join(FAISS_PERSIST_DIRECTORY, fileIdentifier)

        vector_store = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)

    # 返回与query相似度Top5的文档，并返回相似度分数
    related_docs_with_score = vector_store.similarity_search_with_score_by_vector(embeddings.embed_query(query), k=6)

    # 将相似度分数添加至metadata中
    related_docs = get_docs_with_score(related_docs_with_score)

    # 将相关文档的内容拼接起来，作为context
    related_context = "\n".join([doc.page_content for doc in related_docs])

    # 生成prompt
    prompt = generate_prompt(query, related_context)

    # 生成tokens，判断是否超过gpt-3.5-turbo的最大输入长度
    # encoder = tiktoken.get_encoding("cl100k_base")
    tokens_of_context = num_tokens_from_string(related_context)
    tokens_of_prompt = num_tokens_from_string(prompt)
    print("The tokens of prompt is: ", tokens_of_prompt)

    # 如果超过最大长度，截断context，重新生成prompt
    if tokens_of_prompt > MAX_TOKENS - MIN_BACKUP_TOKENS:
        cut_off_index = tokens_of_context - (tokens_of_prompt - MAX_TOKENS) - MIN_BACKUP_TOKENS
        related_context = related_context[:cut_off_index]
        prompt = generate_prompt(query, related_context)
        print("The tokens of prompt after cutting is: ", num_tokens_from_string(prompt))

    chat_completion = qianfan.ChatCompletion()

    response = chat_completion.do(
        model=CHAT_MODEL_NAME, 
        messages=[{"role": "user", "content": prompt }]
    )

    answer = response["body"]["result"]
    
    # 将相关文档的来源拼接起来
    source = ";".join(set(os.path.basename(doc.metadata["source"]) for doc in related_docs))

    formatted_result = {
        "query": query,
        "answer": answer,
        "source": source
    }

    print(formatted_result)

    return formatted_result

def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs

def generate_prompt(query: str,
                    context: str,
                    prompt_template: str = PROMPT_TEMPLATE) -> str:
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt

@app.route('/ask', methods=['GET','POST'])
def ask():
    #data = request_parse(request)
    data = request.values
    print(data)
    query = data.get('query')
    sub_id = data.get('sub_id')
    print("query : ", query, "sub_id : ", sub_id)

    result = None

    try:
        result = chat_with_file(query, sub_id)

    except (KeyError, AttributeError) as error:
        print(f"Caught {error.__class__.__name__}: {error}")
        return jsonify({'error': 'Data not found when answering. Please check your secret key'})

    except Exception as e:
        print("Caught an unexpected exception:", e)
        return jsonify({'error': str(e)})

    if result is not None:
        return jsonify(result)
    else:
        return jsonify({'error': 'An error occurred but no result was generated. Please contact the customer service.'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Upload API')
    parser.add_argument('--port', type=int, default=19000, help='export port')

    args, run_mindformer_args = parser.parse_known_args()

    app.run(host='0.0.0.0', port=args.port, debug=False)
