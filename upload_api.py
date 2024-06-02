import os
from flask import Flask, request, render_template, jsonify
from constants import BASE_SOURCE_DIRECTORY, ALLOWED_EXTENSIONS
import argparse
import tiktoken

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/')
def index():
    msg = "welcome to file upload server."
    return render_template("web.html", data=msg)

@app.route('/tokens', methods=['POST'])
def tokens():
    data = request.values

    context = data.get('context')
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens_of_context = len(encoder.encode(context))
    
    formatted_result = {
        "context": context,
        "num_tokens": tokens_of_context
    }

    print(formatted_result)

    return jsonify(formatted_result)

@app.route('/tokens_plus', methods=['POST'])
def tokens_plus():
    data = request.values

    prompt = data.get('prompt')
    completion = data.get('completion')

    encoder = tiktoken.get_encoding("cl100k_base")
    tokens_of_prompt = len(encoder.encode(prompt))
    tokens_of_completion = len(encoder.encode(completion))

    formatted_result = {
        "prompt_tokens": tokens_of_prompt,
        "completion_tokens": tokens_of_completion,
        "total_tokens": tokens_of_prompt + tokens_of_completion
    }

    print(formatted_result)

    return jsonify(formatted_result)

@app.route('/upload', methods=['POST'])
def upload():
    file_identifier = request.form.get('fileIdentifier')
    if len(file_identifier) == 0:
        return "Please send a POST request with a file identifier", 200

    if 'file' not in request.files:
        return "Please send a POST request with a file", 200
    
    uploaded_file = request.files["file"]
    filename = uploaded_file.filename
    if allowed_file(filename):
        SOURCE_DI = BASE_SOURCE_DIRECTORY + f'/{file_identifier}'
        if not os.path.exists(SOURCE_DI):
            os.makedirs(SOURCE_DI)
        filepath = os.path.join(SOURCE_DI, os.path.basename(filename)) 
        uploaded_file.save(filepath)
        return f"Upload successfuly! \nFile name: {filename} \nFile identifier: {file_identifier}", 200
    else:
        return f"Please send a POST request with a file end up with {ALLOWED_EXTENSIONS}", 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Upload API')
    parser.add_argument('--port', type=int, default=19100, help='export port')

    args, run_mindformer_args = parser.parse_known_args()

    if not os.path.exists(BASE_SOURCE_DIRECTORY):
        os.makedirs(BASE_SOURCE_DIRECTORY)

    app.run(host='0.0.0.0', port=args.port, debug=False)
