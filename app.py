from flask import Flask, request, jsonify
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent

app = Flask(__name__)

openai.api_key = "your_openai_api_key"

llm = OpenAI(model="gpt-4", temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 200

documents = SimpleDirectoryReader(input_files=["1.docx", "2.docx", "3.docx"]).load_data()
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(system_prompt="Your system prompt here", similarity_top_k=3)

def get_crypto_price(user_input: str) -> str:
    price = "$1000"
    user_input = f"Latest Price of Cryptocurrency is- {price} " + user_input
    response = chat_engine.chat(user_input)
    return str(response)

price_tool = FunctionTool.from_defaults(fn=get_crypto_price)

def fn_chat_engine(user_input: str) -> str:
    response = chat_engine.chat(user_input)
    return response

chat_tool = FunctionTool.from_defaults(fn=fn_chat_engine)
agent = OpenAIAgent.from_tools([price_tool, chat_tool], llm=llm, verbose=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')
    if user_input:
        response = agent.chat(user_input, tool_choice="auto")
        return jsonify({'response': response.response})
    return jsonify({'error': 'No input provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
