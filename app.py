from flask import Flask, render_template, jsonify, request

from src.helper import download_huggingface_embeddings
from src.prompt import *

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os


app = Flask(__name__)

load_dotenv()



PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "medicalbot"

# Load existing index
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",

    temperature=0.3,
    max_tokens=500
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    user_input = msg
    print(user_input)
    response = rag_chain.invoke({"input": user_input})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)