from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

from pymongo import MongoClient
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from src.helper import download_huggingface_embeddings, build_state_graph


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

if PINECONE_API_KEY is None or GROQ_API_KEY is None or MONGO_URI is None:
    raise ValueError("PINECONE_API_KEY, GROQ_API_KEY, and MONGO_URI")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["medical_chatbot"]
chats_collection = db.chats

embeddings = download_huggingface_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=500,
)

state_graph = build_state_graph(llm, retriever)


def serialize_chat(chat, include_messages=False):
    result = {
        "chat_id": chat["chat_id"],
        "user_id": chat["user_id"],
        "title": chat.get("title", "New Chat"),
        "created_at": chat["created_at"].isoformat() + "Z",
    }
    if include_messages:
        result["messages"] = chat.get("messages", [])
    return result


def build_title_from_message(message: str) -> str:
    title = message.strip().replace("\n", " ")
    if len(title) > 50:
        title = title[:47].rsplit(" ", 1)[0] + "..."
    return title or "New Chat"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chats", methods=["GET"])
def list_chats():
    user_id = request.args.get("user_id", "")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    chats = list(
        chats_collection.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
    )
    return jsonify([serialize_chat(chat) for chat in chats])


@app.route("/api/chat/new", methods=["POST"])
def create_chat():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    chat_id = uuid.uuid4().hex
    chat = {
        "chat_id": chat_id,
        "user_id": user_id,
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.utcnow(),
    }
    chats_collection.insert_one(chat)
    return jsonify(serialize_chat(chat)), 201


@app.route("/api/chat/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    user_id = request.args.get("user_id", "")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    chat = chats_collection.find_one({"chat_id": chat_id, "user_id": user_id})
    if chat is None:
        return jsonify({"error": "chat not found"}), 404

    return jsonify(serialize_chat(chat, include_messages=True))


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    user_id = request.form.get("user_id")
    chat_id = request.form.get("chat_id")

    if not user_id or not chat_id or not msg:
        return jsonify({"error": "user_id, chat_id, and msg are required"}), 400

    chat = chats_collection.find_one({"chat_id": chat_id, "user_id": user_id})
    if chat is None:
        return jsonify({"error": "chat not found"}), 404

    first_message = len(chat.get("messages", [])) == 0
    user_message = {"role": "user", "content": msg}
    messages = chat.get("messages", []) + [user_message]

    title = chat.get("title", "New Chat")
    if first_message:
        title = build_title_from_message(msg)

    result = state_graph.invoke(
        {
            "question": msg,
            "need_retrieval": False,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
            "retries": 0,
        },
        config={"recursion_limit": 10},
    )

    assistant_text = str(result.get("answer", ""))
    assistant_message = {"role": "assistant", "content": assistant_text}
    messages.append(assistant_message)

    update_fields = {"messages": messages}
    if title != chat.get("title"):
        update_fields["title"] = title

    chats_collection.update_one(
        {"chat_id": chat_id, "user_id": user_id},
        {"$set": update_fields},
    )

    return assistant_text


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
