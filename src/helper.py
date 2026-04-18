import json
from typing import List, TypedDict, Literal

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.prompt import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
)


def parse_json_response(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise


class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int


# Extract data from PDF files

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Split the data into text chunks

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# 384 dimensional vector embeddings

def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def build_state_graph(llm, retriever):
    def decide_retrieval(state: State):
        out = llm.invoke(decide_retrieval_prompt.format_messages(question=state["question"]))
        decision = parse_json_response(out.content)
        return {"need_retrieval": decision["should_retrieve"]}

    def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
        return "retrieve" if state["need_retrieval"] else "generate_direct"

    def generate_direct(state: State):
        out = llm.invoke(direct_generation_prompt.format_messages(question=state["question"]))
        return {"answer": out.content}

    def retrieve(state: State):
        return {"docs": retriever.invoke(state["question"])}

    def is_relevant(state: State):
        relevant_docs: List[Document] = []
        for doc in state.get("docs", []):
            out = llm.invoke(
                is_relevant_prompt.format_messages(
                    question=state["question"],
                    document=doc.page_content,
                )
            )
            decision = parse_json_response(out.content)
            if decision.get("is_relevant"):
                relevant_docs.append(doc)
        return {"relevant_docs": relevant_docs}

    def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
        if state.get("relevant_docs"):
            return "generate_from_context"
        return "no_answer_found"

    def generate_from_context(state: State):
        context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
        if not context:
            return {"answer": "No answer found.", "context": ""}
        out = llm.invoke(
            rag_generation_prompt.format_messages(question=state["question"], context=context)
        )
        return {"answer": out.content, "context": context}

    def no_answer_found(state: State):
        return {"answer": "No answer found.", "context": ""}

    def is_sup(state: State):
        out = llm.invoke(
            issup_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        decision = parse_json_response(out.content)
        return {
            "issup": decision["issup"],
            "evidence": decision.get("evidence", []),
        }

    MAX_RETRIES = 4

    def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
        if state.get("issup") == "fully_supported":
            return "accept_answer"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "accept_answer"
        return "revise_answer"

    def accept_answer(state: State):
        return {}

    def revise_answer(state: State):
        out = llm.invoke(
            revise_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        return {"answer": out.content, "retries": state.get("retries", 0) + 1}

    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", is_sup)
    g.add_node("accept_answer", accept_answer)
    g.add_node("revise_answer", revise_answer)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("no_answer_found", END)

    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {"accept_answer": "accept_answer", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "is_sup")
    g.add_edge("accept_answer", END)

    return g.compile()
