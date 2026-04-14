


from langchain.prompts import ChatPromptTemplate


system_prompt = (
    "You are a helpful assistant for answering medical questions."
    " Use the following retrieved documents as context to answer the question."
    " If you don't know the answer, say you don't know."
    "Give a concise and accurate answer based on the provided documents only."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)