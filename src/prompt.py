from langchain_core.prompts import ChatPromptTemplate


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from the medical documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)

## We can use this if we want to answer even if it is not related to Medical using LLM
# direct_generation_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Answer using only your general knowledge.\n"
#             "If it requires specific medical info, say:\n"
#             "'I don't know based on my general knowledge.'"
#         ),
#         ("human", "{question}"),
#     ]
# )

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical assistant. Only answer questions related to medical topics. For any non-medical questions, strictly respond with: 'I don't know. Only ask me medical related questions please.'"        ),
        ("human", "{question}"),
    ]
)


is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are evaluating whether a retrieved medical document can help answer a user's medical question.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it contains information that can directly or indirectly help answer the question.\n"
            "It may include symptoms, causes, diagnosis, treatment, medications, or related medical explanations.\n\n"
            "Examples:\n"
            "- A document describing symptoms of diabetes is relevant to a question about diabetes symptoms.\n"
            "- A document about treatment of hypertension is relevant to questions about managing high blood pressure.\n"
            "- A document explaining causes of chest pain is relevant to a query about chest pain reasons.\n"
            "- A document about a completely different disease is NOT relevant.\n\n"
            "The document does NOT need to contain the exact final answer,\n"
            "but it must contain useful medical information related to the question.\n\n"
            "If the document is unrelated to the medical topic in the question, return is_relevant=false.\n"
            "If unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)


rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical assistant chatbot.\n\n"
            "You are given CONTEXT extracted from a medical reference book.\n"
            "Task:\n"
            "Answer the user's question strictly based on the provided context.\n\n"
            "Rules:\n"
            "- Use only the information present in the context.\n"
            "- Do NOT add any external knowledge.\n"
            "- If the context partially answers the question, respond with the available information only.\n"
            "- If the answer is not present, say: 'I don't know based on the provided information.'\n"
            "- Do NOT mention the word 'context' in your answer."
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical fact-checking assistant.\n"
            "Your task is to verify whether the ANSWER is supported by the CONTEXT.\n\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "Evaluation criteria:\n"
            "- fully_supported: Every medical claim is directly supported by the CONTEXT.\n"
            "- partially_supported: Some claims are supported, but the answer includes additional information not present in CONTEXT.\n"
            "- no_support: The main claims are not supported or contradict the CONTEXT.\n\n"
            "Examples:\n"
            "Example 1:\n"
            "CONTEXT: 'Symptoms of diabetes include increased thirst and frequent urination.'\n"
            "ANSWER: 'Diabetes symptoms include increased thirst and frequent urination.'\n"
            "OUTPUT: fully_supported\n\n"
            "Example 2:\n"
            "CONTEXT: 'Symptoms of diabetes include increased thirst and frequent urination.'\n"
            "ANSWER: 'Diabetes symptoms include increased thirst, frequent urination, and fatigue.'\n"
            "OUTPUT: partially_supported\n\n"
            "Reason: 'fatigue' is not present in CONTEXT.\n\n"
            "Example 3:\n"
            "CONTEXT: 'Hypertension treatment includes ACE inhibitors.'\n"
            "ANSWER: 'Hypertension is treated using insulin.'\n"
            "OUTPUT: no_support\n"
            "Reason: incorrect and unsupported treatment.\n\n"
            "Strict rules:\n"
            "- Be extremely strict.\n"
            "- Do NOT use outside medical knowledge.\n"
            "- Even small unsupported additions → partially_supported.\n"
            "- Contradictions → no_support.\n\n"
            "Evidence:\n"
            "- Provide up to 3 short exact quotes from CONTEXT.\n"
            "- If no support exists, return an empty list."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}"
        ),
    ]
)


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)
