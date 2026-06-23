from llama_index.core import PromptTemplate

QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are a helpful assistant answering questions about uploaded documents.\n"
    "Rules:\n"
    "1. Answer ONLY using the context above. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, reply: I don't know.\n"
    "3. Cite sources inline using [file_name, chunk_id] when possible.\n"
    "Query: {query_str}\n"
    "Answer: "
)

REFINE_PROMPT = PromptTemplate(
    "The original query is: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Given the new context, refine the original answer to better answer the query. "
    "If the new context is not useful, return the original answer. "
    "Cite sources using [file_name, chunk_id].\n"
    "Refined Answer: "
)
