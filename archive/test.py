import pandas as pd
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import *
from chain.preprocessor import prepare_documents

# LangSmith Tracker
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "local-ollama-analytics-agent"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------- Index --------
# vectorstore = load_or_create_vector_store()
# retriever = vectorstore.as_retriever()
# docs = retriever.invoke('What are the key insights for May 2025')
df_metrics = pd.read_excel(METRICS_FILE, sheet_name="Metrics")
df_insights = pd.read_excel(INSIGHTS_FILE, sheet_name="Insights")
documents = prepare_documents(df_metrics, df_insights)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Index documents in-memory using Chroma (no persistent storage)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever()

# --- Init model ---
llm = ChatOllama(model="llama3", temperature=0, streaming=True)
# vectorstore = load_vector_store()

# --- Step 1: Decomposition ---
# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
decompose_chain = prompt_decomposition | llm

# Chain
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "how does TH visits look like in May 2025?"
questions = generate_queries_decomposition.invoke({"question":question})
print(questions)


# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


q_a_pairs = ""
for q in questions:
    rag_chain = (
            {"context": itemgetter("question") | retriever,
             "question": itemgetter("question"),
             "q_a_pairs": itemgetter("q_a_pairs")}
            | decomposition_prompt
            | llm
            | StrOutputParser())

    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

# Answer each sub-question individually

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAG prompt
prompt_rag = hub.pull("rlm/rag-prompt")


def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """RAG on each sub-question"""

    # Use our decomposition /
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs,
                                                                "question": sub_question})
        rag_results.append(answer)

    return rag_results, sub_questions


# Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition)


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


context = format_qa_pairs(questions, answers)

# Prompt
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
)

final_answer = final_rag_chain.invoke({"context": context, "question": question})
print(final_answer)

