from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(model="llama3")

# Define a chat-style prompt
decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that breaks down complex analytical questions into simpler sub-questions."),
    ("human", "Decompose the following question into 3 bullet-point sub-questions that can each be answered independently:\n\n{question}")
])

# Create the chain
decompose_chain = decompose_prompt | llm

# user_question = "What are key insights for May 2025"
# response = decompose_chain.invoke({"question": user_question})
# print(response.content)




