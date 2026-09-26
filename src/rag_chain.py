import os
from dotenv import load_dotenv
from langchain_community.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def create_rag_chain(vectorData):
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        temperature=0,
        # Updated to an active model ID
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
    )

    retriever = vectorData.as_retriever()

    qa_values = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

    return qa_values