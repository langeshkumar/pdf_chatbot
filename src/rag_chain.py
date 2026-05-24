from langchain_classic.chains import RetrievalQA
from langchain_community.llms import Ollama

def create_rag_chain(vectorData):

    llm = Ollama(
        temperature=0,
        model="llama3"
    )

    retriever = vectorData.as_retriever()

    qa_values = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_values