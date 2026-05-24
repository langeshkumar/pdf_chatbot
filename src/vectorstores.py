from langchain_community.vectorstores import FAISS

def vector_store(document, embedding):

    # store vector db
    vectordata = FAISS.from_texts(document, embedding)

    # vector data save local 
    # vectordata.save_local("vector_db/faiss_index")

    return vectordata