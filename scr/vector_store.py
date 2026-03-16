from langchain_community.vectorstores import Chroma

def create_vector_store(documents, embedding_model, persist_directory="./chroma_db"):
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def load_vector_store(embedding_model, persist_directory="./chroma_db"):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )