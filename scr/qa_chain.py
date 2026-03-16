import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from vector_store import load_vector_store
from embeddings import get_embedding_model
from llm import get_llm

def create_qa_chain(persist_directory="./chroma_db", k=3):
    """
    创建RetrievalQA链
    """
    # 加载向量库
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model, persist_directory)
    
    # 创建检索器
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # 定义提示词模板
    template = """你是一个智能客服助手，请根据以下上下文内容回答问题。如果上下文中没有相关信息，请说“我没有找到相关信息”，不要编造答案。

上下文：
{context}

问题：{question}
回答："""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # 初始化LLM
    llm = get_llm()
    
    # 构建链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )
    return qa_chain

