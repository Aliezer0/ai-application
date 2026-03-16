import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_CACHE'] = './cache'        # 指定缓存目录
os.environ['HUGGINGFACE_HUB_CACHE'] = './cache'
from document_loader import load_pdf
from text_splitter import split_documents, get_splitter
from embeddings import get_embedding_model
from vector_store import create_vector_store

def main():
    # 1. 加载文档
    pdf_path = os.path.join("data", "knowledge_base.pdf")
    documents = load_pdf(pdf_path)
    
    # 2. 分割文档
    splitter = get_splitter(chunk_size=500, chunk_overlap=50)
    chunks = split_documents(documents, splitter)
    
    # 3. 初始化嵌入模型
    embedding_model = get_embedding_model()
    
    # 4. 创建向量库并持久化
    persist_dir = "./chroma_db"
    vectordb = create_vector_store(chunks, embedding_model, persist_dir)
    
    print("知识库构建完成！")

if __name__ == "__main__":
    main()