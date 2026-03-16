from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """加载PDF文档，返回Document列表"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"加载了 {len(documents)} 页")
    return documents

