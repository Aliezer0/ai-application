from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

def split_documents(documents, splitter=None):
    if splitter is None:
        splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    print(f"分割后得到 {len(chunks)} 个文档块")
    return chunks


