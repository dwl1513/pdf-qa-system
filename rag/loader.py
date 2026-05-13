from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？",
    "；", "，",
    " ", "",
]


def load_and_split(pdf_path: str) -> list[Document]:
    """
    加载PDF并切块，返回Document列表
    这是RAG建库的第一步
    """
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    print(f"PDF加载完成，共 {len(raw_documents)} 页")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )
    chunks = splitter.split_documents(raw_documents)
    print(f"切块完成，共 {len(chunks)} 块")

    return chunks
