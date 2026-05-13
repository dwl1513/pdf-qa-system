import os
import jieba

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


def _chinese_tokenize(text: str) -> list[str]:
    """jieba 分词，给 BM25 用——BM25 默认按空格分词，对中文几乎失效。"""
    return list(jieba.cut(text))


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """把文档块向量化存入向量库，分批处理避免超出限制"""
    embeddings = OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )

    BATCH_SIZE = 50

    print(f"共 {len(chunks)} 个块，分批向量化中...")

    vectorstore = Chroma.from_documents(chunks[:BATCH_SIZE], embeddings)

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"已处理 {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} 块")

    print("向量库构建完成")
    return vectorstore


def build_hybrid_retriever(
    chunks: list[Document],
    vectorstore: Chroma,
    recall_k: int = 20,
    rerank_top_n: int = 5,
):
    """三段式检索：向量召回 + BM25 召回 → RRF 融合 → BGE cross-encoder 精排。

    流程：
      ① 向量检索 top-k（语义相似）   ┐
                                    ├→ EnsembleRetriever 用 RRF 融合排名
      ② BM25 检索 top-k（关键词命中）┘            ↓
                                          ③ BGE-reranker 用 cross-encoder
                                             把 top-k 精排成 top-n
    """
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": recall_k})

    bm25 = BM25Retriever.from_documents(
        chunks,
        preprocess_func=_chinese_tokenize,
    )
    bm25.k = recall_k

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.5, 0.5],
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=rerank_top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )
    return compression_retriever


def build_qa_chain(chunks: list[Document], vectorstore: Chroma) -> ConversationalRetrievalChain:
    """带 hybrid + rerank 检索链路的对话式 QA。

    注意签名变化：现在需要传 chunks（BM25 要全量文档建索引）。
    """
    llm = ChatOpenAI(
        model="glm-4-flash",
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0,
    )

    retriever = build_hybrid_retriever(chunks, vectorstore, recall_k=20, rerank_top_n=5)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return qa_chain
