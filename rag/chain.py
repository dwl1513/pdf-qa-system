import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """把文档块向量化存入向量库，分批处理避免超出限制"""
    embeddings = OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )

    BATCH_SIZE = 50  # 留一点余量，不要用满64
    
    print(f"共 {len(chunks)} 个块，分批向量化中...")
    
    # 第一批单独建库
    vectorstore = Chroma.from_documents(chunks[:BATCH_SIZE], embeddings)
    
    # 剩余批次逐批加入
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"已处理 {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} 块")
    
    print("向量库构建完成")
    return vectorstore


from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

def build_qa_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(
        model="glm-4-flash",
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0,
    )

    # Memory模块：自动维护对话历史
    memory = ConversationBufferMemory(
        memory_key="chat_history",      # 这个key名固定，Chain内部用这个名字取历史
        return_messages=True,           # 以Message对象格式存储，而不是纯字符串
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,                  # 把memory传进去
        return_source_documents=True,
    )
    return qa_chain