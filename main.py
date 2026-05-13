import os
import shutil
from pathlib import Path
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from rag.loader import load_and_split
from rag.chain import build_vectorstore, build_qa_chain
import json

load_dotenv()

# ── FastAPI 应用 ──────────────────────────────
app = FastAPI(title="PDF智能问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

current_qa_chain = None


# ── 接口1：上传PDF ────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    # 保存文件
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 处理PDF
    global current_qa_chain
    try:
        chunks = load_and_split(str(save_path))
        vectorstore = build_vectorstore(chunks)
        current_qa_chain = build_qa_chain(chunks, vectorstore)
        return {"status": "success", "chunks": len(chunks), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 接口2：流式问答 ───────────────────────────
@app.post("/chat")
async def chat(payload: dict):
    if current_qa_chain is None:
        raise HTTPException(status_code=400, detail="请先上传PDF")

    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def generate():
        try:
            source_docs = []
            retriever_done = False

            async for event in current_qa_chain.astream_events(
                {"question": question}, version="v2"
            ):
                kind = event["event"]

                if kind == "on_retriever_end":
                    source_docs = event["data"].get("output", []) or []
                    retriever_done = True

                elif kind == "on_chat_model_stream" and retriever_done:
                    chunk = event["data"]["chunk"]
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        data = json.dumps({"delta": text}, ensure_ascii=False)
                        yield f"data: {data}\n\n"

            if source_docs:
                page = source_docs[0].metadata.get("page", 0) + 1
                tail = json.dumps(
                    {"delta": f"\n\n📄 来源：第 {page} 页"},
                    ensure_ascii=False,
                )
                yield f"data: {tail}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ── Gradio界面 ────────────────────────────────
def process_pdf(pdf_file) -> str:
    global current_qa_chain
    if pdf_file is None:
        return "请先上传PDF文件"
    try:
        chunks = load_and_split(pdf_file.name)
        vectorstore = build_vectorstore(chunks)
        current_qa_chain = build_qa_chain(chunks, vectorstore)
        return f"✅ PDF处理完成，共切分为 {len(chunks)} 个块，可以开始提问了"
    except Exception as e:
        return f"❌ 处理失败：{str(e)}"


async def answer_question(question: str, _history: list):
    """Gradio ChatInterface 用的流式回调：async generator，每次 yield 累加后的答案。"""
    if current_qa_chain is None:
        yield "请先上传PDF文件"
        return
    if not question.strip():
        yield "请输入问题"
        return

    try:
        accumulated = ""
        source_docs = []
        retriever_done = False

        async for event in current_qa_chain.astream_events(
            {"question": question}, version="v2"
        ):
            kind = event["event"]

            if kind == "on_retriever_end":
                source_docs = event["data"].get("output", []) or []
                retriever_done = True

            elif kind == "on_chat_model_stream" and retriever_done:
                chunk = event["data"]["chunk"]
                text = getattr(chunk, "content", "") or ""
                if text:
                    accumulated += text
                    yield accumulated

        if source_docs:
            page = source_docs[0].metadata.get("page", 0) + 1
            accumulated += f"\n\n📄 来源：第 {page} 页"
            yield accumulated

    except Exception as e:
        yield f"回答出错：{str(e)}"


with gr.Blocks(title="PDF智能问答") as gradio_app:
    gr.Markdown("# 📄 PDF智能问答系统")
    gr.Markdown("上传PDF文件，然后针对内容提问")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传PDF", file_types=[".pdf"])
            upload_btn = gr.Button("处理PDF", variant="primary")
            upload_status = gr.Textbox(label="处理状态", interactive=False)
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=answer_question,
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(placeholder="请输入你的问题..."),
                submit_btn="发送",
            )

    upload_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[upload_status])


# ── 把Gradio挂载到FastAPI上 ───────────────────
# 这样一个服务同时提供API接口和Gradio界面
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)