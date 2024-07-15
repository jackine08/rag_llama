from fastapi import FastAPI, File, UploadFile
from utils import rag
from langchain_community.document_loaders import PyPDFLoader
import uvicorn
import os

app = FastAPI()

rag = rag()

@app.get("/")
#async def root():
def root():
    return {"message": "Hello World"}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIR = "./.chroma_db/originals/" + file.content_type.split('/')[1]
    content = await file.read()
    with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
    
    return {
        "content_type": file.content_type,
        "filename": file.filename
    }

@app.get("/vectorize_db")
def vectorize_db():
    try:
        rag.store_data()
    except:
        print("error occur")
    
    return {"message": "Success"}


@app.get("/llama_answer")
def llama_answer(question: str):
    try:
        return_answer = rag.answer_without_data(question=question)
        return {"message": "Success", "answer": return_answer}
    except:
        return {"message": "Fail", "answer": ""}



@app.get("/rag_answer")
def rag_answer(question: str):
    try:
        return_answer = rag.answer_with_data(question=question)
        return {"message": "Success", "answer": return_answer}
    except:
        return {"message": "Fail", "answer": ""}
