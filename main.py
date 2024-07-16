from fastapi import FastAPI, File, UploadFile
from utils import rag
import os
import uvicorn
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


db_path = "./.chroma_db"
FILE_DIRECTORY = Path("./originals/pdf")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag = rag()

@app.get("/")
#async def root():
def root():
    return {"message": "Hello World"}

@app.get("/files")
async def list_files():
    files = os.listdir(path=FILE_DIRECTORY)
    return JSONResponse(content=files)

@app.post("/upload_file")
async def upload_file(files: list[UploadFile] = File(...)):
    for file in files:
        UPLOAD_DIR = "./originals/" + file.content_type.split('/')[1]
        content = await file.read()
        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as fp:
            fp.write(content)
    
        rag.store_data()  # 자동으로 파일을 벡터화
    
    return {"message": "파일 업로드 완료"}


@app.get("/vectorize_db") # execute vectorize db
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
        return_answer, evidence = rag.answer_with_data(question=question)
        return {"message": "Success", "answer": return_answer, "evidence": evidence}
    except:
        return {"message": "Fail", "answer": ""}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)