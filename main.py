from fastapi import FastAPI, File, UploadFile
from utils import rag
import os
db_path = "./.chroma_db"
app = FastAPI()

rag = rag()

@app.get("/")
#async def root():
def root():
    return {"message": "Hello World"}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIR = "./originals/" + file.content_type.split('/')[1] # store original files ex) ./originals/pdf
    content = await file.read()
    with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as fp:
        fp.write(content)  # store the file in the storage
    
    rag.store_data() # automatically vectorize the file into the database
    
    return {
        "content_type": file.content_type,
        "filename": file.filename
    }

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
