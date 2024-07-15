from fastapi import FastAPI
from utils import rag

app = FastAPI()

rag = rag()

@app.get("/")
#async def root():
def root():
    return {"message": "Hello World"}


@app.get("/input_data")
def input_data(data: object):
    try:
        rag.store_data(data=data)
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
def rag_answer(question: object):
    try:
        return_answer = rag.answer_with_data(question=question)
        return {"message": "Success", "answer": return_answer}
    except:
        return {"message": "Fail", "answer": ""}
