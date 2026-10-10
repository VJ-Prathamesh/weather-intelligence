from fastapi import FastAPI
from Agent.orchastrator import orchastrator
from pydantic import BaseModel

app=FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
def chat(request: ChatRequest):
    print(f" error : {request.message} in thread: {request.thread_id}" )
    return orchastrator(message=request.message, 
                        thread_id=request.thread_id)










