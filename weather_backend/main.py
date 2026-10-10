from fastapi import FastAPI
from Agent.orchastrator import orchastrator
from pydantic import BaseModel

app=FastAPI()



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # the address of your React app; add the real website address later
    allow_methods=["*"],
    allow_headers=["*"],
)


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










