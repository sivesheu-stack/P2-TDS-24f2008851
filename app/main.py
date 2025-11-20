from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from .config import QUIZ_SECRET
from .quiz_solver import solve_quiz_chain

app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/quiz")
async def quiz_endpoint(request_data: QuizRequest):

    if request_data.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Start background task
    asyncio.create_task(solve_quiz_chain(request_data.url))

    return {
        "status": "accepted",
        "message": "Quiz solving started in background.",
        "email": request_data.email,
        "url": request_data.url
    }
