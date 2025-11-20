import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import QUIZ_SECRET
from .quiz_solver import solve_quiz_chain

app = FastAPI()

# Optional CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/quiz")
async def quiz_endpoint(request: Request):
    """
    Main webhook endpoint.

    Expected JSON:
    {
      "email": "...",
      "secret": "...",
      "url": "https://...."
      // ... other fields (ignored)
    }
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")

    if not isinstance(secret, str) or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing or invalid fields")

    if secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # At this point, the payload is accepted as valid.
    # We must respond 200, and *also* start solving the quiz.
    # Use asyncio.create_task so we don't block the HTTP response.
    asyncio.create_task(solve_quiz_chain(url))

    return {
        "status": "accepted",
        "message": "Quiz solving started in background.",
        "email": email,
        "url": url,
    }
