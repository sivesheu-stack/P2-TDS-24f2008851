# main.py
import sys
import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# Import the single-URL solver (must be async and return a dict)
# Ensure quiz_solver.solve_single_quiz(url, email, secret) is implemented to solve one quiz and return JSON.
from .config import QUIZ_SECRET, MAX_TOTAL_SECONDS
from .quiz_solver import solve_single_quiz

# If Windows, set appropriate loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Quiz Receiver (Manual Start)")

TASKS_FILE = os.environ.get("TASKS_FILE", "tasks_store.json")
_TASKS_LOCK = asyncio.Lock()
# in-memory cache of tasks (loaded from TASKS_FILE)
TASKS: Dict[str, Dict[str, Any]] = {}


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.on_event("startup")
async def load_tasks_on_startup():
    """Load persisted tasks from disk into TASKS (if file exists)."""
    global TASKS
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as fh:
                TASKS = json.load(fh)
            logger.info("Loaded %d tasks from %s", len(TASKS), TASKS_FILE)
        except Exception as e:
            logger.exception("Failed to load tasks file: %s", e)
            TASKS = {}
    else:
        TASKS = {}


async def persist_tasks_to_disk():
    """Persist TASKS to disk (safe write)."""
    tmp = TASKS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(TASKS, fh, indent=2, default=str)
        os.replace(tmp, TASKS_FILE)
    except Exception:
        logger.exception("Failed to persist tasks to disk")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Return 400 for invalid JSON / schema validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON payload", "errors": exc.errors()},
    )


@app.post("/submit", status_code=200)
async def submit_quiz_endpoint(request_data: QuizRequest):
    """
    Receive the evaluator's POST and queue the task for manual processing.
    Validates the secret. Does NOT auto-solve the quiz chain.
    """
    logger.info("Received /submit request for email=%s url=%s", request_data.email, request_data.url)

    # Verify secret
    if request_data.secret != QUIZ_SECRET:
        logger.warning("Invalid secret for email=%s", request_data.email)
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Create a task entry
    task_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    task = {
        "id": task_id,
        "email": request_data.email,
        "secret": request_data.secret,
        "url": request_data.url,
        "received_at": now,
        "status": "queued",  # queued, running, done, failed, cancelled
        "result": None,
        "attempts": 0,
    }

    async with _TASKS_LOCK:
        TASKS[task_id] = task
        await persist_tasks_to_disk()

    return {"status": "queued", "task_id": task_id}


# Support the old route name (/quiz) as an alias for backwards compatibility.
@app.post("/quiz", status_code=200)
async def quiz_alias(request_data: QuizRequest):
    return await submit_quiz_endpoint(request_data)


@app.get("/tasks", status_code=200)
async def list_tasks():
    """Return a list of all tasks (basic info)."""
    async with _TASKS_LOCK:
        items = [
            {
                "id": t["id"],
                "email": t["email"],
                "url": t["url"],
                "received_at": t["received_at"],
                "status": t["status"],
            }
            for t in TASKS.values()
        ]
    return {"tasks": items}


@app.get("/task/{task_id}", status_code=200)
async def get_task(task_id: str):
    """Return full task data by id."""
    async with _TASKS_LOCK:
        t = TASKS.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="task not found")
    return t


@app.post("/start/{task_id}", status_code=200)
async def start_task(task_id: str):
    """
    Manually start solving a single stored task.
    - Calls solve_single_quiz(url, email, secret)
    - Enforces MAX_TOTAL_SECONDS timeout for the solver
    - Updates task status/result and persists
    """
    async with _TASKS_LOCK:
        task = TASKS.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="task not found")

    if task["status"] not in ("queued", "failed"):
        raise HTTPException(status_code=400, detail=f"task in invalid state: {task['status']}")

    # mark running
    task["status"] = "running"
    task["started_at"] = datetime.utcnow().isoformat() + "Z"
    task["attempts"] = task.get("attempts", 0) + 1

    async with _TASKS_LOCK:
        TASKS[task_id] = task
        await persist_tasks_to_disk()

    # Call the solver with a timeout; solver must solve exactly ONE URL and return JSON.
    try:
        logger.info("Starting solver for task %s (url=%s)", task_id, task["url"])
        # ensure solve_single_quiz is awaitable
        coro = solve_single_quiz(task["url"], task["email"], task["secret"])
        # Enforce timeout from config
        result = await asyncio.wait_for(coro, timeout=MAX_TOTAL_SECONDS)
        # Expect result to be a dict / JSON-serializable
        if not isinstance(result, dict):
            result = {"error": "solver-returned-non-dict", "raw": str(result)}
        task["status"] = "done"
        task["completed_at"] = datetime.utcnow().isoformat() + "Z"
        task["result"] = result
        async with _TASKS_LOCK:
            TASKS[task_id] = task
            await persist_tasks_to_disk()

        # Important: we do NOT follow any result["url"] or chain. We return solver's JSON as-is.
        return {"status": "done", "task_id": task_id, "result": result}

    except asyncio.TimeoutError:
        logger.exception("Solver timeout for task %s", task_id)
        task["status"] = "failed"
        task["result"] = {"error": "solver-timeout", "timeout_seconds": MAX_TOTAL_SECONDS}
        task["completed_at"] = datetime.utcnow().isoformat() + "Z"
        async with _TASKS_LOCK:
            TASKS[task_id] = task
            await persist_tasks_to_disk()
        raise HTTPException(status_code=504, detail="solver timeout")

    except Exception as e:
        logger.exception("Solver failed for task %s: %s", task_id, str(e))
        task["status"] = "failed"
        task["result"] = {"error": "solver-exception", "reason": str(e)}
        task["completed_at"] = datetime.utcnow().isoformat() + "Z"
        async with _TASKS_LOCK:
            TASKS[task_id] = task
            await persist_tasks_to_disk()
        raise HTTPException(status_code=500, detail=f"solver error: {e}")


@app.post("/cancel/{task_id}", status_code=200)
async def cancel_task(task_id: str):
    """Mark a task cancelled (if queued)."""
    async with _TASKS_LOCK:
        t = TASKS.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task not found")
        if t["status"] in ("done", "running"):
            raise HTTPException(status_code=400, detail=f"cannot cancel task in status {t['status']}")
        t["status"] = "cancelled"
        t["completed_at"] = datetime.utcnow().isoformat() + "Z"
        TASKS[task_id] = t
        await persist_tasks_to_disk()
    return {"status": "cancelled", "task_id": task_id}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "quiz-solver-manual"}


@app.get("/")
async def root():
    return {
        "service": "Quiz Solver (manual start)",
        "endpoints": {
            "POST /submit": "Accept quiz task (validate secret, queue for manual start)",
            "POST /start/{task_id}": "Manually start solving a queued task (single URL only)",
            "GET /tasks": "List tasks",
            "GET /task/{task_id}": "Get task details",
            "POST /cancel/{task_id}": "Cancel a queued task",
        },
    }
