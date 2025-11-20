# LLM Quiz Solver

This service receives quiz tasks via POST, verifies a shared secret, and uses a
headless browser (Playwright) plus Python data tools to solve data-related
quizzes and submit answers automatically.

## Tech stack

- Python 3.11+
- FastAPI
- Playwright (Chromium)
- httpx
- pandas, pymupdf, etc.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
playwright install  # installs browser binaries
