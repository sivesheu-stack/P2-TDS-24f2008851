import time
import json
import asyncio
from typing import Optional, Tuple

import httpx
from playwright.async_api import async_playwright

from .config import QUIZ_EMAIL, QUIZ_SECRET, MAX_TOTAL_SECONDS


async def solve_quiz_chain(start_url: str) -> None:
    """
    Visit the given quiz URL, solve it, submit the answer, and
    follow any chained quiz URLs, all within MAX_TOTAL_SECONDS.
    """
    start_time = time.time()
    current_url = start_url

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()

        try:
            while current_url:
                elapsed = time.time() - start_time
                if elapsed > MAX_TOTAL_SECONDS:
                    print(f"[solver] Time budget exceeded ({elapsed:.1f}s). Stopping.")
                    break

                print(f"[solver] Solving quiz at: {current_url}")
                await page.goto(current_url, wait_until="networkidle", timeout=60_000)

                answer, submit_url = await solve_single_quiz(page, current_url)

                if submit_url is None:
                    print("[solver] No submit URL detected; stopping.")
                    break

                payload = {
                    "email": QUIZ_EMAIL,
                    "secret": QUIZ_SECRET,
                    "url": current_url,
                    "answer": answer,
                }

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(submit_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                print(f"[solver] Submission result: {data}")
                # Expected keys: "correct": bool, optionally "url": next_url
                next_url = data.get("url")
                if not next_url:
                    print("[solver] No next URL; quiz sequence complete.")
                    break

                current_url = next_url
        finally:
            await browser.close()


async def solve_single_quiz(page, quiz_url: str) -> Tuple[object, Optional[str]]:
    """
    Inspect the page, extract the question + submit URL, compute the answer.

    Returns:
        (answer, submit_url)
    - answer: A Python object that will be JSON-serialised (number/string/bool/dict/etc).
    - submit_url: The endpoint to POST the answer to.
    """
    # 1) Try to find the submit URL in a <meta> or data attribute (you’ll adapt this)
    submit_url = None
    try:
        submit_url = await page.eval_on_selector(
            'meta[name="submit-url"]',
            "el => el.content",
        )
    except Exception:
        pass

    # Fallback: search inside page text for something that looks like JSON with a submit URL
    if not submit_url:
        try:
            body_text = await page.inner_text("body")
            # TODO: Improve this logic for real quizzes
            # Example heuristic: look for "https://" + "submit" etc.
            import re
            m = re.search(r'https?://[^\s"]*submit[^\s"]*', body_text)
            if m:
                submit_url = m.group(0)
        except Exception:
            pass

    # 2) Compute the answer.
    # You will extend this function to:
    #   - click "download" links,
    #   - read CSV/PDF files,
    #   - clean data,
    #   - compute aggregates / charts, etc.
    #
    # For now, we put a placeholder that you will replace.

    answer = await simple_example_answer(page, quiz_url)

    return answer, submit_url


async def simple_example_answer(page, quiz_url: str) -> object:
    """
    Minimal example of extracting something from a JS-rendered page.

    You’ll completely replace/extend this with:
      - PDF parsing (via pymupdf),
      - CSV via pandas,
      - OCR / vision if needed.
    """
    # Example: a div#result holds clear text of the question or answer
    try:
        text = await page.inner_text("#result")
        print(f"[solver] #result text snippet: {text[:200]!r}")
        # In a real quiz, you'd parse 'text' to compute answer.
        # Here we just return a dummy.
        return 12345
    except Exception:
        # If nothing obvious, return something safe or raise
        print("[solver] Could not find #result; returning null answer.")
        return None
