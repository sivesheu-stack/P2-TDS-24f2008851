import time
import json
import asyncio
from typing import Optional, Tuple
import logging
import httpx
from playwright.async_api import async_playwright

from .config import QUIZ_EMAIL, QUIZ_SECRET, MAX_TOTAL_SECONDS

logger = logging.getLogger("uvicorn.error")
async def solve_quiz_chain(start_url: str) -> None:
    logger.info(f"[solver] Starting quiz solving chain for URL: {start_url}")   
    start_time = time.time()
    current_url = start_url

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()

        try:
            while current_url:
                elapsed = time.time() - start_time
                if elapsed > MAX_TOTAL_SECONDS:
                    logger.info(f"[solver] Time limit exceeded ({elapsed:.1f}s). Stopping.")
                    break

                logger.info(f"[solver] >>> Solving quiz at: {current_url}")
                await page.goto(current_url, wait_until="networkidle", timeout=60_000)

                answer, submit_url = await solve_single_quiz(page, current_url)

                if submit_url is None:
                    logger.info("[solver] No submit URL found on page; stopping.")
                    break

                payload = {
                    "email": QUIZ_EMAIL,
                    "secret": QUIZ_SECRET,
                    "url": current_url,
                    "answer": answer,
                }
                logger.info(f"[solver] Submitting payload to {submit_url}: {payload}")

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(submit_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                logger.info(f"[solver] Submission result: {data}")
                next_url = data.get("url")
                if not next_url:
                    logger.info("[solver] No next URL in response; quiz sequence complete.")
                    break

                current_url = next_url
        except Exception as e:
            logger.info(f"[solver] ERROR during quiz solving: {e!r}")
        finally:
            await browser.close()


async def solve_single_quiz(page, quiz_url: str) -> Tuple[object, Optional[str]]:
    """
    TEMP implementation:
    - Print the page text (for debugging)
    - Try to detect a submit URL
    - Return a dummy answer (12345)
    """
    # 1) Dump some text so you can see what the quiz looks like
    try:
        body_text = await page.inner_text("body")
        print("[solver] Page text snippet:")
        print(body_text[:1000])  # first 1000 chars
    except Exception as e:
        print(f"[solver] Could not read body text: {e!r}")
        body_text = ""

    # 2) Try to find a submit URL
    submit_url = None

    # Try meta tag first
    try:
        submit_url = await page.eval_on_selector(
            'meta[name="submit-url"]',
            "el => el.content",
        )
        if submit_url:
            print(f"[solver] Found submit URL via <meta>: {submit_url}")
    except Exception:
        pass

    # If not found, try to search all links for the word 'submit'
    if not submit_url:
        try:
            links = await page.eval_on_selector_all(
                "a",
                "els => els.map(e => ({ href: e.href, text: e.innerText }))",
            )
            for link in links:
                if "submit" in link["href"].lower():
                    submit_url = link["href"]
                    print(f"[solver] Found submit URL via <a>: {submit_url}")
                    break
        except Exception as e:
            print(f"[solver] Error while scanning links: {e!r}")

    # 3) TODO: Real logic to compute answer based on the question
    # For now, just return a dummy number so that the flow works.
    dummy_answer = 12345
    print(f"[solver] Using dummy answer: {dummy_answer}")

    return dummy_answer, submit_url
