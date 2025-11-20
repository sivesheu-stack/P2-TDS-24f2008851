import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
from playwright.async_api import async_playwright
from urllib.parse import urljoin

from .config import QUIZ_EMAIL, QUIZ_SECRET, MAX_TOTAL_SECONDS
from .processors.csv_processor import load_csv
from .processors.pdf_processor import extract_text_from_pdf, extract_table_from_pdf

logger = logging.getLogger("uvicorn.error")


# --------- Public entrypoint called from main.py --------- #

async def solve_quiz_chain(start_url: str) -> None:
    """
    Visit the given quiz URL, solve it, submit the answer, and
    follow any chained quiz URLs, all within MAX_TOTAL_SECONDS.
    """
    logger.info("[solver] Starting quiz solving chain for URL: %s", start_url)
    start_time = time.time()
    current_url = start_url

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()

        try:
            while current_url:
                elapsed = time.time() - start_time
                if elapsed > MAX_TOTAL_SECONDS:
                    logger.warning(
                        "[solver] Time limit exceeded (%.1fs). Stopping.", elapsed
                    )
                    break

                logger.info("[solver] >>> Solving quiz at: %s", current_url)
                await page.goto(current_url, wait_until="networkidle", timeout=60_000)

                answer, submit_url = await solve_single_quiz(page, current_url)

                if submit_url is None:
                    logger.error("[solver] No submit URL found on page; stopping.")
                    break

                payload = {
                    "email": QUIZ_EMAIL,
                    "secret": QUIZ_SECRET,
                    "url": current_url,
                    "answer": answer,
                }
                logger.info("[solver] Submitting payload to %s: %s", submit_url, payload)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(submit_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                logger.info("[solver] Submission result: %s", data)
                next_url = data.get("url")
                if not next_url:
                    logger.info("[solver] No next URL; quiz sequence complete.")
                    break

                current_url = next_url
        except Exception as e:
            logger.exception("[solver] ERROR during quiz solving: %r", e)
        finally:
            await browser.close()


# --------- Core per-page solver --------- #

async def solve_single_quiz(page, quiz_url: str) -> Tuple[object, Optional[str]]:
    """
    Inspect the page, extract the question + submit URL, compute the answer.

    Returns:
        (answer, submit_url)
    """
    # 1) Get question text (main description of the task)
    question_text = await extract_question_text(page)
    logger.info("[solver] Question text: %s", question_text[:300])

    # 2) Find submit URL on the page
    submit_url = await extract_submit_url(page, quiz_url)
    logger.info("[solver] Submit URL: %s", submit_url)

    # 3) Find potential download links (CSV/PDF/etc.)
    downloads = await find_download_links(page, quiz_url)
    logger.info("[solver] Found download links: %s", downloads)

    # 4) Compute the answer based on question + downloads
    answer = await compute_answer_from_question(question_text, downloads, page)

    logger.info("[solver] Computed answer: %r", answer)
    return answer, submit_url


# --------- Helpers: extracting info from the page --------- #

async def extract_question_text(page) -> str:
    """
    Try to get the main question text from the page.

    Heuristics:
    - #result div text
    - <h1>, <h2> headings
    - entire body (fallback)
    """
    selectors = ["#result", "main", "article", "h1", "h2"]
    for sel in selectors:
        try:
            text = await page.inner_text(sel)
            if text.strip():
                return text.strip()
        except Exception:
            continue

    # Fallback: whole body
    try:
        return (await page.inner_text("body")).strip()
    except Exception:
        return ""


async def extract_submit_url(page, base: str) -> Optional[str]:
    """
    Try multiple strategies to find the submit URL on the page.
    """
    # Strategy 1: meta tag
    try:
        submit_url = await page.eval_on_selector(
            'meta[name="submit-url"]', "el => el.content"
        )
        if submit_url:
            return urljoin(base, submit_url)
    except Exception:
        pass

    # Strategy 2: look for JSON blob in <pre> or scripts that contain "submit"
    try:
        # Any <pre> elements with JSON
        pres = await page.eval_on_selector_all(
            "pre",
            "els => els.map(e => e.innerText)",
        )
        for txt in pres:
            try:
                j = json.loads(txt)
                for key in ("submit", "submit_url", "submitUrl", "url"):
                    if key in j and "submit" in key.lower():
                        return urljoin(base, j[key])
            except Exception:
                # maybe embedded JSON-like snippet; skip
                continue
    except Exception:
        pass

    # Strategy 3: scan all links for "submit"
    try:
        links = await page.eval_on_selector_all(
            "a",
            "els => els.map(e => ({ href: e.href, text: e.innerText }))",
        )
        for link in links:
            href = link.get("href") or ""
            if "submit" in href.lower():
                return urljoin(base, href)
    except Exception:
        pass

    return None


async def find_download_links(page, base: str) -> List[Dict[str, str]]:
    """
    Return a list of downloadable file links found on the page.

    Each item: { "url": ..., "text": ..., "ext": ... }
    """
    results: List[Dict[str, str]] = []
    try:
        links = await page.eval_on_selector_all(
            "a",
            "els => els.map(e => ({ href: e.getAttribute('href'), text: e.innerText }))",
        )
        for link in links:
            href = link.get("href")
            if not href:
                continue
            full = urljoin(base, href)
            ext = os.path.splitext(full.split("?", 1)[0])[1].lower()
            results.append({"url": full, "text": link.get("text", "").strip(), "ext": ext})
    except Exception as e:
        logger.warning("[solver] Error while scanning download links: %r", e)
    return results


# --------- Helpers: downloading & parsing data --------- #

async def download_file(url: str) -> str:
    """
    Download a file to a temporary location and return the local path.
    """
    logger.info("[solver] Downloading file: %s", url)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        suffix = os.path.splitext(url.split("?", 1)[0])[1]
        fd, path = tempfile.mkstemp(suffix=suffix or ".dat")
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
    logger.info("[solver] Saved file to: %s", path)
    return path


async def compute_answer_from_question(
    question: str, downloads: List[Dict[str, str]], page
) -> object:
    """
    Very simple "NLP + data" engine for the quiz.

    Handles typical patterns such as:
    - "What is the sum of the 'value' column in the table on page 2?"
    """

    q_lower = question.lower()

    # 1) Detect "sum of the 'X' column"
    sum_match = re.search(
        r"sum of the ['“\"]?(.+?)['”\"]? column", question, flags=re.IGNORECASE
    )
    page_match = re.search(r"page\s+(\d+)", question, flags=re.IGNORECASE)

    if sum_match:
        col_name = sum_match.group(1).strip()
        logger.info("[solver] Detected sum-of-column question for column: %s", col_name)

        page_index = 0
        if page_match:
            # pages are 1-based in text, 0-based in parser
            page_index = max(int(page_match.group(1)) - 1, 0)
            logger.info("[solver] Detected page constraint: page %d (index %d)",
                        int(page_match.group(1)), page_index)

        # Prefer CSV if available, otherwise PDF
        csv_links = [d for d in downloads if d["ext"] in (".csv", ".tsv")]
        pdf_links = [d for d in downloads if d["ext"] == ".pdf"]

        if csv_links:
            path = await download_file(csv_links[0]["url"])
            df = load_csv(path)
            # Try exact match; fallback to case-insensitive
            if col_name in df.columns:
                return float(df[col_name].sum())
            else:
                for c in df.columns:
                    if c.lower().strip() == col_name.lower().strip():
                        return float(df[c].sum())
            logger.warning("[solver] Column %s not found in CSV columns: %s",
                           col_name, df.columns.tolist())
            # Fallback: sum first numeric column
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                return float(df[num_cols[0]].sum())

        elif pdf_links:
            path = await download_file(pdf_links[0]["url"])
            try:
                df = extract_table_from_pdf(path, page_index)
                # Use first row as header if looks like header
                df.columns = [str(c) for c in df.iloc[0]]
                df = df[1:]
                if col_name in df.columns:
                    numeric = pd.to_numeric(df[col_name], errors="coerce")
                    return float(numeric.sum())
                else:
                    for c in df.columns:
                        if c.lower().strip() == col_name.lower().strip():
                            numeric = pd.to_numeric(df[c], errors="coerce")
                            return float(numeric.sum())
            except Exception as e:
                logger.warning("[solver] Error parsing PDF table: %r", e)

    # 2) Simple HTML table sum if no files
    if "sum" in q_lower and "column" in q_lower:
        try:
            # Grab first table on page
            table_data = await page.eval_on_selector_all(
                "table",
                """
                els => {
                    if (els.length === 0) return null;
                    const rows = Array.from(els[0].rows);
                    return rows.map(r =>
                        Array.from(r.cells).map(c => c.innerText.trim())
                    );
                }
                """,
            )
            if table_data:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                sum_match = re.search(
                    r"sum of the ['“\"]?(.+?)['”\"]? column",
                    question,
                    flags=re.IGNORECASE,
                )
                if sum_match:
                    col_name = sum_match.group(1).strip()
                    if col_name in df.columns:
                        numeric = pd.to_numeric(df[col_name], errors="coerce")
                        return float(numeric.sum())
        except Exception as e:
            logger.warning("[solver] Error computing sum from HTML table: %r", e)

    # 3) Fallback: if quiz expects a boolean or text-based answer, you can
    # add more heuristics here. For now, just return None so at least the
    # submit request is well-formed.
    logger.warning("[solver] Could not understand question; returning None.")
    return None

