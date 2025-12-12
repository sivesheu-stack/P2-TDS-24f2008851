# quiz_solver.py
import asyncio
import json
import logging
import os
import re
import base64
from io import BytesIO, StringIO
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urljoin

import httpx
import pandas as pd
import fitz  # PyMuPDF
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger("quiz_solver")
logger.setLevel(logging.INFO)


# -------------------------
# Utility: robust submit URL extraction
# -------------------------
async def find_submit_url(page, base_url: str) -> Optional[str]:
    """
    Heuristic extraction of a submit URL from a page.
    Checks: <form action>, anchors, data-* attributes, onclick handlers, script blobs, base64-encoded payloads.
    Returns absolute URL or None.
    """
    try:
        # 1) <form action>
        form = await page.query_selector("form[action]")
        if form:
            action = await form.get_attribute("action")
            if action:
                logger.debug("[solver] Found form action: %s", action)
                return urljoin(base_url, action)

        # 2) anchors with submit
        anchor = await page.query_selector("a[href*='/submit'], a[href*='submit']")
        if anchor:
            href = await anchor.get_attribute("href")
            if href:
                logger.debug("[solver] Found anchor href submit: %s", href)
                return urljoin(base_url, href)

        # 3) direct button data attributes
        el = await page.query_selector("[data-submit-url], button[data-submit-url]")
        if el:
            u = await el.get_attribute("data-submit-url")
            if u:
                logger.debug("[solver] Found data-submit-url: %s", u)
                return urljoin(base_url, u)

        # 4) onclick handlers (may contain full url)
        onclick_el = await page.query_selector("button[onclick], a[onclick]")
        if onclick_el:
            oc = await onclick_el.get_attribute("onclick") or ""
            m = re.search(r"(https?://[^\s'\"\\)]+/submit[^\s'\"\\)]*)", oc)
            if m:
                logger.debug("[solver] Found onclick full URL: %s", m.group(1))
                return m.group(1)

        # 5) search script tags and page content for submit URL
        content = await page.content()

        # full absolute URL
        m_full = re.search(r"https?://[^\s'\"\\]+/submit[^\s'\"\\]*", content)
        if m_full:
            logger.debug("[solver] Found full submit in page content: %s", m_full.group(0))
            return m_full.group(0)

        # relative path in quotes
        m_rel = re.search(r"['\"](/[^'\"\\]*?/submit[^'\"\\]*)['\"]", content)
        if m_rel:
            logger.debug("[solver] Found relative submit in page content: %s", m_rel.group(1))
            return urljoin(base_url, m_rel.group(1))

        # 6) base64-encoded JSON in page (common pattern in tasks)
        # find base64 strings that decode to JSON containing "submit" or "/submit"
        base64_matches = re.findall(r"[A-Za-z0-9+/=]{40,}", content)
        for b64 in base64_matches:
            try:
                decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                if "/submit" in decoded or '"submit"' in decoded:
                    m = re.search(r"https?://[^\s'\"\\]+/submit[^\s'\"\\]*", decoded)
                    if m:
                        logger.debug("[solver] Found submit in base64 JSON: %s", m.group(0))
                        return m.group(0)
                    m2 = re.search(r'["\'](/[^"\']*?/submit[^"\']*)["\']', decoded)
                    if m2:
                        logger.debug("[solver] Found relative submit in base64 JSON: %s", m2.group(1))
                        return urljoin(base_url, m2.group(1))
            except Exception:
                continue

    except Exception as e:
        logger.debug("[solver] find_submit_url error: %s", e)

    return None


# -------------------------
# Utilities: download and parse helpers
# -------------------------
async def download_bytes(client: httpx.AsyncClient, url: str, timeout: float = 30.0) -> Optional[bytes]:
    try:
        r = await client.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.warning("[solver] download failed %s : %s", url, e)
        return None


def parse_csv_bytes(data: bytes) -> Optional[pd.DataFrame]:
    try:
        try:
            text = data.decode("utf-8")
            return pd.read_csv(StringIO(text))
        except Exception:
            return pd.read_csv(BytesIO(data))
    except Exception as e:
        logger.debug("[solver] parse_csv_bytes failed: %s", e)
        return None


def extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(html)
        return tables
    except Exception as e:
        logger.debug("[solver] extract_tables_from_html failed: %s", e)
        return []


def extract_tables_from_pdf_bytes(pdf_bytes: bytes, page_num: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Simple extraction: use PyMuPDF to extract text from target page(s)
    and attempt to parse tabular-looking blocks with pandas.read_csv on whitespace-separated text.
    Not perfect, but handles many common task PDFs.
    """
    tables = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.debug("[solver] open pdf failed: %s", e)
        return tables

    pages = []
    if page_num is not None and 1 <= page_num <= len(doc):
        pages = [page_num - 1]
    else:
        pages = range(len(doc))

    for pidx in pages:
        try:
            page = doc[pidx]
            text = page.get_text("text")
            # quick heuristic: if there's a table-like chunk with multiple whitespace columns,
            # try to parse by treating multiple spaces as delimiter
            text = str(text) if text else ""
            blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 0]
            for blk in blocks:
                # detect if block looks like rows with multiple columns
                lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
                if len(lines) >= 2 and any(re.search(r"\s{2,}", ln) for ln in lines):
                    # convert multiple spaces to comma and parse
                    blob = "\n".join(re.sub(r"\s{2,}", ",", ln) for ln in lines)
                    try:
                        df = pd.read_csv(StringIO(blob))
                        if not df.empty:
                            tables.append(df)
                    except Exception:
                        continue
        except Exception:
            continue

    return tables


# -------------------------
# NLP-ish heuristics to compute requested aggregation
# -------------------------
def detect_operation_and_column(question_text: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Returns (operation, column_name, pdf_page_number)
    operation one of: 'sum','mean','count','min','max'
    pdf_page_number if question mentions page X, else None
    """
    if not question_text:
        return None, None, None
    q = question_text.lower()

    # detect operation
    op = None
    if re.search(r"\bsum\b", q):
        op = "sum"
    elif re.search(r"\baverage\b|\bmean\b", q):
        op = "mean"
    elif re.search(r"\bcount\b", q):
        op = "count"
    elif re.search(r"\bmax\b|\bmaximum\b|\blargest\b", q):
        op = "max"
    elif re.search(r"\bmin\b|\bminimum\b|\bsmallest\b", q):
        op = "min"

    # detect column name like "the 'value' column" or column named value
    col = None
    m = re.search(r"['\"]?([\w\s\-]+?)['\"]?\s+column", q)
    if m:
        col = m.group(1).strip()
    else:
        # look for column-name after 'column <name>' or 'column named X'
        m2 = re.search(r"column (named|called)?\s*['\"]?([\w\s\-]+?)['\"]?(?:[\. ,]|$)", q)
        if m2:
            col = m2.group(2).strip()

    # detect page number (pdf)
    page_num = None
    mpage = re.search(r"page\s+(\d+)", q)
    if mpage:
        try:
            page_num = int(mpage.group(1))
        except Exception:
            page_num = None

    return op, col, page_num


def compute_aggregation_from_df(df: pd.DataFrame, op: str, col_hint: Optional[str]) -> Optional[Any]:
    """Try to compute an aggregation on df given operation and column hint."""
    if df is None or df.empty:
        return None

    # normalize column names
    cols = list(df.columns)
    normalized = {c.lower().strip(): c for c in cols}

    def choose_column(hint):
        if not hint:
            # prefer obvious numeric columns
            for cand in ("value", "amount", "price", "total", "count"):
                if cand in normalized:
                    return normalized[cand]
            # fallback to any numeric column
            for c in cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
            return None
        # match hint case-insensitively
        h = hint.lower().strip()
        # exact match
        if h in normalized:
            return normalized[h]
        # try substring match
        for k, orig in normalized.items():
            if h in k:
                return orig
        # fallback to any numeric
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    colname = choose_column(col_hint)

    if not colname:
        return None

    try:
        series = pd.to_numeric(df[colname], errors="coerce")
        if op == "sum":
            return float(series.sum(skipna=True))
        if op == "mean":
            return float(series.mean(skipna=True))
        if op == "count":
            return int(series.count())
        if op == "max":
            return float(series.max(skipna=True))
        if op == "min":
            return float(series.min(skipna=True))
    except Exception as e:
        logger.debug("[solver] compute_aggregation_from_df error: %s", e)
        return None

    return None


def detect_answer_type(answer: Any) -> Any:
    """
    Convert answer to appropriate type (bool, int, float, string).
    Handles common patterns in quiz answers.
    """
    if answer is None:
        return None

    # If already a number, return as-is
    if isinstance(answer, (int, float)):
        # Convert to int if it's a whole number
        if isinstance(answer, float) and answer.is_integer():
            return int(answer)
        return answer

    # If boolean
    if isinstance(answer, bool):
        return answer

    # If string, try to parse
    if isinstance(answer, str):
        answer_lower = answer.lower().strip()

        # Check for boolean strings
        if answer_lower in ('true', 'yes', 'y'):
            return True
        if answer_lower in ('false', 'no', 'n'):
            return False

        # Try to parse as number
        try:
            if '.' in answer:
                val = float(answer)
                return int(val) if val.is_integer() else val
            else:
                return int(answer)
        except ValueError:
            # Return as string
            return answer

    return answer


# -------------------------
# Main solver logic - now returns response instead of just payload
# -------------------------
async def solve_single_quiz(quiz_url: str, email: str, secret: str) -> Dict[str, Any]:
    """
    Visit quiz_url, compute answer using multiple strategies,
    build JSON payload and POST to the submit URL.

    Returns a JSON-serializable dict describing the attempt and any submit response.
    Never follows subsequent URLs; returns only the submit response for this single URL.
    """
    logger.info("[solver] Starting quiz solving for URL: %s", quiz_url)

    browser = None
    page = None
    submit_url = None
    answer = None
    warnings: List[str] = []
    errors: List[str] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                await page.goto(quiz_url, timeout=30000)
            except PlaywrightTimeoutError:
                msg = f"Timeout loading quiz url: {quiz_url}"
                logger.error("[solver] %s", msg)
                return {"submitted": False, "error": "page-load-timeout", "message": msg}

            # extract question text
            q_text = None
            try:
                for sel in ("#result", "main", "article", "pre", ".question", "body"):
                    el = await page.query_selector(sel)
                    if el:
                        txt = (await el.inner_text()).strip()
                        if txt:
                            q_text = txt
                            break
                if not q_text:
                    q_text = (await page.inner_text("body")).strip()[:4000]
            except Exception as e:
                logger.debug("[solver] extracting question text failed: %s", e)
                q_text = None

            logger.info(
                "[solver] Question text: %s",
                (q_text[:500] + "...") if q_text and len(q_text) > 500 else q_text,
            )

            # find submit URL on page
            submit_url = await find_submit_url(page, quiz_url)
            if not submit_url:
                msg = "Could not find submit URL on page"
                logger.error("[solver] %s", msg)
                return {"submitted": False, "error": "no-submit-url", "message": msg, "quiz_url": quiz_url}

            logger.info("[solver] Submit URL: %s", submit_url)

            # collect downloadable links for data
            download_links: List[str] = []
            try:
                anchors = await page.query_selector_all("a[href]")
                for a in anchors:
                    href = await a.get_attribute("href")
                    if href:
                        href_abs = urljoin(quiz_url, href)
                        if any(href_abs.lower().endswith(ext) for ext in (".csv", ".pdf", ".xlsx")):
                            download_links.append(href_abs)
            except Exception:
                pass

            logger.info("[solver] Found %d downloadable files", len(download_links))

            # attempt multiple solving strategies
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1) If there's a CSV, try it first
                csv_link = next((u for u in download_links if u.lower().endswith(".csv")), None)
                if csv_link:
                    logger.info("[solver] Downloading CSV: %s", csv_link)
                    b = await download_bytes(client, csv_link)
                    if b:
                        df = parse_csv_bytes(b)
                        if df is not None:
                            logger.info("[solver] Parsed CSV with shape: %s", df.shape)
                            op, col_hint, page_num = detect_operation_and_column(q_text)
                            logger.info("[solver] Detected operation=%s, column=%s", op, col_hint)
                            if op:
                                answer = compute_aggregation_from_df(df, op, col_hint)
                                if answer is not None:
                                    logger.info("[solver] Computed answer from CSV: %s", answer)

                # 2) If PDF exists, attempt table extraction
                if answer is None:
                    pdf_link = next((u for u in download_links if u.lower().endswith(".pdf")), None)
                    if pdf_link:
                        logger.info("[solver] Downloading PDF: %s", pdf_link)
                        bpdf = await download_bytes(client, pdf_link)
                        if bpdf:
                            op, col_hint, page_num = detect_operation_and_column(q_text)
                            logger.info("[solver] Extracting tables from PDF (page: %s)", page_num or "all")
                            # try extracting tables for the mentioned page first
                            tables = extract_tables_from_pdf_bytes(bpdf, page_num)
                            logger.info("[solver] Extracted %d tables from PDF", len(tables))
                            for idx, tbl in enumerate(tables):
                                candidate = compute_aggregation_from_df(tbl, op or "sum", col_hint)
                                if candidate is not None:
                                    logger.info("[solver] Computed answer from PDF table %d: %s", idx, candidate)
                                    answer = candidate
                                    break

                # 3) If HTML tables exist on the page itself
                if answer is None:
                    html = await page.content()
                    tables = extract_tables_from_html(html)
                    if tables:
                        logger.info("[solver] Found %d HTML tables on page", len(tables))
                        for idx, tbl in enumerate(tables):
                            op, col_hint, _ = detect_operation_and_column(q_text)
                            candidate = compute_aggregation_from_df(tbl, op or "sum", col_hint)
                            if candidate is not None:
                                logger.info("[solver] Computed answer from HTML table %d: %s", idx, candidate)
                                answer = candidate
                                break

                # 4) If still none, try to parse numbers or direct pattern from question text
                if answer is None and q_text:
                    # look for simple explicit "answer is 123" patterns
                    m = re.search(
                        r"(?:answer\s*(?:is|:)|the answer is)\s*([+-]?\d+(?:\.\d+)?)",
                        q_text,
                        re.IGNORECASE,
                    )
                    if m:
                        try:
                            v = m.group(1)
                            answer = float(v) if "." in v else int(v)
                            logger.info("[solver] Found explicit answer in question text: %s", answer)
                        except Exception:
                            answer = None

                # 5) Last-ditch: for CSVs that weren't picked up, try generic numeric column sum
                if answer is None and csv_link:
                    b = await download_bytes(client, csv_link)
                    if b:
                        df2 = parse_csv_bytes(b)
                        if df2 is not None:
                            # choose first numeric column
                            for c in df2.columns:
                                if pd.api.types.is_numeric_dtype(df2[c]):
                                    try:
                                        answer = float(df2[c].sum(skipna=True))
                                        logger.info("[solver] Fallback: sum of column '%s': %s", c, answer)
                                        break
                                    except Exception:
                                        continue

    except Exception as e:
        logger.exception("[solver] Unexpected exception: %s", e)
        errors.append(str(e))
    finally:
        # Ensure the browser closes if it was opened
        try:
            if browser:
                await browser.close()
        except Exception:
            logger.exception("[solver] Error closing browser")

    # Normalize answer type
    answer = detect_answer_type(answer)

    if answer is None:
        warnings.append("computed_answer_is_none")

    logger.info("[solver] Final computed answer: %s (type: %s)", repr(answer), type(answer).__name__)

    # Ensure submit_url exists before attempting POST
    if not submit_url:
        msg = "submit_url not found; cannot POST answer"
        logger.error("[solver] %s for quiz_url=%s", msg, quiz_url)
        return {
            "submitted": False,
            "error": "no-submit-url",
            "message": msg,
            "quiz_url": quiz_url,
            "computed_answer": answer,
            "warnings": warnings or None,
            "errors": errors or None,
        }

    # Build payload (keep consistent shape even if answer is None)
    payload: Dict[str, Any] = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    # Mask secret for logging
    safe_payload = {**payload, "secret": "***REDACTED***"}
    logger.info("[solver] Submitting to: %s", submit_url)
    logger.debug("[solver] Payload (safe): %s", safe_payload)

    submit_result: Dict[str, Any] = {
        "submitted": False,
        "payload": payload,
        "warnings": warnings or None,
        "errors": errors or None,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(submit_url, json=payload)
            submit_result["submitted"] = True
            submit_result["http_status"] = resp.status_code
            try:
                submit_result["response_json"] = resp.json()
            except Exception:
                submit_result["response_text"] = (resp.text or "")[:10000]
    except Exception as e:
        logger.exception("[solver] Error posting payload: %s", e)
        submit_result["submitted"] = False
        submit_result["post_error"] = str(e)

    return submit_result
