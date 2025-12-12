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


# ======================================================
# Utility: Extract submit URL (not used now, but kept)
# ======================================================
async def find_submit_url(page, base_url: str) -> Optional[str]:
    """
    Extract submit URL from quiz page.
    (Unused since you no longer submit, but kept for future use.)
    """
    try:
        # form action
        form = await page.query_selector("form[action]")
        if form:
            action = await form.get_attribute("action")
            if action:
                return urljoin(base_url, action)

        # anchor submit
        anchor = await page.query_selector("a[href*='/submit'], a[href*='submit']")
        if anchor:
            href = await anchor.get_attribute("href")
            if href:
                return urljoin(base_url, href)

        # data-submit-url
        el = await page.query_selector("[data-submit-url], button[data-submit-url]")
        if el:
            u = await el.get_attribute("data-submit-url")
            if u:
                return urljoin(base_url, u)

        # onclick handlers
        onclick_el = await page.query_selector("button[onclick], a[onclick]")
        if onclick_el:
            oc = await onclick_el.get_attribute("onclick") or ""
            m = re.search(r"(https?://[^\s'\"\\)]+/submit[^\s'\"\\)]*)", oc)
            if m:
                return m.group(1)

        content = await page.content()

        # absolute
        m_full = re.search(r"https?://[^\s'\"\\]+/submit[^\s'\"\\]*", content)
        if m_full:
            return m_full.group(0)

        # relative
        m_rel = re.search(r"['\"](/[^'\"\\]*?/submit[^'\"\\]*)['\"]", content)
        if m_rel:
            return urljoin(base_url, m_rel.group(1))

        # base64 search
        base64_matches = re.findall(r"[A-Za-z0-9+/=]{40,}", content)
        for b64 in base64_matches:
            try:
                decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                if "/submit" in decoded:
                    m2 = re.search(r"https?://[^\s'\"\\]+/submit[^\s'\"\\]*", decoded)
                    if m2:
                        return m2.group(0)
                rel = re.search(r'["\'](/[^"\']*?/submit[^"\']*)["\']', decoded)
                if rel:
                    return urljoin(base_url, rel.group(1))
            except Exception:
                continue

    except Exception:
        return None

    return None


# ======================================================
# Download helpers
# ======================================================
async def download_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        r = await client.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.warning(f"[solver] download failed {url} : {e}")
        return None


def parse_csv_bytes(data: bytes) -> Optional[pd.DataFrame]:
    try:
        try:
            text = data.decode("utf-8")
            return pd.read_csv(StringIO(text))
        except Exception:
            return pd.read_csv(BytesIO(data))
    except Exception:
        return None


def extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    try:
        return pd.read_html(html)
    except Exception:
        return []


def extract_tables_from_pdf_bytes(pdf_bytes: bytes, page_num: Optional[int] = None) -> List[pd.DataFrame]:
    tables = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return tables

    if page_num:
        pages = [page_num - 1]
    else:
        pages = range(len(doc))

    for pidx in pages:
        try:
            page = doc[pidx]
            text = str(page.get_text("text"))
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            for blk in blocks:
                lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
                if len(lines) >= 2 and any(re.search(r"\s{2,}", ln) for ln in lines):
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


# ======================================================
# NLP: extract operation & column from question text
# ======================================================
def detect_operation_and_column(question: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if not question:
        return None, None, None
    q = question.lower()

    op = None
    if "sum" in q:
        op = "sum"
    elif "mean" in q or "average" in q:
        op = "mean"
    elif "count" in q:
        op = "count"
    elif "max" in q or "largest" in q:
        op = "max"
    elif "min" in q or "smallest" in q:
        op = "min"

    col = None
    m = re.search(r"['\"]?([\w\s\-]+?)['\"]?\s+column", q)
    if m:
        col = m.group(1).strip()

    m2 = re.search(r"page\s+(\d+)", q)
    page = int(m2.group(1)) if m2 else None

    return op, col, page


# ======================================================
# Compute aggregation
# ======================================================
def compute_aggregation_from_df(df: pd.DataFrame, op: str, col_hint: Optional[str]):
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    normalized = {c.lower(): c for c in cols}

    def choose_col(hint):
        if not hint:
            for cand in ("value", "amount", "price", "total", "count"):
                if cand in normalized:
                    return normalized[cand]
            # any numeric column
            for c in cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
            return None

        h = hint.lower()
        if h in normalized:
            return normalized[h]

        for k, orig in normalized.items():
            if h in k:
                return orig

        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c

        return None

    col = choose_col(col_hint)
    if col is None:
        return None

    s = pd.to_numeric(df[col], errors="coerce")

    if op == "sum":
        return float(s.sum())
    if op == "mean":
        return float(s.mean())
    if op == "count":
        return int(s.count())
    if op == "max":
        return float(s.max())
    if op == "min":
        return float(s.min())

    return None


# ======================================================
# Detect type for final answer
# ======================================================
def detect_answer_type(ans: Any) -> Any:
    if ans is None:
        return None
    if isinstance(ans, (int, float, bool)):
        return ans

    if isinstance(ans, str):
        a = ans.strip().lower()
        if a in ("true", "yes", "y"):
            return True
        if a in ("false", "no", "n"):
            return False
        try:
            if "." in ans:
                v = float(ans)
                return int(v) if v.is_integer() else v
            return int(ans)
        except:
            return ans

    return ans


# ======================================================
# MAIN SOLVER — Now returns answer directly
# ======================================================
async def solve_single_quiz(quiz_url: str, email: str, secret: str) -> Dict[str, Any]:

    warnings = []
    errors = []

    logger.info(f"[solver] Opening quiz page: {quiz_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Load page
        try:
            await page.goto(quiz_url, timeout=30000)
        except Exception as e:
            await browser.close()
            return {
                "email": email,
                "secret": secret,
                "url": quiz_url,
                "answer": None,
                "errors": [f"Page load failed: {e}"]
            }

        # Extract question text
        q_text = None
        try:
            for sel in ("#result", "main", ".question", "pre", "body"):
                el = await page.query_selector(sel)
                if el:
                    t = (await el.inner_text()).strip()
                    if t:
                        q_text = t
                        break
        except Exception:
            pass

        if q_text is None:
            try:
                q_text = (await page.inner_text("body")).strip()[:5000]
            except:
                q_text = ""

        logger.info(f"[solver] QTEXT: {q_text[:200]}")

        # Data extraction logic
        answer = None

        async with httpx.AsyncClient(timeout=60) as client:

            # Find downloadable CSV/PDF
            download_links = []
            anchors = await page.query_selector_all("a[href]")
            for a in anchors:
                href = await a.get_attribute("href")
                if href:
                    abs_url = urljoin(quiz_url, href)
                    if abs_url.lower().endswith((".csv", ".pdf", ".xlsx")):
                        download_links.append(abs_url)

            # CSV handling
            csv_link = next((u for u in download_links if u.endswith(".csv")), None)
            if csv_link:
                data = await download_bytes(client, csv_link)
                if data:
                    df = parse_csv_bytes(data)
                    if df is not None:
                        op, col_hint, page_num = detect_operation_and_column(q_text)
                        if op:
                            answer = compute_aggregation_from_df(df, op, col_hint)

            # PDF handling
            if answer is None:
                pdf = next((u for u in download_links if u.endswith(".pdf")), None)
                if pdf:
                    data = await download_bytes(client, pdf)
                    if data:
                        op, col_hint, page_num = detect_operation_and_column(q_text)
                        pdf_tables = extract_tables_from_pdf_bytes(data, page_num)
                        for tbl in pdf_tables:
                            ans = compute_aggregation_from_df(tbl, op or "sum", col_hint)
                            if ans is not None:
                                answer = ans
                                break

            # HTML tables
            if answer is None:
                html = await page.content()
                tables = extract_tables_from_html(html)
                for tbl in tables:
                    op, col_hint, _ = detect_operation_and_column(q_text)
                    ans = compute_aggregation_from_df(tbl, op or "sum", col_hint)
                    if ans is not None:
                        answer = ans
                        break

            # Plain text answer pattern
            if answer is None:
                m = re.search(r"answer\s*(?:is|:)\s*([+-]?\d+(?:\.\d+)?)", q_text, re.I)
                if m:
                    v = m.group(1)
                    try:
                        answer = float(v) if "." in v else int(v)
                    except:
                        pass

        await browser.close()

    # Normalize type
    answer = detect_answer_type(answer)
    if answer is None:
        warnings.append("computed_answer_is_none")

    # Final output — no submission
    result = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
        "warnings": warnings or None,
        "errors": errors or None
    }

    logger.info(f"[solver] Returning final answer: {result}")
    return result
