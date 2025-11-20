import fitz  # pymupdf
import pandas as pd


def extract_text_from_pdf(path: str) -> str:
    """
    Extracts and returns the full text of a PDF file.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        # Coerce various possible return types of get_text to a string
        res = page.get_text("text")
        if isinstance(res, str):
            text += res
        elif isinstance(res, (list, tuple)):
            # join list-like results into text lines
            text += "\n".join(map(str, res))
        elif isinstance(res, dict):
            # try to pick a textual field if available, otherwise stringify
            if "text" in res and isinstance(res["text"], str):
                text += res["text"]
            else:
                text += str(res)
        else:
            text += str(res)
    doc.close()
    return text


def extract_table_from_pdf(path: str, page_number: int = 0):
    """
    Extracts table-like blocks from a PDF page and returns a pandas DataFrame.
    Works best with tabular PDFs (not scanned images).
    """
    doc = fitz.open(path)
    page = doc[page_number]
    text = page.get_text("text")
    
    # Ensure text is a string, using the same coercion logic as extract_text_from_pdf
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            text = "\n".join(map(str, text))
        elif isinstance(text, dict):
            text = text.get("text", "") if isinstance(text.get("text"), str) else str(text)
        else:
            text = str(text)

    # Simple, robust heuristic: split rows by newline, columns by whitespace.
    rows = [r.strip() for r in text.split("\n") if r.strip()]
    split_rows = [r.split() for r in rows]

    # Convert into DataFrame with best-effort formatting
    max_cols = max(len(r) for r in split_rows)
    df = pd.DataFrame([r + [""] * (max_cols - len(r)) for r in split_rows])

    doc.close()
    return df
