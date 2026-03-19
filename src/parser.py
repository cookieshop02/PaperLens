import PyPDF2
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.
    Returns a single string with pages separated by newlines.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap.
    Better than paragraph splitting for PDFs which have inconsistent formatting.

    Args:
        text: full extracted text
        chunk_size: characters per chunk
        overlap: characters shared between consecutive chunks (context continuity)

    Returns:
        list of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size - overlap (sliding window)
        start += chunk_size - overlap

    return chunks
