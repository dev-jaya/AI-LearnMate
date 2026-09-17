from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse
from zipfile import BadZipFile, ZipFile
import re
import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .db import Base


class Material(Base):
    __tablename__ = "learning_materials"

    id = Column(Integer, primary_key=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    filename = Column(String(255), nullable=False)
    mime_type = Column(String(120), default="application/octet-stream", nullable=False)
    source_type = Column(String(30), default="upload", nullable=False)
    source_url = Column(String(1000), default="", nullable=False)
    extracted_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner")


Index("ix_learning_materials_learner_title", Material.learner_id, Material.title)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".docx", ".pptx"}
MAX_MATERIAL_BYTES = 12 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 24 * 1024 * 1024
MAX_IGOT_REDIRECTS = 3
ALLOWED_IGOT_HOST = "igotkarmayogi.gov.in"


def _clean_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _xml_text(raw: bytes, tags: tuple[str, ...]) -> str:
    try:
        with ZipFile(BytesIO(raw)) as archive:
            total_uncompressed = 0
            texts: list[str] = []
            for info in archive.infolist():
                if not any(marker in info.filename for marker in tags):
                    continue
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValueError("The document contains an archive member that is too large to process safely.")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    raise ValueError("The document expands beyond the safe processing limit.")
                try:
                    root = ET.fromstring(archive.read(info))
                except ET.ParseError:
                    continue
                for node in root.iter():
                    if node.tag.endswith("}t") or node.tag.endswith("}a"):
                        if node.text and node.text.strip():
                            texts.append(node.text.strip())
            return " ".join(texts).strip()
    except ValueError:
        raise
    except (OSError, KeyError, BadZipFile):
        return ""


def extract_text(filename: str, raw: bytes, mime_type: str = "") -> str:
    if len(raw) > MAX_MATERIAL_BYTES:
        raise ValueError("Material is too large. Maximum supported size is 12 MB.")

    suffix = PurePosixPath(filename or "material").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use PDF, TXT, Markdown, CSV, JSON, HTML, DOCX or PPTX.")

    try:
        if suffix == ".pdf":
            if not raw.startswith(b"%PDF-"):
                raise ValueError("The uploaded PDF file is invalid or unreadable.")
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError("PDF support is not installed on the server.") from exc
            reader = PdfReader(BytesIO(raw))
            parts = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(parts)
        elif suffix in {".docx", ".pptx"}:
            if not raw.startswith(b"PK"):
                raise ValueError("The uploaded Office document is invalid or unreadable.")
            text = _xml_text(
                raw,
                ("word/document.xml",)
                if suffix == ".docx"
                else ("ppt/slides/", "ppt/notesSlides/"),
            )
        elif suffix in {".html", ".htm"}:
            text = _clean_html(raw.decode("utf-8", errors="ignore"))
        else:
            text = raw.decode("utf-8", errors="ignore")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read the document: {type(exc).__name__}.") from exc

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        raise ValueError("The material does not contain enough readable text to build a useful learning set.")
    return text


def material_chunks(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            split = normalized.rfind(" ", start, end)
            if split > start + int(size * 0.65):
                end = split
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve_material_context(text: str, query: str, limit: int = 24, max_chars: int = 50000) -> str:
    chunks = material_chunks(text)
    if not chunks:
        return ""
    words = {word for word in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())}
    scored = []
    for index, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"[a-zA-Z0-9]{3,}", chunk.lower()))
        score = len(words & chunk_words)
        scored.append((score, -index, chunk))

    ranked = sorted(scored, reverse=True)
    selected: list[str] = []
    seen_indexes: set[int] = set()
    # Keep representative coverage across the document so document-wide quizzes
    # do not accidentally depend on only the first few pages/paragraphs.
    if len(chunks) <= limit:
        selected = chunks[:]
    else:
        spread = max(1, len(chunks) // min(limit, 12))
        for index in range(0, len(chunks), spread):
            selected.append(chunks[index])
            seen_indexes.add(index)
            if len(selected) >= min(limit // 2, 12):
                break
        for _score, neg_index, chunk in ranked:
            index = -neg_index
            if index in seen_indexes:
                continue
            selected.append(chunk)
            seen_indexes.add(index)
            if len(selected) >= limit:
                break

    selected_text = "\n\n--- MATERIAL CHUNK ---\n\n".join(selected)
    return selected_text[:max_chars]


def validate_igot_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    allowed = host == ALLOWED_IGOT_HOST or host.endswith("." + ALLOWED_IGOT_HOST)
    if parsed.scheme not in {"http", "https"} or not allowed:
        raise ValueError("Only official iGOT Karmayogi URLs are accepted.")
    return url.strip()


async def import_igot_resource(url: str) -> tuple[str, str, str]:
    current_url = validate_igot_url(url)
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for _ in range(MAX_IGOT_REDIRECTS + 1):
            response = await client.get(current_url)
            if response.status_code not in {301, 302, 303, 307, 308}:
                response.raise_for_status()
                break
            location = response.headers.get("location")
            if not location:
                raise ValueError("The iGOT resource returned an invalid redirect.")
            current_url = validate_igot_url(urljoin(current_url, location))
        else:
            raise ValueError("The iGOT resource redirected too many times.")

    parsed = urlparse(current_url)
    filename = PurePosixPath(parsed.path).name or "igot-resource"
    content_type = response.headers.get("content-type", "").lower()
    raw = response.content
    if len(raw) > MAX_MATERIAL_BYTES:
        raise ValueError("The iGOT resource is larger than the 12 MB import limit for this demo.")

    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        text = extract_text(filename if filename.lower().endswith(".pdf") else f"{filename}.pdf", raw, content_type)
    elif "html" in content_type or filename.lower().endswith((".html", ".htm")):
        text = _clean_html(raw.decode("utf-8", errors="ignore"))
    else:
        text = raw.decode("utf-8", errors="ignore")

    if len(text) < 80:
        raise ValueError("The iGOT URL did not expose enough readable public text. Download the official material and upload the file instead.")
    raw_text = raw.decode("utf-8", errors="ignore")
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
    title = _clean_html(title_match.group(1)) if title_match else filename
    title = (title or filename or "iGOT learning resource")[:200]
    return title, text[:1_500_000], content_type or "text/plain"
