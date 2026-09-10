from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import PurePosixPath
from urllib.parse import urlparse
from zipfile import ZipFile
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


def _clean_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _xml_text(raw: bytes, tags: tuple[str, ...]) -> str:
    try:
        with ZipFile(BytesIO(raw)) as archive:
            texts: list[str] = []
            for name in archive.namelist():
                if not any(marker in name for marker in tags):
                    continue
                try:
                    root = ET.fromstring(archive.read(name))
                except ET.ParseError:
                    continue
                for node in root.iter():
                    if node.tag.endswith("}t") or node.tag.endswith("}a"):
                        if node.text and node.text.strip():
                            texts.append(node.text.strip())
            return " ".join(texts).strip()
    except (OSError, ValueError, KeyError):
        return ""


def extract_text(filename: str, raw: bytes, mime_type: str = "") -> str:
    if len(raw) > MAX_MATERIAL_BYTES:
        raise ValueError("Material is too large. Maximum supported size is 12 MB.")

    suffix = PurePosixPath(filename or "material").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use PDF, TXT, Markdown, CSV, JSON, HTML, DOCX or PPTX.")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support is not installed on the server.") from exc
        reader = PdfReader(BytesIO(raw))
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts)
    elif suffix == ".docx":
        text = _xml_text(raw, ("word/document.xml",))
    elif suffix == ".pptx":
        text = _xml_text(raw, ("ppt/slides/", "ppt/notesSlides/"))
    elif suffix in {".html", ".htm"}:
        text = _clean_html(raw.decode("utf-8", errors="ignore"))
    else:
        text = raw.decode("utf-8", errors="ignore")

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


def retrieve_material_context(text: str, query: str, limit: int = 8, max_chars: int = 10000) -> str:
    chunks = material_chunks(text)
    if not chunks:
        return ""
    words = {word for word in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())}
    scored = []
    for index, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"[a-zA-Z0-9]{3,}", chunk.lower()))
        score = len(words & chunk_words)
        scored.append((score, -index, chunk))
    selected = [chunk for _, _, chunk in sorted(scored, reverse=True)[:limit]]
    selected_text = "\n\n--- MATERIAL CHUNK ---\n\n".join(selected)
    return selected_text[:max_chars]


def validate_igot_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host.endswith("igotkarmayogi.gov.in"):
        raise ValueError("Only official iGOT Karmayogi URLs are accepted.")
    return url.strip()


async def import_igot_resource(url: str) -> tuple[str, str, str]:
    url = validate_igot_url(url)
    parsed = urlparse(url)
    filename = PurePosixPath(parsed.path).name or "igot-resource"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
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
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw.decode("utf-8", errors="ignore"))
    title = _clean_html(title_match.group(1)) if title_match else filename
    title = (title or filename or "iGOT learning resource")[:200]
    return title, text[:1_500_000], content_type or "text/plain"
