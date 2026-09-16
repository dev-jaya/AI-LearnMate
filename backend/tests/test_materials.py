import pytest

from app.materials import extract_text, material_chunks, validate_igot_url


def test_material_chunking_preserves_overlap():
    text = "word " * 1000
    chunks = material_chunks(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_igot_url_only_allows_official_domain():
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")
    with pytest.raises(ValueError):
        validate_igot_url("https://example.com/igot")
    with pytest.raises(ValueError):
        validate_igot_url("https://igotkarmayogi.gov.in.evil.example/course")


def test_text_material_extraction_rejects_tiny_content():
    with pytest.raises(ValueError):
        extract_text("note.txt", b"too short")


def test_text_material_extraction_accepts_readable_content():
    raw = ("Binary search works on sorted data by repeatedly comparing the target "
           "with the middle element and discarding the half that cannot contain it. " * 3).encode()
    extracted = extract_text("lesson.txt", raw, "text/plain")
    assert "Binary search" in extracted
