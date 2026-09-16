"""Fast deterministic smoke tests for the backend CI pipeline."""

from app.agent import _safe_calculate
from app.materials import extract_text, material_chunks, validate_igot_url
from app.openai_responses import OpenAIResponsesProvider


def run():
    assert _safe_calculate("2 + 10")["result"] == 12
    assert _safe_calculate("3 * (4 + 2)")["result"] == 18

    text = "Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")

    provider = OpenAIResponsesProvider()
    assert provider.name == "openai"
    assert callable(provider.generate_material_questions)

    print("AI LearnMate backend smoke tests: PASS")


if __name__ == "__main__":
    run()
