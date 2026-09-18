"""Deterministic backend contract tests with real function execution.

These tests do not call Gemini over the network. Live Gemini verification is covered
by the optional integration test that runs only when GEMINI_API_KEY is present.
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.ai_providers import (
    MCQ_SCHEMA,
    SdkGeminiProvider,
    question_similarity,
    validate_question,
)
from app.config import settings
from app.main import adaptive_difficulty_plan, choose_material_coverage_groups
from app.materials import extract_segments, material_chunk_records, material_chunks, validate_igot_url
from app.models import MaterialChunk


def run():
    text = (
        "Binary search works on sorted data by repeatedly checking the middle "
        "element and discarding the impossible half. "
    ) * 12

    segments = extract_segments("lesson.txt", text.encode(), "text/plain")
    assert segments and "Binary search" in segments[0][1]

    records = material_chunk_records(segments, size=100, overlap=20)
    assert len(records) > 1
    assert records[0]["source_ref"] == "document"
    assert records[-1]["chunk_index"] == len(records) - 1

    assert "Binary search" in "".join(material_chunks(text, size=100, overlap=20))
    assert validate_igot_url(
        "https://igotkarmayogi.gov.in/course/123"
    ).startswith("https://igotkarmayogi.gov.in")

    for filename, message in [
        ("lesson.exe", "Unsupported file type"),
        ("empty.txt", "enough readable text"),
    ]:
        try:
            extract_segments(filename, b"tiny", "application/octet-stream")
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid document should be rejected")

    assert settings.gemini_model == "gemini-3.8-flash"

    question = validate_question(
        {
            "question": "Which condition does binary search require?",
            "options": ["Sorted data", "Random data only", "An image", "A server"],
            "correct_answer": "a",
            "explanation": "Binary search relies on ordered data.",
            "difficulty": "easy",
        },
        "Algorithms",
        "Binary Search",
        "Searching",
        "easy",
    )
    assert question and question["answer"] == 0 and question["provider"] == "gemini"

    material_question = validate_question(
        {
            "question": "What must binary search operate on?",
            "options": ["Sorted data", "Only images", "Only servers", "Only databases"],
            "correct_answer": "Sorted data",
            "explanation": "The source states that binary search works on sorted data.",
            "difficulty": "easy",
            "source_evidence": "Binary search works on sorted data",
        },
        "Algorithms",
        "Binary Search",
        "Searching",
        "easy",
        source_text=text,
        require_source_evidence=True,
    )
    assert material_question and material_question["source_evidence"] == "Binary search works on sorted data"

    invalid_evidence = validate_question(
        {
            "question": "Unsupported claim",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "not grounded",
            "difficulty": "easy",
            "source_evidence": "This sentence is not present in the source.",
        },
        "Algorithms",
        "Binary Search",
        "Searching",
        "easy",
        source_text=text,
        require_source_evidence=True,
    )
    assert invalid_evidence is None

    assert question_similarity(
        "Which condition does binary search require?",
        "Which condition does binary-search require?"
    ) >= 0.9
    assert question_similarity("Java arrays", "Python dictionaries") < 0.9

    fake_chunks = [
        MaterialChunk(
            material_id=1,
            chunk_index=i,
            source_ref=f"page {i + 1}",
            content=f"FACT-{i} " + ("document content " * 20),
        )
        for i in range(40)
    ]
    groups = choose_material_coverage_groups(fake_chunks, 10)
    joined = "
".join(groups)
    assert len(groups) >= 2
    assert "FACT-0" in joined
    assert "FACT-39" in joined

    assert adaptive_difficulty_plan(None, 5) == [
        "easy", "easy", "medium", "medium", "hard"
    ]
    assert adaptive_difficulty_plan(40, 5) == [
        "easy", "easy", "easy", "medium", "medium"
    ]
    assert adaptive_difficulty_plan(65, 5) == [
        "easy", "medium", "medium", "hard", "hard"
    ]
    assert adaptive_difficulty_plan(90, 5) == [
        "medium", "medium", "hard", "hard", "hard"
    ]

    provider = SdkGeminiProvider()
    assert provider.name == "gemini"
    assert provider.model == "gemini-3.8-flash"
    assert provider._category(429, "RESOURCE_EXHAUSTED quota") == "quota"
    assert provider._category(401, "API key rejected") == "authentication"
    assert provider._category(404, "model not found") == "model_not_found"
    assert provider._category(500, "server error") == "service_unavailable"

    async def provider_checks():
        captured = {}

        async def fake_generate(input_data, system, response_schema=None, previous_interaction_id=None, json_mode=False, store=False):
            captured["input"] = input_data
            captured["system"] = system
            assert response_schema is None
            assert previous_interaction_id is None
            assert store is False
            return "5"

        provider._generate = fake_generate
        reply = await provider.chat(
            "2+3",
            {
                "learner_name": "Test",
                "conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "2+3"},
                ],
            },
        )
        assert reply == "5"
        types = [item["type"] for item in captured["input"]]
        texts = [
            item["content"][0]["text"]
            for item in captured["input"]
            if item.get("content")
        ]
        assert types == ["user_input", "model_output", "user_input"]
        assert texts.count("2+3") == 1

        async def fake_structured(
            input_data,
            system,
            response_schema=None,
            previous_interaction_id=None,
            json_mode=False,
            store=False,
        ):
            assert response_schema == MCQ_SCHEMA
            assert json_mode is True
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "Which condition does binary search require?",
                            "options": [
                                "Sorted data",
                                "Random data only",
                                "An image",
                                "A database server",
                            ],
                            "correct_answer": "Sorted data",
                            "explanation": "Binary search relies on ordered data.",
                            "difficulty": "easy",
                            "question_type": "conceptual",
                            "subject": "Algorithms",
                            "topic": "Binary Search",
                            "subtopic": "Searching",
                            "source_evidence": "Binary search works on sorted data",
                        }
                    ]
                }
            )

        provider._generate = fake_structured
        questions = await provider.generate_material_questions(
            text,
            "Algorithms",
            "Binary Search",
            "easy",
            1,
            [],
            source_reference="page 1",
        )
        assert len(questions) == 1
        assert questions[0]["provider"] == "gemini"
        assert questions[0]["answer"] == 0

    asyncio.run(provider_checks())
    print("AI LearnMate deterministic backend smoke tests: PASS")


if __name__ == "__main__":
    run()
