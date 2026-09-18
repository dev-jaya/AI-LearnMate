#!/usr/bin/env python3
"""Production smoke tests for the deployed AI LearnMate application."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BACKEND = os.environ.get("BACKEND_URL", "https://ai-learnmate-backend.onrender.com").rstrip("/")
FRONTEND = os.environ.get("FRONTEND_URL", "https://ai-learnmate-frontend.onrender.com").rstrip("/")
EXPECTED_VERSION = os.environ.get("EXPECTED_VERSION", "1.5.0")

def http(method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 90):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {url} -> HTTP {exc.code}: {detail[:800]}") from exc

def json_request(method: str, path: str, payload=None):
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    status, _, raw = http(method, BACKEND + path, body, headers)
    assert status == 200, f"{method} {path}: expected 200, got {status}"
    return json.loads(raw.decode())

def multipart_upload(path: str, field: str, file_path: Path):
    boundary = "----AILearnMateSmoke" + uuid.uuid4().hex
    file_bytes = file_path.read_bytes()
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{file_path.name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    body = prefix + file_bytes + suffix
    headers = {"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    status, _, raw = http("POST", BACKEND + path, body, headers, timeout=120)
    assert status == 200, f"upload failed with HTTP {status}"
    return json.loads(raw.decode())

def wait_for_deploy():
    deadline = time.time() + 600
    last = None
    while time.time() < deadline:
        try:
            data = json_request("GET", "/health")
            last = data
            if data.get("version") == EXPECTED_VERSION:
                return data
        except Exception as exc:
            last = str(exc)
        time.sleep(10)
    raise AssertionError(f"new backend version {EXPECTED_VERSION} did not become reachable: {last}")

def main():
    print("1/13 waiting for deployed backend version...")
    print("PASS", wait_for_deploy())

    print("2/13 checking AI configuration...")
    ai = json_request("GET", "/health/ai")
    assert ai.get("configured") is True, ai
    assert ai.get("model") == "gemini-2.5-flash", ai
    print("PASS", {"configured": ai["configured"], "model": ai["model"]})

    print("3/13 checking frontend...")
    status, _, raw = http("GET", FRONTEND + "/")
    html = raw.decode("utf-8", errors="replace")
    assert status == 200 and 'id="root"' in html
    print("PASS frontend HTTP 200")

    print("4/13 creating smoke learner...")
    learner = json_request("POST", "/api/learners", {"name": "Production Smoke"})
    learner_id = learner["id"]
    print("PASS learner", learner_id)

    print("5/13 real Gemini chat: Hi")
    first = json_request("POST", "/api/chat", {"learner_id": learner_id, "message": "Hi", "topic": "Java"})
    reply1 = first["message"]["content"]
    assert reply1.strip() and "could not reach the AI assistant" not in reply1.lower()
    conversation_id = first["conversation_id"]
    print("PASS", reply1[:180].replace("\n", " "))

    print("6/13 real Gemini chat: 2 + 3")
    second = json_request("POST", "/api/chat", {"learner_id": learner_id, "message": "What is 2 + 3?", "topic": "Mathematics", "conversation_id": conversation_id})
    reply2 = second["message"]["content"]
    assert re.search(r"\b5\b", reply2), reply2
    print("PASS", reply2[:180].replace("\n", " "))

    print("7/13 stateful follow-up")
    third = json_request("POST", "/api/chat", {"learner_id": learner_id, "message": "What was the result in the previous calculation?", "topic": "Mathematics", "conversation_id": conversation_id})
    reply3 = third["message"]["content"]
    assert re.search(r"\b5\b", reply3), reply3
    print("PASS", reply3[:180].replace("\n", " "))

    print("8/13 Java quiz")
    quiz = json_request("POST", "/api/assessment/start", {"learner_id": learner_id, "topic": "Java", "subject": "Java", "difficulty": "medium", "count": 3})
    assert len(quiz["questions"]) == 3
    for question in quiz["questions"]:
        assert len(question["options"]) == 4
        assert len({item.strip().lower() for item in question["options"]}) == 4
        assert question["subject"] == "Java"
        assert 0 <= question["answer"] < 4
    print("PASS Java quiz")

    print("9/13 quiz submission/scoring")
    answers = [question["answer"] for question in quiz["questions"]]
    result = json_request("POST", "/api/assessment/submit", {"quiz_id": quiz["id"], "learner_id": learner_id, "topic": "Java", "difficulty": "medium", "answers": answers})
    assert result["score"] == 100, result
    print("PASS score 100")

    print("10/13 PDF upload")
    fixture = Path("backend/tests/fixtures/multi_section_test.pdf")
    assert fixture.exists(), fixture
    material = multipart_upload(f"/api/materials/upload?learner_id={learner_id}", "file", fixture)
    assert material["chunks"] >= 3, material
    material_id = material["id"]
    print("PASS material", material_id, "chunks", material["chunks"])

    print("11/13 material-grounded quiz")
    mquiz = json_request("POST", f"/api/materials/{material_id}/quiz", {"learner_id": learner_id, "topic": "Document", "subject": "Java", "difficulty": "easy", "count": 3})
    assert len(mquiz["questions"]) == 3
    for question in mquiz["questions"]:
        assert len(question["options"]) == 4 and question["fingerprint"]
    print("PASS material quiz")

    print("12/13 learner analytics and recommendations")
    dashboard = json_request("GET", f"/api/dashboard/{learner_id}")
    progress = json_request("GET", f"/api/learners/{learner_id}/progress")
    analytics = json_request("GET", f"/api/learners/{learner_id}/analytics")
    recommendations = json_request("GET", f"/api/learners/{learner_id}/recommendations")
    assert dashboard["attempts"] >= 1 and progress["mastery"] and analytics["attempts"] >= 1
    assert "recommendations" in recommendations
    print("PASS analytics")

    print("13/13 invalid-input handling")
    bad = urllib.request.Request(
        BACKEND + "/api/chat",
        data=json.dumps({"learner_id": learner_id, "message": ""}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(bad, timeout=30)
    except urllib.error.HTTPError as exc:
        assert exc.code == 422, exc.code
    else:
        raise AssertionError("empty chat message unexpectedly succeeded")
    print("PASS HTTP 422 for invalid input")
    print("PRODUCTION SMOKE PASSED")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PRODUCTION SMOKE FAILED:", exc)
        sys.exit(1)
