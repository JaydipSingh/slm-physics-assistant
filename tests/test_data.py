"""
Lightweight data-integrity tests (no heavy dependencies).

These validate the shipped physics QA benchmark and the 32K tokenizer model
file, so a reviewer can confirm the artifacts the paper relies on are
well-formed without training anything.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_physics_qa_dataset_wellformed():
    """The physics QA benchmark has the documented structure."""
    path = ROOT / "data" / "physics_qa_dataset.json"
    assert path.exists(), "physics_qa_dataset.json is missing"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "questions" in data
    questions = data["questions"]
    assert len(questions) > 0

    for q in questions:
        # Each question must have the fields the evaluator consumes.
        assert "id" in q
        assert "category" in q
        assert "question" in q and q["question"].strip()
        assert "expected_answer" in q and q["expected_answer"].strip()
        assert "distractors" in q and isinstance(q["distractors"], list)
        assert len(q["distractors"]) >= 1
        # The correct answer should not also appear among the distractors.
        assert q["expected_answer"] not in q["distractors"]


def test_question_ids_unique():
    """Question identifiers are unique."""
    path = ROOT / "data" / "physics_qa_dataset.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids)), "duplicate question ids found"


def test_tokenizer_model_wellformed():
    """The shipped 32K tokenizer model file is valid JSON with a vocabulary."""
    path = ROOT / "model_32k.json"
    assert path.exists(), "model_32k.json is missing"
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    vocab = model.get("vocab")
    assert vocab is not None, "tokenizer model has no 'vocab' field"
    assert len(vocab) > 0
