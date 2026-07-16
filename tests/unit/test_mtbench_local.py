import json
from pathlib import Path

from dico.mtbench_local import evaluate_mtbench_local, parse_mtbench_score


def test_parse_mtbench_score_reads_fastchat_style_double_brackets():
    assert parse_mtbench_score("The answer is solid. [[8]]") == 8.0
    assert parse_mtbench_score("Rating: 6.5/10") == 6.5
    assert parse_mtbench_score("No score here") is None


def test_evaluate_mtbench_local_scores_answers_and_writes_artifacts(tmp_path: Path):
    questions = [
        {
            "question_id": 1,
            "category": "reasoning",
            "turns": ["What is 2+2?", "Explain briefly."],
        }
    ]
    answers = [
        {
            "question_id": 1,
            "model_id": "covra",
            "choices": [{"index": 0, "turns": ["4", "Because two pairs make four."]}],
        }
    ]

    prompts_seen: list[str] = []

    def judge(prompt: str) -> str:
        prompts_seen.append(prompt)
        return "The response is correct and concise. [[8.5]]"

    payload = evaluate_mtbench_local(
        questions=questions,
        answers=answers,
        judge=judge,
        output_dir=tmp_path,
        judge_model="local-test-judge",
        judge_prompt_version="fastchat-v0.2.36",
        conversation_template="llama-3",
        temperature=0.0,
        seed=0,
        max_retries=1,
    )

    assert payload["metrics"]["mtbench_local_score"] == 8.5
    assert payload["metrics"]["num_questions"] == 1
    assert prompts_seen and "What is 2+2?" in prompts_seen[0]
    assert "Because two pairs make four." in prompts_seen[0]

    judgment_rows = [
        json.loads(line)
        for line in (tmp_path / "mtbench_local_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert judgment_rows[0]["question_id"] == 1
    assert judgment_rows[0]["score"] == 8.5
    assert judgment_rows[0]["metric"] == "mtbench_local_score"
    assert judgment_rows[0]["judge_model"] == "local-test-judge"

    protocol = json.loads((tmp_path / "mtbench_local_protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "EXECUTED"
    assert protocol["judge_prompt_version"] == "fastchat-v0.2.36"
