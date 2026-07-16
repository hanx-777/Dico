from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


SCORE_RE = re.compile(r"\[\[\s*(10(?:\.0)?|[1-9](?:\.\d+)?)\s*\]\]")
FALLBACK_SCORE_RE = re.compile(r"(?:rating|score)\s*[:=]\s*(10(?:\.0)?|[1-9](?:\.\d+)?)", re.IGNORECASE)


def parse_mtbench_score(text: str) -> float | None:
    """Parse a 1--10 MTBench-style judge score from a raw judge response."""

    raw = str(text)
    match = SCORE_RE.search(raw) or FALLBACK_SCORE_RE.search(raw)
    if not match:
        return None
    score = float(match.group(1))
    if score < 1.0 or score > 10.0:
        return None
    return score


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _answer_turns(record: dict[str, Any]) -> list[str]:
    if isinstance(record.get("turns"), list):
        return [str(item) for item in record["turns"]]
    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("turns"), list):
            return [str(item) for item in first["turns"]]
    raise ValueError(f"MTBench answer record has no choices[0].turns or turns field: {record}")


def _question_turns(record: dict[str, Any]) -> list[str]:
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"MTBench question record has no non-empty turns field: {record}")
    return [str(item) for item in turns]


def build_single_answer_judge_prompt(
    *,
    question: dict[str, Any],
    answer: dict[str, Any],
    judge_prompt_version: str,
) -> str:
    """Build a deterministic FastChat-style single-answer judge prompt.

    This is intentionally explicit rather than importing FastChat internals:
    the artifact records the prompt version, and every raw prompt/judgment is
    archived so later paper tables can distinguish this local protocol from
    GoRA's unavailable final benchmark scripts.
    """

    q_turns = _question_turns(question)
    a_turns = _answer_turns(answer)
    if len(a_turns) < len(q_turns):
        a_turns = a_turns + [""] * (len(q_turns) - len(a_turns))
    dialogue = []
    for idx, q_text in enumerate(q_turns):
        dialogue.append(f"[Question turn {idx + 1}]\n{q_text}")
        dialogue.append(f"[Assistant answer turn {idx + 1}]\n{a_turns[idx]}")
    return (
        f"You are a local MTBench judge using prompt protocol {judge_prompt_version}.\n"
        "Rate the assistant answer on a 1 to 10 scale for helpfulness, correctness, "
        "reasoning quality, and instruction following. Reply with a short rationale "
        "and put the final numeric rating in double brackets, e.g. [[7]].\n\n"
        + "\n\n".join(dialogue)
        + "\n\nFinal rating:"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_mtbench_protocol(
    *,
    output_dir: Path | str,
    status: str,
    judge_model: str,
    judge_prompt_version: str,
    conversation_template: str,
    temperature: float,
    seed: int,
    max_retries: int,
    num_questions: int,
    num_answers: int,
    swap_positions: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "conversation_template": conversation_template,
        "temperature": float(temperature),
        "seed": int(seed),
        "max_retries": int(max_retries),
        "num_questions": int(num_questions),
        "num_answers": int(num_answers),
        "swap_positions": bool(swap_positions),
        "score_range": [1, 10],
        "output_files": {
            "protocol": "mtbench_local_protocol.json",
            "judgments": "mtbench_local_judgments.jsonl",
            "metrics": "mtbench_local_metrics.json",
        },
        "note": note
        or (
            "Dry-run only freezes inputs/protocol; executed mode must call a local judge model "
            "and archive raw judgments before any MTBench-local score is reported."
        ),
    }
    _write_json(Path(output_dir) / "mtbench_local_protocol.json", payload)
    return payload


def evaluate_mtbench_local(
    *,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    judge: Callable[[str], str],
    output_dir: Path | str,
    judge_model: str,
    judge_prompt_version: str,
    conversation_template: str,
    temperature: float,
    seed: int,
    max_retries: int,
    swap_positions: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    answer_by_id = {str(row.get("question_id")): row for row in answers}
    judgment_path = output / "mtbench_local_judgments.jsonl"
    scores: list[float] = []
    failed = 0

    with judgment_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            question_id = str(question.get("question_id"))
            if question_id not in answer_by_id:
                raise ValueError(f"Missing MTBench answer for question_id={question_id}")
            answer = answer_by_id[question_id]
            prompt = build_single_answer_judge_prompt(
                question=question,
                answer=answer,
                judge_prompt_version=judge_prompt_version,
            )
            raw_judgment = ""
            score = None
            for attempt in range(max(1, int(max_retries) + 1)):
                raw_judgment = judge(prompt)
                score = parse_mtbench_score(raw_judgment)
                if score is not None:
                    break
            if score is None:
                failed += 1
            else:
                scores.append(float(score))
            row = {
                "question_id": question.get("question_id"),
                "category": question.get("category"),
                "model_id": answer.get("model_id"),
                "metric": "mtbench_local_score",
                "score": score,
                "raw_judgment": raw_judgment,
                "judge_model": judge_model,
                "judge_prompt_version": judge_prompt_version,
                "conversation_template": conversation_template,
                "temperature": float(temperature),
                "seed": int(seed),
                "swap_positions": bool(swap_positions),
                "prompt": prompt,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    average = float(sum(scores) / len(scores)) if scores else 0.0
    metrics = {
        "metric": "mtbench_local_score",
        "mtbench_local_score": average,
        "num_questions": len(questions),
        "num_answers": len(answers),
        "num_scored": len(scores),
        "num_failed_judgments": failed,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
    }
    protocol = write_mtbench_protocol(
        output_dir=output,
        status="EXECUTED",
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
        conversation_template=conversation_template,
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        num_questions=len(questions),
        num_answers=len(answers),
        swap_positions=swap_positions,
        note="Executed MTBench-local scoring with archived prompts and raw judge responses.",
    )
    _write_json(output / "mtbench_local_metrics.json", metrics)
    return {"protocol": protocol, "metrics": metrics}
