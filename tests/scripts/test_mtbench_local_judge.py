import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mtbench_local_judge_dry_run_writes_protocol_without_scores(tmp_path: Path):
    questions_path = tmp_path / "questions.jsonl"
    answers_path = tmp_path / "answers.jsonl"
    output_dir = tmp_path / "judge_out"
    questions_path.write_text(
        json.dumps({"question_id": 1, "turns": ["Say hi."]}) + "\n",
        encoding="utf-8",
    )
    answers_path.write_text(
        json.dumps({"question_id": 1, "model_id": "covra", "choices": [{"turns": ["Hi!"]}]}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python",
            "scripts/mtbench_local_judge.py",
            "--questions-jsonl",
            str(questions_path),
            "--answers-jsonl",
            str(answers_path),
            "--output-dir",
            str(output_dir),
            "--judge-model",
            "local-70b-placeholder",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    protocol = json.loads((output_dir / "mtbench_local_protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "DRY_RUN_CONFIGURED"
    assert protocol["judge_model"] == "local-70b-placeholder"
    assert protocol["num_questions"] == 1
    assert protocol["num_answers"] == 1
    assert not (output_dir / "mtbench_local_metrics.json").exists()
