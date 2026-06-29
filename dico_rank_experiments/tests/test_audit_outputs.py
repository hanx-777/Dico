import importlib.util
import json
from pathlib import Path

import yaml


def load_audit_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "audit_outputs.py"
    spec = importlib.util.spec_from_file_location("audit_outputs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def write_rank_history(path: Path, predynamic: bool = False):
    path.write_text(
        "\n".join(
            [
                "step,module_name,active_rank,max_rank,module_score,total_active_rank,total_active_params,target_budget,budget_error_ratio,rank_distance_from_initial,rank_distance_from_preallocation",
                f"0,m,4,8,,4,32,32,0.0,0,{0 if predynamic else ''}",
                f"2,m,4,8,1.0,4,32,32,0.0,0,{0 if predynamic else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_experiment(output_dir: Path, name: str, method: str, rank: int, dynamic: bool = False):
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True)
    write_yaml(
        exp_dir / "config_resolved.yaml",
        {
            "experiment_name": name,
            "method": method,
            "rank": rank,
            "training": {"max_steps": 10},
            "data": {"eval_limit": 2},
            "dynamic": {
                "enabled": dynamic,
                "move_ratio": 0.10 if method == "dico_predynamic" else 0.20,
                "update_ratios": [0.2, 0.4, 0.6],
            },
            "evaluation": {
                "compute_accuracy": True,
                "protocol": "internal_zero_shot",
                "prompt_style": "sft_cot_hash",
                "answer_extraction": "strict_then_flexible",
            },
            "preallocation": {
                "aggregation_mode": "weighted_topk",
                "atom_weight_normalization": "none",
                "use_cost_aware_allocation": True,
            },
        },
    )
    write_json(
        exp_dir / "metrics.json",
        {
            "experiment": name,
            "method": method,
            "rank": rank,
            "target_budget": 32,
            "actual_budget": 32,
            "budget_error_ratio": 0.0,
            "evaluation_protocol": "internal_zero_shot",
            "evaluation_prompt_style": "sft_cot_hash",
            "answer_extraction": "strict_then_flexible",
            "eval_sample_count": 2,
            "final_eval_accuracy": 0.25,
            "final_exact_match": 0.25,
            "eval_correct": 1,
            "eval_total": 2,
            "preallocation": {
                "aggregation_mode": "weighted_topk",
                "atom_weight_normalization": "none",
                "use_cost_aware_allocation": True,
                "atom_mode": "module_proxy",
                "atom_mode_limitation": "module_proxy limitation",
            }
            if method in {"dico_pre", "dico_predynamic"}
            else None,
        },
    )
    write_json(
        exp_dir / "budget.json",
        {
            "target_budget": 32,
            "actual_budget": 32,
            "budget_error_ratio": 0.0,
            "warning": None,
        },
    )
    write_json(
        exp_dir / "rank_allocation_initial.json",
        {
            "rank_allocation": {"m": rank},
            "aggregation_mode": "weighted_topk",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
            "atom_mode": "module_proxy",
            "atom_mode_limitation": "module_proxy limitation",
            "module_logs": [{"module_name": "m", "final_rank": rank}],
        }
        if method in {"dico_pre", "dico_predynamic"}
        else {"rank_allocation": {"m": rank}},
    )
    write_json(exp_dir / "rank_allocation_final.json", {"m": rank})
    write_rank_history(exp_dir / "rank_history.csv", predynamic=(method == "dico_predynamic"))
    (exp_dir / "train_log.jsonl").write_text("", encoding="utf-8")
    (exp_dir / "eval_log.jsonl").write_text("", encoding="utf-8")
    (exp_dir / "eval_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"question": "q1", "correct": True}),
                json.dumps({"question": "q2", "correct": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if dynamic:
        (exp_dir / "dynamic_adjustments.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"step": 2, "rank_distance_from_preallocation": 0 if method == "dico_predynamic" else None}),
                    json.dumps({"step": 4, "rank_distance_from_preallocation": 0 if method == "dico_predynamic" else None}),
                    json.dumps({"step": 6, "rank_distance_from_preallocation": 0 if method == "dico_predynamic" else None}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def test_audit_outputs_reports_missing_experiments_as_critical(tmp_path: Path):
    audit = load_audit_module()
    make_experiment(tmp_path, "lora_r4", "lora", 4)

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "fail"
    assert any("Missing experiment directory" in item for item in report["critical"])


def test_audit_outputs_accepts_complete_mock_outputs(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
            dynamic=meta["method"] in {"dico_dynamic", "dico_predynamic"},
        )
    write_json(tmp_path / "summary.json", {})
    (tmp_path / "summary.csv").write_text("experiment,method\n", encoding="utf-8")
    (tmp_path / "summary.md").write_text("| Method |\n", encoding="utf-8")

    report = audit.audit_outputs(tmp_path)

    assert report["status"] in {"pass", "warning"}
    assert report["critical"] == []
    assert report["experiments"]["dico_predynamic_r4"]["method"] == "dico_predynamic"
    assert report["experiments"]["lora_r4"]["evaluation_protocol"] == "internal_zero_shot"
    assert report["experiments"]["lora_r4"]["eval_scope"] == "2-sample subset"


def test_audit_outputs_warns_on_budget_error_ratio(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
            dynamic=meta["method"] in {"dico_dynamic", "dico_predynamic"},
        )
    budget_path = tmp_path / "lora_r4" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["budget_error_ratio"] = 0.02
    write_json(budget_path, budget)

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "warning"
    assert any("budget_error_ratio" in item for item in report["warnings"])


def test_audit_outputs_marks_over_budget_as_critical(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
            dynamic=meta["method"] in {"dico_dynamic", "dico_predynamic"},
        )
    budget_path = tmp_path / "lora_r4" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["actual_budget"] = 40
    budget["target_budget"] = 32
    budget["over_budget"] = True
    write_json(budget_path, budget)

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "fail"
    assert any("actual_budget exceeds target_budget" in item for item in report["critical"])


def test_audit_outputs_warns_on_missing_evaluation_protocol(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
            dynamic=meta["method"] in {"dico_dynamic", "dico_predynamic"},
        )
    metrics_path = tmp_path / "lora_r4" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("evaluation_protocol")
    write_json(metrics_path, metrics)

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "warning"
    assert any("evaluation_protocol" in item for item in report["warnings"])


def test_audit_outputs_warns_when_prediction_count_mismatches_eval_total(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
            dynamic=meta["method"] in {"dico_dynamic", "dico_predynamic"},
        )
    (tmp_path / "lora_r4" / "eval_predictions.jsonl").write_text(
        json.dumps({"question": "only one"}) + "\n",
        encoding="utf-8",
    )

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "warning"
    assert any("eval_predictions.jsonl has 1 rows but eval_total=2" in item for item in report["warnings"])
