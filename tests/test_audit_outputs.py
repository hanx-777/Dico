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


def write_rank_history(path: Path):
    path.write_text(
        "\n".join(
            [
                "step,module_name,active_rank,max_rank,module_score,total_active_rank,total_active_params,target_budget,budget_error_ratio,rank_distance_from_initial,rank_distance_from_preallocation",
                "0,m,4,8,,4,32,32,0.0,0,0",
                "2,m,4,8,1.0,4,32,32,0.0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_experiment(output_dir: Path, name: str, method: str, rank: int):
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True)
    prealloc_methods = {"dico_pre", "dico_predynamic"}
    dynamic_methods = {"dico_dynamic", "dico_predynamic"}
    write_yaml(
        exp_dir / "config_resolved.yaml",
        {
            "experiment_name": name,
            "method": method,
            "rank": rank,
            "training": {"max_steps": 10},
            "data": {"eval_limit": 2},
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
                "eta": 0.98,
                "allow_rank_beyond_selected_evidence": True,
            },
            "dynamic": {"enabled": method in dynamic_methods},
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
            "budget_ratio": 1.0,
            "preallocation_eta": 0.98 if method in prealloc_methods else None,
            "budget_eta_reached": True,
            "budget_interval_pass": True,
            "generic_repair_applied": method not in prealloc_methods,
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
                "eta": 0.98,
                "allow_rank_beyond_selected_evidence": True,
            }
            if method in prealloc_methods
            else None,
        },
    )
    write_json(
        exp_dir / "budget.json",
        {
            "target_budget": 32,
            "actual_budget": 32,
            "budget_ratio": 1.0,
            "budget_interval_pass": True,
            "generic_repair_applied": method not in prealloc_methods,
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
            "eta": 0.98,
            "budget_ratio": 1.0,
            "budget_interval_pass": True,
            "generic_repair_applied": False,
            "module_logs": [{"module_name": "m", "final_rank": rank}],
        }
        if method in prealloc_methods
        else {"rank_allocation": {"m": rank}},
    )
    write_json(exp_dir / "rank_allocation_final.json", {"m": rank})
    write_rank_history(exp_dir / "rank_history.csv")
    (exp_dir / "train_log.jsonl").write_text("", encoding="utf-8")
    (exp_dir / "eval_log.jsonl").write_text("", encoding="utf-8")
    if method in dynamic_methods:
        (exp_dir / "dynamic_adjustments.jsonl").write_text("", encoding="utf-8")
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
        )
    write_json(tmp_path / "summary.json", {})
    (tmp_path / "summary.csv").write_text("experiment,method\n", encoding="utf-8")
    (tmp_path / "summary.md").write_text("| Method |\n", encoding="utf-8")

    report = audit.audit_outputs(tmp_path)

    assert report["status"] in {"pass", "warning"}
    assert report["critical"] == []
    assert report["experiments"]["lora_r4"]["evaluation_protocol"] == "internal_zero_shot"
    assert report["experiments"]["lora_r4"]["eval_scope"] == "2-sample subset"


def test_audit_outputs_checks_multiseed_coverage(tmp_path: Path):
    audit = load_audit_module()
    for seed in [42, 43]:
        for name, meta in audit.EXPECTED_EXPERIMENTS.items():
            make_experiment(
                tmp_path,
                f"{name}__seed{seed}",
                meta["method"],
                meta["rank"],
            )
    (tmp_path / "summary.csv").write_text("experiment,n\n", encoding="utf-8")
    (tmp_path / "summary_per_run.csv").write_text("experiment,seed\n", encoding="utf-8")
    (tmp_path / "summary.md").write_text("| Method |\n", encoding="utf-8")

    report = audit.audit_outputs(tmp_path)

    assert report["multiseed"] is True
    assert report["seed_coverage"]["expected_seeds"] == [42, 43]
    assert report["seed_coverage"]["missing_runs"] == []
    assert report["critical"] == []


def test_audit_outputs_warns_on_budget_error_ratio(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
        )
    budget_path = tmp_path / "lora_r4" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["budget_error_ratio"] = 0.02
    write_json(budget_path, budget)

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "warning"
    assert any("budget_error_ratio" in item for item in report["warnings"])


def test_audit_outputs_accepts_dico_pre_budget_ratio_at_eta(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
        )
    for filename in ["summary.csv", "summary.md"]:
        (tmp_path / filename).write_text("ok\n", encoding="utf-8")
    budget_path = tmp_path / "dico_pre_r4" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget.update(
        {
            "target_budget": 100,
            "actual_budget": 98,
            "budget_ratio": 0.98,
            "budget_error_ratio": 0.02,
            "budget_interval_pass": True,
            "generic_repair_applied": False,
        }
    )
    write_json(budget_path, budget)
    metrics_path = tmp_path / "dico_pre_r4" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "target_budget": 100,
            "actual_budget": 98,
            "budget_ratio": 0.98,
            "budget_error_ratio": 0.02,
            "preallocation_eta": 0.98,
            "budget_eta_reached": True,
            "budget_interval_pass": True,
            "generic_repair_applied": False,
        }
    )
    metrics["preallocation"]["eta"] = 0.98
    write_json(metrics_path, metrics)

    report = audit.audit_outputs(tmp_path)

    assert not any("dico_pre_r4: budget_error_ratio" in item for item in report["warnings"])
    assert report["experiments"]["dico_pre_r4"]["budget_ratio"] == 0.98
    assert report["experiments"]["dico_pre_r4"]["budget_interval_pass"] is True


def test_audit_budget_accepts_lora_eta98_interval():
    audit = load_audit_module()
    critical = []
    warnings = []

    result = audit._audit_budget(
        "lora_r4_eta98",
        "lora",
        {
            "target_budget": 100,
            "actual_budget": 98,
            "budget_ratio": 0.98,
            "budget_error_ratio": -0.02,
        },
        {"budget": {"enforce_target_ratio": 0.98}},
        {},
        critical,
        warnings,
    )

    assert critical == []
    assert warnings == []
    assert result["budget_interval_pass"] is True


def test_audit_warns_when_evidence_relaxation_is_large():
    audit = load_audit_module()
    warnings = []

    result = audit._audit_evidence_relaxation(
        "dico_pre_r8",
        "dico_pre",
        {
            "evidence_relaxation": {
                "selected_evidence_total": 10,
                "final_rank_total": 20,
                "rank_beyond_evidence_total": 8,
                "rank_beyond_evidence_ratio": 0.4,
                "modules_with_beyond": 3,
                "modules_total": 7,
            }
        },
        warnings,
    )

    assert result["evidence_relaxation"]["rank_beyond_evidence_ratio"] == 0.4
    assert any("rank_beyond_evidence_ratio" in item for item in warnings)


def test_audit_outputs_marks_over_budget_as_critical(tmp_path: Path):
    audit = load_audit_module()
    for name, meta in audit.EXPECTED_EXPERIMENTS.items():
        make_experiment(
            tmp_path,
            name,
            meta["method"],
            meta["rank"],
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
        )
    (tmp_path / "lora_r4" / "eval_predictions.jsonl").write_text(
        json.dumps({"question": "only one"}) + "\n",
        encoding="utf-8",
    )

    report = audit.audit_outputs(tmp_path)

    assert report["status"] == "warning"
    assert any("eval_predictions.jsonl has 1 rows but eval_total=2" in item for item in report["warnings"])
