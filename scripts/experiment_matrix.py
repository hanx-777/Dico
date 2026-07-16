#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.baselines import get_baseline


@dataclass(frozen=True)
class ExperimentCommand:
    id: str
    title: str
    priority: str
    status: str
    purpose: str
    commands: list[str] = field(default_factory=list)
    blocked_items: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _single_run(config: str, experiment_name: str, seed: int = 42, max_steps: int | None = None) -> str:
    parts = [
        "CUDA_VISIBLE_DEVICES=0",
        "python scripts/run_experiment.py",
        f"--config {config}",
        f"--override experiment_name={experiment_name}_seed{seed}",
        f"--override seed={seed}",
        f"--override calibration.seed={seed}",
        f"--override preallocation.sketch_seed={seed}",
        "--override training.batch_size=4",
        "--override training.gradient_accumulation_steps=16",
    ]
    if max_steps is not None:
        parts.append(f"--override training.max_steps={int(max_steps)}")
    return " ".join(parts)


def _platform(configs: list[str], output_dir: str, extra: str = "") -> str:
    config_args = " ".join(f"--config {path}" for path in configs)
    suffix = f" {extra.strip()}" if extra.strip() else ""
    return (
        "python scripts/platform_train.py "
        f"{config_args} "
        "--num-gpus 3 --child-num-processes 1 --seeds 42,43,44 "
        "--batch-size 4 --grad-accum 16 "
        f"--output-dir {output_dir}{suffix}"
    )


def build_matrix() -> list[ExperimentCommand]:
    adalora = get_baseline("adalora")
    eva = get_baseline("eva")
    gora_public = get_baseline("gora_public")
    return [
        ExperimentCommand(
            id="E00",
            title="single-GPU scheduling pilot",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Verify single-GPU global batch 64, logging, parameter counts, and CovRA allocation before real GPU runs.",
            commands=[
                _single_run("configs/dico/lora_r8.yaml", "e00_lora_pilot", max_steps=1),
                _single_run("configs/dico/dico_cd_da_r8.yaml", "e00_covra_pilot", max_steps=1),
                (
                    "NUM_GPUS=3 bash scripts/run_ddp.sh configs/dico/dico_cd_da_r8.yaml "
                    "--override experiment_name=e00_covra_ddp_fallback_seed42 "
                    "--override seed=42 --override calibration.seed=42 "
                    "--override training.max_steps=1"
                ),
            ],
            notes=[
                "Default protocol is single GPU per seed: batch 4 × grad_accum 16 = global batch 64.",
                "DDP command is fallback only: 3 × per-device batch 3 × grad_accum 7 = global batch 63 by script default.",
            ],
        ),
        ExperimentCommand(
            id="E01",
            title="Llama3 r8 main results",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Main fair comparison for implemented methods under GoRA-aligned single-GPU global batch 64 scheduling.",
            commands=[
                _platform(
                    [
                        "configs/dico/lora_r8.yaml",
                        "configs/dico/adalora_r8.yaml",
                        "configs/dico/gora_public_r8.yaml",
                        "configs/dico/dico_cd_da_r8.yaml",
                    ],
                    "outputs/e01_llama3_r8_aligned_sdpa_v4",
                )
            ],
            blocked_items=["eva"],
            notes=[
                "GoRA-public uses the locked official commit behavior and preserves its realized method-faithful budget.",
                "EVA is outside the current implementation scope and remains protocol-blocked.",
                "CovRA-I and CovRA-M are mechanism controls and are scheduled separately in E04/E05, not mixed into the E01 main table.",
                (
                    f"adalora status={adalora.status} using configs/dico/adalora_r8.yaml; "
                    f"eva status={eva.status}; gora_public status={gora_public.status}."
                ),
            ],
        ),
        ExperimentCommand(
            id="E02",
            title="strict-budget audit",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Audit LoRA, GoRA-BM, and CovRA actual trainable parameter budgets; rerun only if E01 logs are not strict-budget.",
            commands=[
                _platform(
                    [
                        "configs/dico/lora_r8.yaml",
                        "configs/dico/gora_bm_r8.yaml",
                        "configs/dico/dico_cd_da_r8.yaml",
                    ],
                    "outputs/e02_llama3_r8_strict_budget_sdpa_v4",
                )
            ],
            notes=["Use E01 artifacts if their budget error is already zero or within the registered strict rule."],
        ),
        ExperimentCommand(
            id="E03",
            title="GoRA configuration conflict sensitivity",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Appendix-only sensitivity for GoRA N/gamma/warmup conflicts.",
            commands=[
                _single_run("configs/dico/gora_paper_gamma08.yaml", "e03_gora_gamma08", max_steps=1563),
                _single_run("configs/dico/gora_paper_n64.yaml", "e03_gora_n64", max_steps=1563),
            ],
            notes=["Do not use E03 to select the main GoRA-public setting from test results."],
        ),
        ExperimentCommand(
            id="E04",
            title="conditional valuation",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Compare CovRA against CovRA-I with one unique difference: conditional residual value vs fixed independent energy.",
            commands=[
                _platform(
                    ["configs/dico/dico_cd_da_r8.yaml", "configs/ablations/covra_independent.yaml"],
                    "outputs/e04_covra_vs_covra_i",
                )
            ],
        ),
        ExperimentCommand(
            id="E05",
            title="direction-level structure",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Compare CovRA-I against CovRA-M; response-overlap analysis reuses artifacts and adds no training.",
            commands=[
                _platform(
                    ["configs/ablations/covra_independent.yaml", "configs/ablations/covra_module_scalar.yaml"],
                    "outputs/e05_covra_i_vs_covra_m",
                )
            ],
        ),
        ExperimentCommand(
            id="E06",
            title="rank/init separation",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Separate rank allocation benefit from direction-subspace initialization benefit.",
            commands=[
                _platform(
                    [
                        "configs/dico/lora_r8.yaml",
                        "configs/ablations/uniform_rank_covra_init.yaml",
                        "configs/ablations/covra_rank_random_init.yaml",
                        "configs/dico/dico_cd_da_r8.yaml",
                    ],
                    "outputs/e06_rank_init_separation",
                )
            ],
        ),
        ExperimentCommand(
            id="E07",
            title="global/grouped sketch ablation",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Ablate grouped response statistics without changing training, budget, or evaluation interface.",
            commands=[
                _platform(
                    ["configs/ablations/global_only.yaml", "configs/ablations/grouped_only.yaml"],
                    "outputs/e07_global_grouped",
                )
            ],
        ),
        ExperimentCommand(
            id="E08",
            title="supporting component ablations",
            priority="must",
            status="READY_DRY_RUN",
            purpose="Test sign split, type scaling, log compression, and proportional rounding as one-factor changes.",
            commands=[
                _platform(
                    [
                        "configs/ablations/no_sign_split.yaml",
                        "configs/ablations/no_type_scaling.yaml",
                        "configs/ablations/no_log_compression.yaml",
                        "configs/ablations/proportional_rounding.yaml",
                    ],
                    "outputs/e08_component_ablations",
                )
            ],
        ),
        ExperimentCommand(
            id="E09",
            title="efficiency and budget report",
            priority="must",
            status="PARTIALLY_READY",
            purpose="Collect calibration/allocation/training timing, memory, tokens/s, and budget fields from E01/E04-E08 artifacts.",
            commands=[
                "python scripts/baseline_status.py --json-output reports/baseline_status.json --markdown-output reports/baseline_status.md",
                "python scripts/experiment_matrix.py --json-output reports/experiment_matrix.json --markdown-output reports/experiment_matrix.md",
                "python scripts/audit_status.py --output-dir reports/audit",
                "python scripts/e00_readiness.py --json-output reports/e00_readiness.json --markdown-output reports/e00_readiness.md --require-gpu-count 0",
                "python scripts/protocol_preflight.py --json-output reports/protocol_preflight.json --markdown-output reports/protocol_preflight.md",
                "python scripts/static_acceptance.py --json-output reports/static_acceptance.json --markdown-output reports/static_acceptance.md",
                "python scripts/directory_structure.py --json-output reports/directory_structure.json --markdown-output reports/directory_structure.md",
                "python scripts/changed_files_report.py --json-output reports/changed_files.json --markdown-output reports/changed_files.md",
                (
                    "python scripts/validate_run_manifest.py --output-dir outputs/e01_llama3_r8_main "
                    "--json-output reports/run_manifest_validation.json --markdown-output reports/run_manifest_validation.md"
                ),
                (
                    "python scripts/collect_run_manifests.py --output-dir outputs/e01_llama3_r8_main "
                    "--json-output reports/run_manifest_summary.json --markdown-output reports/run_manifest_summary.md"
                ),
                (
                    "python scripts/mtbench_local_judge.py --questions-jsonl data/mtbench/questions.jsonl "
                    "--answers-jsonl outputs/mtbench/model_answer/covra.jsonl "
                    "--output-dir outputs/mtbench/local_judge/covra --dry-run"
                ),
                (
                    "python scripts/final_delivery_report.py --json-output reports/final_delivery.json "
                    "--markdown-output reports/final_delivery.md --test-result '<paste latest pytest result>'"
                ),
            ],
            blocked_items=["mtbench_local_70b_judge_execution:NOT_EXECUTED"],
        ),
        ExperimentCommand(
            id="E10",
            title="recommended extensions",
            priority="recommended",
            status="PARTIALLY_READY",
            purpose="r32, Llama2, EVA, LoRA-GA/rsLoRA/LoRA+ and code-task mechanism extensions after must-run claims are supported.",
            commands=[
                _single_run("configs/dico/dico_cd_da_r32_pilot.yaml", "e10_covra_r32_pilot", max_steps=1),
                _platform(
                    ["configs/dico/dico_cd_da_r32_pilot.yaml"],
                    "outputs/e10_covra_r32_pilot",
                    "--dry-run",
                ),
            ],
            blocked_items=[
                "r32 formal training requires successful E10 GPU pilot",
                "Llama2 protocol/revisions unresolved",
                "eva",
                "LoRA-GA/rsLoRA/LoRA+ wrappers not implemented",
            ],
        ),
    ]


def render_markdown(rows: list[ExperimentCommand]) -> str:
    lines = [
        "| id | title | priority | status | purpose | commands | blocked_items | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        commands = "<br>".join(f"`{command}`" for command in row.commands) if row.commands else "-"
        blocked = ", ".join(row.blocked_items) if row.blocked_items else "-"
        notes = "<br>".join(row.notes) if row.notes else "-"
        lines.append(
            f"| {row.id} | {row.title} | {row.priority} | {row.status} | "
            f"{row.purpose} | {commands} | {blocked} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write E00-E10 experiment command/status matrix.")
    parser.add_argument("--json-output", default="reports/experiment_matrix.json")
    parser.add_argument("--markdown-output", default="reports/experiment_matrix.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_matrix()
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"experiments": [asdict(row) for row in rows]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"[experiment_matrix] wrote {json_path}")
    print(f"[experiment_matrix] wrote {md_path}")


if __name__ == "__main__":
    main()
