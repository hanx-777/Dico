from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


BASELINE_STATUS_VALUES = {
    "IMPLEMENTED_AND_CPU_VERIFIED",
    "IMPLEMENTED_NOT_GPU_RUN",
    "BLOCKED_BY_UNRESOLVED_PROTOCOL",
    "NOT_IMPLEMENTED",
}

REQUIRED_BASELINES = (
    "uniform_lora",
    "adalora",
    "gora_public",
    "gora_bm",
    "eva",
    "covra",
    "covra_independent",
    "covra_module_scalar",
    "uniform_rank_covra_init",
    "covra_rank_random_init",
)

REQUIRED_PARAMETER_METRICS = (
    "requires_grad_params",
    "peak_active_params",
    "final_active_params",
    "budget_target",
    "budget_actual",
    "budget_error",
)


@dataclass(frozen=True)
class BaselineSpec:
    method: str
    display_name: str
    status: str
    runnable_config: str | None
    config_method: str | None
    allocation_method: str | None
    is_official_reference: bool
    protocol_notes: str
    parameter_metrics: tuple[str, ...] = REQUIRED_PARAMETER_METRICS
    unresolved_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parameter_metrics"] = list(self.parameter_metrics)
        payload["unresolved_fields"] = list(self.unresolved_fields)
        return payload


_REGISTRY: dict[str, BaselineSpec] = {
    "uniform_lora": BaselineSpec(
        method="uniform_lora",
        display_name="Uniform LoRA",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/lora_r8.yaml",
        config_method="lora",
        allocation_method=None,
        is_official_reference=False,
        protocol_notes="Uniform rank-r LoRA baseline using the shared trainer and budget logger; GPU E00 not run here.",
    ),
    "adalora": BaselineSpec(
        method="adalora",
        display_name="AdaLoRA",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/adalora_r8.yaml",
        config_method="adalora",
        allocation_method=None,
        is_official_reference=False,
        protocol_notes=(
            "Formal in-repository AdaLoRA A/E/B implementation aligned to commit "
            "d10f5ebee16c478fa2f41a44a237b38e8c9b0338 with pre-clip EMA importance, "
            "global no-floor pruning, recoverable E gradients, cubic schedule, unsquared "
            "orthogonal regularization, and physical/peak/final reporting."
        ),
        unresolved_fields=(
            "GPU validation under the one-epoch GoRA protocol",
        ),
    ),
    "gora_public": BaselineSpec(
        method="gora_public",
        display_name="GoRA-public",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/gora_public_r8.yaml",
        config_method="gora_public",
        allocation_method="gora_public",
        is_official_reference=True,
        protocol_notes=(
            "Reimplemented in the shared trainer from locked official commit "
            "4037d4d6ba67ff88de87f90b943ff4e3a3649b67; direct target-weight gradient hooks "
            "perform one backward per calibration batch and preserve the method-faithful budget."
        ),
        unresolved_fields=(
            "official final benchmark scripts",
            "target-GPU validation",
        ),
    ),
    "gora_bm": BaselineSpec(
        method="gora_bm",
        display_name="GoRA-BM",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/gora_bm_r8.yaml",
        config_method="gora_bm",
        allocation_method="gora_bm",
        is_official_reference=False,
        protocol_notes=(
            "GoRA-public plus one controlled strict-budget repair; it must not be labelled GoRA-public."
        ),
        unresolved_fields=(
            "target-GPU validation",
        ),
    ),
    "eva": BaselineSpec(
        method="eva",
        display_name="EVA",
        status="BLOCKED_BY_UNRESOLVED_PROTOCOL",
        runnable_config=None,
        config_method=None,
        allocation_method=None,
        is_official_reference=False,
        protocol_notes="EVA is required as a main-plan baseline but has no verified wrapper in this repository yet.",
        unresolved_fields=(
            "official implementation/version",
            "activation dataset/preprocessing parity",
            "actual parameter-budget matching rule",
            "subspace initialization compatibility with this trainer",
        ),
    ),
    "covra": BaselineSpec(
        method="covra",
        display_name="CovRA",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/dico_cd_da_r8.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_v05",
        is_official_reference=False,
        protocol_notes=(
            "Default reference-aligned CovRA path with taxonomy, virtual candidates, NSW coverage, "
            "physical utility, quota-aware procurement, and direction-anchored init."
        ),
    ),
    "covra_full_experimental": BaselineSpec(
        method="covra_full_experimental",
        display_name="CovRA-full (experimental)",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/dico/dico_cd_da_r8_covra_full_experimental.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_full",
        is_official_reference=False,
        protocol_notes=(
            "Experimental conditional-coverage plus DP allocator retained for compatibility; "
            "it is not the default reference CovRA protocol."
        ),
    ),
    "covra_independent": BaselineSpec(
        method="covra_independent",
        display_name="CovRA-I",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/ablations/covra_independent.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_independent",
        is_official_reference=False,
        protocol_notes="Controlled CovRA-I ablation: fixed independent candidate response energy replaces conditional residual gains.",
    ),
    "covra_module_scalar": BaselineSpec(
        method="covra_module_scalar",
        display_name="CovRA-M",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/ablations/covra_module_scalar.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_module_scalar",
        is_official_reference=False,
        protocol_notes="Controlled CovRA-M ablation: direction-level structure is collapsed to module energy plus a fixed rank template.",
    ),
    "uniform_rank_covra_init": BaselineSpec(
        method="uniform_rank_covra_init",
        display_name="Uniform-rank + CovRA-init",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/ablations/uniform_rank_covra_init.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_full",
        is_official_reference=False,
        protocol_notes=(
            "Rank/init separation ablation: run CovRA candidate extraction and direction bank, "
            "then override final ranks to the uniform reference rank."
        ),
    ),
    "covra_rank_random_init": BaselineSpec(
        method="covra_rank_random_init",
        display_name="CovRA-rank + Random-init",
        status="IMPLEMENTED_NOT_GPU_RUN",
        runnable_config="configs/ablations/covra_rank_random_init.yaml",
        config_method="dico_cd_da",
        allocation_method="covra_full",
        is_official_reference=False,
        protocol_notes=(
            "Rank/init separation ablation: keep CovRA rank allocation but use the default random LoRA initialization."
        ),
    ),
}


def list_baselines() -> list[BaselineSpec]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def get_baseline(method: str) -> BaselineSpec:
    try:
        return _REGISTRY[str(method)]
    except KeyError as exc:
        raise KeyError(f"Unknown baseline method {method!r}; known={sorted(_REGISTRY)}") from exc


def baseline_status_matrix() -> list[dict[str, object]]:
    return [row.to_dict() for row in list_baselines()]


def require_baseline_registry_complete(required: Sequence[str] = REQUIRED_BASELINES) -> list[BaselineSpec]:
    missing = [name for name in required if name not in _REGISTRY]
    if missing:
        raise AssertionError(f"Baseline registry is missing required methods: {missing}")
    invalid = [row.method for row in _REGISTRY.values() if row.status not in BASELINE_STATUS_VALUES]
    if invalid:
        raise AssertionError(f"Baseline registry contains invalid statuses for: {invalid}")
    return [_REGISTRY[name] for name in required]


def render_baseline_status_markdown(rows: Sequence[BaselineSpec] | None = None) -> str:
    selected = list(rows) if rows is not None else list_baselines()
    lines = [
        (
            "| method | display_name | status | runnable_config | config_method | allocation_method | "
            "is_official_reference | parameter_metrics | unresolved_fields | protocol_notes |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        parameter_metrics = ", ".join(row.parameter_metrics) if row.parameter_metrics else "-"
        unresolved = ", ".join(row.unresolved_fields) if row.unresolved_fields else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    row.display_name,
                    row.status,
                    row.runnable_config or "-",
                    row.config_method or "-",
                    row.allocation_method or "-",
                    str(bool(row.is_official_reference)).lower(),
                    parameter_metrics,
                    unresolved,
                    row.protocol_notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def baseline_by_config_method() -> Mapping[tuple[str | None, str | None], BaselineSpec]:
    return {
        (row.config_method, row.allocation_method): row
        for row in _REGISTRY.values()
        if row.config_method is not None
    }
