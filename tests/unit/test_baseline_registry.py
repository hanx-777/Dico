from dico.baselines import (
    BASELINE_STATUS_VALUES,
    get_baseline,
    render_baseline_status_markdown,
    require_baseline_registry_complete,
)


def test_baseline_registry_covers_required_methods_with_explicit_statuses():
    rows = require_baseline_registry_complete()
    methods = {row.method for row in rows}

    assert {
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
    } <= methods
    assert all(row.status in BASELINE_STATUS_VALUES for row in rows)
    required_parameter_metrics = {
        "requires_grad_params",
        "peak_active_params",
        "final_active_params",
        "budget_target",
        "budget_actual",
        "budget_error",
    }
    for row in rows:
        assert required_parameter_metrics <= set(row.parameter_metrics), row.method


def test_gora_public_and_gora_bm_are_distinct_and_not_mislabelled():
    public = get_baseline("gora_public")
    bm = get_baseline("gora_bm")

    assert public.method != bm.method
    assert public.display_name == "GoRA-public"
    assert bm.display_name == "GoRA-BM"
    assert public.is_official_reference is True
    assert bm.is_official_reference is False
    assert "must not be labelled GoRA-public" in bm.protocol_notes


def test_unresolved_external_baselines_are_not_marked_verified():
    for method in ["adalora", "gora_public", "eva"]:
        row = get_baseline(method)
        assert row.status != "IMPLEMENTED_AND_GPU_VERIFIED"
        assert row.unresolved_fields


def test_adalora_has_runnable_formal_triplet_config():
    row = get_baseline("adalora")

    assert row.status == "IMPLEMENTED_NOT_GPU_RUN"
    assert row.runnable_config == "configs/dico/adalora_r8.yaml"
    assert row.config_method == "adalora"
    assert row.is_official_reference is False
    assert "AdaLoRA A/E/B" in row.protocol_notes


def test_baseline_status_markdown_is_machine_auditable_enough_for_reports():
    markdown = render_baseline_status_markdown()

    assert (
        "| method | display_name | status | runnable_config | config_method | allocation_method | "
        "is_official_reference | parameter_metrics | unresolved_fields | protocol_notes |"
    ) in markdown
    assert "| gora_public | GoRA-public | IMPLEMENTED_NOT_GPU_RUN |" in markdown
    assert "| gora_bm | GoRA-BM | IMPLEMENTED_NOT_GPU_RUN | configs/dico/gora_bm_r8.yaml |" in markdown
    assert "| adalora | AdaLoRA | IMPLEMENTED_NOT_GPU_RUN | configs/dico/adalora_r8.yaml |" in markdown
    assert "| gora_public | GoRA-public | IMPLEMENTED_NOT_GPU_RUN | configs/dico/gora_public_r8.yaml |" in markdown
    assert "locked official commit" in markdown
    assert "strict-budget repair" in markdown


def test_covra_registry_marks_reference_default_and_full_allocator_as_experimental():
    reference = get_baseline("covra")
    experimental = get_baseline("covra_full_experimental")

    assert reference.runnable_config == "configs/dico/dico_cd_da_r8.yaml"
    assert reference.allocation_method == "covra_v05"
    assert "taxonomy" in reference.protocol_notes
    assert "procurement" in reference.protocol_notes
    assert experimental.runnable_config == "configs/dico/dico_cd_da_r8_covra_full_experimental.yaml"
    assert experimental.allocation_method == "covra_full"
    assert "experimental" in experimental.protocol_notes.lower()
