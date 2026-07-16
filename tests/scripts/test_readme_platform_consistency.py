from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_readme_platform_launcher_counts_match_default_configs():
    import platform_train

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    config_count = len(platform_train.CONFIGS)
    seed_count = len(platform_train.parse_seeds(platform_train.DEFAULT_SEEDS))
    run_count = config_count * seed_count

    expected = f"{config_count} 个配置 × {seed_count} seed = {run_count}"
    expected_alt = f"{config_count} 个配置 × {seed_count} 个 seed"
    assert expected in readme or expected_alt in readme
    assert "3 配置 × 3 seed = 9" not in readme
    assert "3 配置 × 3 seed 主实验" not in readme
    assert "9 组实验默认" not in readme


def test_readme_does_not_present_legacy_taxonomy_procurement_as_final_path():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "阶段III: 置换检验" not in readme
    assert "阶段V: 预算公平rank采购" not in readme
    assert "关掉taxonomy" not in readme
    assert "procurement_beta_05" not in readme
    assert "dico.taxonomy.enabled=false" not in readme
    assert "dico.coverage.objective=sum" not in readme
    assert "taxonomy_stats.json" not in readme
    assert "covra_trace.json" not in readme
    assert "budget_solver_trace.json" not in readme
    assert "diagnostics.json" in readme
    assert "physical_utility.json" in readme


def test_readme_config_method_whitelist_mentions_adalora():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for method in ("lora", "rs_lora", "adalora", "gora_public", "gora_bm", "dico_cd_da"):
        assert method in readme
    assert "五个\"方法\" config" not in readme


def test_readme_documents_covra_module_scalar_template_formula():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "CovRA-M" in readme
    assert "w_j = 1 / j" in readme
    assert "sum_to_module_energy" in readme
    assert "module_scalar_template" in readme


def test_readme_does_not_overclaim_shell_wrapper_coverage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "scripts/run_*.sh` 与 config 一一对应" not in readme
    assert "所有 `scripts/*.sh" not in readme
    assert "最终方法脚本和保留的 legacy alias" in readme
    assert "run_ablation_no_sign_split.sh" in readme
    assert "run_ablation_covra_independent.sh" in readme
    assert "run_ablation_random_init.sh" in readme
