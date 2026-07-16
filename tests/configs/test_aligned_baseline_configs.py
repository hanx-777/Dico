from pathlib import Path

from dico.config import load_yaml, validate_known_config_fields


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    config = load_yaml(ROOT / "configs" / "dico" / name)
    validate_known_config_fields(config)
    return config


def test_formal_baseline_configs_are_protocol_aligned():
    lora = _load("lora_r8.yaml")
    rs_lora = _load("rs_lora_r8.yaml")
    adalora = _load("adalora_r8.yaml")
    gora = _load("gora_public_r8.yaml")
    gora_bm = _load("gora_bm_r8.yaml")

    assert lora["lora"]["scaling"] == "alpha_over_r"
    assert rs_lora["method"] == "rs_lora"
    assert rs_lora["lora"]["scaling"] == "alpha_over_sqrt_r"
    assert adalora["training"]["learning_rate"] == 5e-4
    assert adalora["adalora"]["beta1"] == adalora["adalora"]["beta2"] == 0.85
    assert gora["method"] == "gora_public"
    assert gora["experiment_name"] == "gora_public_r8_aligned_sdpa_v4"
    assert gora["calibration"]["num_samples"] == 1024
    assert gora["gora"]["official_commit"] == "4037d4d6ba67ff88de87f90b943ff4e3a3649b67"
    assert gora["gora"]["gradient_collection"] == "official_weight_grad_hook"
    assert gora["gora"]["gradient_offload_device"] == "cpu"
    assert gora["gora"]["clear_gradient_after_offload"] is True
    assert gora_bm["method"] == "gora_bm"
    assert gora_bm["experiment_name"] == "gora_bm_r8_aligned_sdpa_v4"
    assert gora_bm["gora"]["strict_budget_repair"] is True

    for config in (lora, adalora, gora, gora_bm):
        assert config["data"]["shuffle"] is True
        assert config["data"]["dataset_seed"] == 42
        assert config["training"]["sample_exposure_policy"] == "repeat_from_fixed_order_to_max_steps"
        assert config["model"]["attn_implementation"] == "sdpa"
        assert config["runtime"]["require_flash_attention_2"] is False


def test_final_eval_is_batched_and_mid_eval_is_disabled():
    config = _load("lora_r8.yaml")
    assert config["evaluation"]["batch_size"] == 4
    assert config["evaluation"]["compute_loss"] is False
    assert config["evaluation"]["mid_eval_loss_only"]["enabled"] is False
