# Status Matrix

| id | status | requirement | evidence | remaining |
|---|---|---|---|---|
| gpu_e00_pilot | NOT_EXECUTED | Run E00 single-GPU LoRA/CovRA pilot on A800 and record memory/batch/steps. | reports/experiment_matrix.md | Requires access to the target 3xA800 runtime and model/data paths. |
| GoRA-public | IMPLEMENTED_NOT_GPU_RUN | Official GoRA-public baseline wrapper/protocol is verified separately from GoRA-BM. | src/dico/gora.py<br>src/dico/trainer.py<br>configs/dico/gora_public_r8.yaml<br>tests/unit/test_gora_aligned.py<br>tests/unit/test_gora_trainer_integration.py | Direct gradient/allocation/init semantics are CPU/tiny verified against locked commit; A800 E00 and unavailable official final benchmark scripts remain unresolved. |
| EVA | BLOCKED_BY_UNRESOLVED_PROTOCOL | EVA baseline wrapper/protocol is available and budget-audited. | src/dico/baselines.py<br>reports/baseline_status.md | Need official implementation/version and budget matching details. |
| AdaLoRA | IMPLEMENTED_NOT_GPU_RUN | AdaLoRA baseline is implemented under the shared protocol. | src/dico/adalora.py<br>configs/dico/adalora_r8.yaml<br>src/dico/baselines.py<br>tests/unit/test_adalora.py<br>tests/unit/test_adalora_trainer_integration.py | Official commit semantics are covered by CPU/tiny tests; GPU E01 run is still required. |
| MTBench-local executor | IMPLEMENTED_NOT_GPU_RUN | Local MTBench judge actually scores answer sets with locked judge config. | src/dico/mtbench_local.py<br>scripts/mtbench_local_judge.py<br>tests/unit/test_mtbench_local.py | Run scripts/mtbench_local_judge.py on real MTBench answer artifacts with the configured local judge before reporting scores. |
