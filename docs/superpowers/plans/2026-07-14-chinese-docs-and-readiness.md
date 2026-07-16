# Chinese Docs and Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the user-facing README, experiment description, launch instructions, and acceptance docs clear in Chinese while preserving the current local/GPU verification boundary.

**Architecture:** Keep configs and training code unchanged. Add a documentation consistency test, then update `README.md`, `launch_covra.py` top comments, and `docs/audits/*.md` so model/data, environment, paths, launch, evaluation, FAQ, and remaining server-only risks are explicit.

**Tech Stack:** Markdown, Python standard-library text tests, pytest.

---

### Task 1: Add documentation consistency checks

**Files:**
- Create: `tests/scripts/test_chinese_documentation.py`
- Modify: `README.md`
- Modify: `launch_covra.py`
- Modify: `docs/audits/covra_local_acceptance_audit.md`
- Modify: `docs/audits/covra_server_gpu_acceptance_checklist.md`

- [x] **Step 1: Write failing tests**

```python
def test_user_facing_docs_are_chinese_and_state_current_model_dataset():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "当前默认实验模型与数据集" in readme
    assert "Meta-Llama-3.1-8B-Base" in readme
```

- [x] **Step 2: Run the tests to verify failure**

Run: `pytest tests/scripts/test_chinese_documentation.py -q`

Expected: FAIL before the docs are updated.

- [ ] **Step 3: Update Chinese documentation**

Rewrite the affected user-facing sections/documents in Chinese; do not edit method configs or training behavior.

- [ ] **Step 4: Run documentation tests**

Run: `pytest tests/scripts/test_chinese_documentation.py tests/scripts/test_readme_platform_consistency.py tests/scripts/test_launch_covra.py -q`

Expected: PASS.

- [ ] **Step 5: Run acceptance tests and refresh reports**

Run `pytest -q`, then refresh static/e00/preflight/final reports.
