# Launch CovRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-argument `launch_covra.py` platform launcher that starts CovRA training through the existing official project entrypoint without changing method logic or experiment hyperparameters.

**Architecture:** `launch_covra.py` is a thin, top-configured wrapper around `scripts/platform_train.py`. It validates configured paths, creates output/log directories, sets runtime environment variables, streams child output to both console and log, and exits with the child process return code.

**Tech Stack:** Python standard library (`subprocess`, `pathlib`, `os`, `sys`, `datetime`), existing `scripts/platform_train.py`, pytest.

---

### Task 1: Add launcher tests

**Files:**
- Create/Modify: `tests/scripts/test_launch_covra.py`
- Create: `launch_covra.py`

- [x] **Step 1: Write failing tests**

```python
def test_launch_covra_is_zero_arg_wrapper_with_editable_top_config():
    text = Path("launch_covra.py").read_text()
    assert "MANUAL CONFIGURATION" in text
    assert "subprocess.Popen" in text
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_launch_covra.py -q`

Expected: FAIL because `launch_covra.py` does not exist yet.

- [ ] **Step 3: Implement minimal launcher**

Create `launch_covra.py` with top-level manual config, path checks, directory creation, environment setup, tee logging, and child return-code propagation.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/scripts/test_launch_covra.py -q`

Expected: PASS.

- [ ] **Step 5: Run broader safety checks**

Run: `pytest tests/scripts/test_launch_covra.py tests/scripts/test_platform_train.py -q`

Expected: PASS; no GPU training is executed.
