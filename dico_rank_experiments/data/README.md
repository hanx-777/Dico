# Local Data

This directory vendors the GSM8K JSONL files used by the default experiment
configs:

```text
data/gsm8k/main/train.jsonl
data/gsm8k/main/test.jsonl
```

The files contain the original `question` and `answer` fields from GSM8K:

- train: 7473 examples
- test: 1319 examples

Keeping these files in the project avoids repeated Hugging Face dataset
downloads on the A800 server. The model path still needs to point to a local
model directory, for example `/ai/lxw/lxw/Qwen3-8B`.
