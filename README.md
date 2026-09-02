# SCX Router

A lightweight GLiClass-based model router that ranks candidate LLMs in one
non-generative pass. Candidate labels can be changed at inference time.

[Hugging Face model](https://huggingface.co/scx-admin/scx-router-v0.1) ·
[Paper](paper/SCX_Model_Router.pdf)

## Install

```bash
pip install gliclass -U
```

The checkpoint is downloaded from Hugging Face on first use.

## Single input

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model_path = "scx-admin/scx-router-v0.1"  # or a local path to this checkpoint

model = GLiClassModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="multi-label", device="cuda:0"
)

text = "Write a Python function that merges two sorted linked lists."
labels = [
    "coder", "DeepSeek-V3.1", "gemma-4-31B-it", "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct", "MAGPiE",
    "Meta-Llama-3.3-70B-Instruct", "Qwen3-32B",
]

for r in pipeline(text, labels, threshold=0.5)[0]:
    print(r["label"], "=>", round(r["score"], 3))
```

Use `device="cpu"` on a CPU-only machine.

## Streamed input

Using the same `model_path` and `labels`:

```python
from gliclass import GLiClassModel, StreamingZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

router = StreamingZeroShotClassificationPipeline(
    model, tokenizer, classification_type="multi-label", device="cuda:0"
)

session_id = "chat-42"
turns = [
    "I need help refactoring some Rust code.",
    "Specifically the borrow checker keeps rejecting this function.",
    "fn parse(&mut self, buf: &[u8]) -> Result<Token, Error> { ... } -- here's the body.",
]

for turn in turns:
    out = router(turn, labels, session_ids=session_id, threshold=0.5)[0]
    print(f"+{out['tokens_added']} new tokens -> cache now {out['cached_length']} tokens")
    if out["triggered"]:
        for pred in sorted(out["predictions"], key=lambda p: -p["score"]):
            print(f"  {pred['label']:<38s} {pred['score']:.3f}")

# only each turn's own tokens are encoded -- earlier turns stay in the KV cache
```

## Paper

[SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology](paper/SCX_Model_Router.pdf)

## License

[Apache License 2.0](LICENSE)
