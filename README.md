# SCX Router

A minimal Python component for ranking candidate LLMs with
[SCX Router v0.1](https://huggingface.co/scx-admin/scx-router-v0.1).

The model is a 0.6B-parameter, GLiClass-based zero-shot router. It scores all
candidates in one non-generative pass, so the candidate list can be changed at
inference time.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/SouthernCrossAI/scx-router.git
cd scx-router
python -m venv .venv
source .venv/bin/activate
pip install .
```

The checkpoint is downloaded from Hugging Face on first use.

## Use

From the command line:

```bash
scx-router "Write a Python function that merges two sorted linked lists."
```

With your own candidate models:

```bash
scx-router "Summarize this contract." \
  --labels fast-model balanced-model accurate-model \
  --top-k 2
```

From Python:

```python
from scx_router import SCXRouter

router = SCXRouter()
scores = router.rank(
    "Write a Python function that merges two sorted linked lists."
)
print(scores[0])
```

CUDA is used when available, followed by Apple MPS and then CPU. Override this
with `SCXRouter(device="cpu")` or the CLI's `--device` option.

## Examples

- [Use SCX Router in OpenCode](examples/opencode/)

## Paper

[SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology](paper/SCX_Model_Router.pdf)

## License

[Apache License 2.0](LICENSE)
