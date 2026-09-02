# SCX Router in OpenCode

This example exposes the hosted SCX `router` model to OpenCode through a small
local OpenAI-compatible proxy. It is the public, runnable version of the
`router-shim` prototype and has no npm dependencies.

This integration calls `https://api.scx.ai/v1`; it does not run the Hugging
Face checkpoint locally.

## Requirements

- Node.js 18 or newer
- [OpenCode](https://opencode.ai)
- An SCX API key from a verified account with router access

## Start the shim

Clone this repository, then run:

```bash
cd scx-router/examples/opencode
export SCX_API_KEY="your-scx-api-key"
node server.mjs
```

The shim listens on `http://127.0.0.1:8787/v1`. Keep it running while using
OpenCode. Confirm it is ready with:

```bash
curl http://127.0.0.1:8787/v1/models
```

## Connect OpenCode

For a quick test, point OpenCode at the included config from a second terminal:

```bash
export OPENCODE_CONFIG=/absolute/path/to/scx-router/examples/opencode/opencode.json
opencode models scx-router
cd /path/to/your/project
opencode --model scx-router/auto
```

For permanent use, merge the `provider.scx-router` object from
[`opencode.json`](opencode.json) into your global
`~/.config/opencode/opencode.json` or a project's `opencode.json`.

Use the router in a non-interactive workflow with:

```bash
opencode run --model scx-router/auto "Implement the next task in this repository."
```

Choose a preset by changing the model name:

| Model | Quality | Speed | Cost |
| --- | ---: | ---: | ---: |
| `scx-router/auto` | 0.65 | 0.10 | 0.25 |
| `scx-router/auto-quality` | 0.90 | 0.05 | 0.05 |
| `scx-router/auto-fast` | 0.30 | 0.60 | 0.10 |
| `scx-router/auto-cheap` | 0.15 | 0.05 | 0.80 |

## How it works

On the first request in a conversation, the shim sends `model: "router"` and
the selected quality, speed, and cost weights to SCX. It remembers the model
returned by SCX and sends later requests in that conversation directly to that
model. This keeps an OpenCode session on one model after routing.

The selection cache is in memory, holds at most 500 conversations, and is
cleared when the shim restarts. The server binds only to `127.0.0.1`, and the
SCX API key stays in the shim process rather than the OpenCode config.

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8787` | Local shim port |
| `SCX_UPSTREAM_URL` | `https://api.scx.ai/v1` | Alternate SCX-compatible upstream |

If OpenCode reports `connection refused`, start the shim and confirm the port
matches `options.baseURL`. An upstream `401` or `403` means the SCX key or
router access needs attention.
