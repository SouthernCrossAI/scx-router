from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .router import DEFAULT_MODEL_LABELS, MODEL_ID, SCXRouter


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scx-router",
        description="Rank candidate LLMs for a prompt with SCX Router v0.1.",
    )
    parser.add_argument("text", help="Prompt or request to route")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_MODEL_LABELS),
        metavar="MODEL",
        help="Candidate model labels",
    )
    parser.add_argument("--top-k", type=_positive_int, default=3, help="Results to print")
    parser.add_argument("--device", help="Torch device, for example cuda:0, mps, or cpu")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    router = SCXRouter(model_id=args.model, device=args.device)
    scores = router.rank(args.text, labels=args.labels)
    print(json.dumps(scores[: args.top_k], indent=2))
