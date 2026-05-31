# able-harness-kit

**Thin, backend-neutral agent-harness middlewares for LangChain / deepagents.**

Three production-distilled patterns that compose on top of any LangChain
`AgentMiddleware` stack (including [deepagents](https://github.com/langchain-ai/deepagents)).
They don't replace your agent — they harden the loop around it.

```bash
pip install able-harness-kit      # or, from a clone:  pip install -e .
```

## Why

Most agent failures aren't "the model isn't smart enough" — they're the loop
*around* the model lacking observation and control:

| Middleware | The failure it kills |
|---|---|
| `LoopGuardMiddleware` | The agent calls the same tool with the same args, making no progress, until it burns its budget. |
| `BinaryReadGuardMiddleware` | `read_file` hands the model a base64 block for a binary file; a lossy gateway drops it; the model *claims it read the file* and hallucinates. |
| `ToolResultBudgetMiddleware` | A 200 KB tool dump blows the context window and forces premature summarization. |

Each is **backend-neutral** (depends only on `langchain` message/middleware
types), small enough to read in one sitting, and composes *with* — rather than
replaces — your existing harness.

## Usage

```python
from langchain.agents import create_agent          # or deepagents.create_deep_agent
from able_harness_kit import (
    LoopGuardMiddleware,
    BinaryReadGuardMiddleware,
    ToolResultBudgetMiddleware,
)

agent = create_agent(
    model,
    tools=[...],
    middleware=[
        BinaryReadGuardMiddleware(),                  # fail loud on binary read_file
        ToolResultBudgetMiddleware(limit=16_000),     # cap oversized tool results
        LoopGuardMiddleware(warn_at=3, stop_at=5),    # stop no-progress tool loops
    ],
)
```

### `LoopGuardMiddleware`
`observe(tool_call) -> LoopSignal -> Action(WARN | STOP)`. Fingerprints each
tool call (name + normalized args), counts consecutive repeats, and
short-circuits with a directive once `stop_at` is hit. The counter resets the
moment a different call is seen — interventions are minimal and reversible.
Pass `on_signal=` to observe loop signals without changing control flow.

### `BinaryReadGuardMiddleware`
Intercepts `read_file` results whose declared media type isn't text-like and
replaces the base64 payload with a structured error pointing at a dedicated
extractor (OCR / doc-to-markdown / type sniff). Turning a silent hallucination
into an explicit "use the right tool" nudge is strictly safer; the only cost is
one extra tool hop on the first attempt. Tool name, metadata keys, and the
directive are configurable; disable per-deployment with
`BINARY_READ_GUARD_ENABLED=0`.

### `ToolResultBudgetMiddleware`
Caps each tool result at a character budget. Oversized results are offloaded via
a caller-supplied `offload(content, tool_name) -> OffloadRef` callback (bring
your own store — filesystem, blob, vector DB), or truncated with a retrieval
hint by default.

## Background

These were distilled from a multi-agent product running on non-Anthropic model
gateways (Qwen / GLM / DeepSeek), and from a controlled experiment across 5
models × 9 behavioral probes (harness on vs. off) that framed a harness as an
*entropy-reducing control system* rather than overhead. The middlewares here are
the reusable, framework-neutral core of that work.

## License

MIT © Hanlin Li
