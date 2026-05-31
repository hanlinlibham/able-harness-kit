# able-harness-kit

**Thin, backend-neutral agent-harness middlewares for LangChain / deepagents.**

Production-distilled patterns that compose on top of any LangChain
`AgentMiddleware` stack (including [deepagents](https://github.com/langchain-ai/deepagents)).
They don't replace your agent — they harden the loop around it.

```bash
pip install able-harness-kit          # after the first PyPI release
# or, from a clone:
pip install -e .
```

## Why

Most agent failures aren't "the model isn't smart enough" — they're the loop
*around* the model lacking observation and control:

| Middleware | The failure it kills |
|---|---|
| `ProgressAwareLoopGuardMiddleware` | The agent repeats a tool call that returns no new information, spinning until it burns its budget — while *not* tripping on polling or pagination, which repeat the call but get new results. |
| `LoopGuardMiddleware` | The cheaper case: the agent calls the same tool with identical args N times in a row (no view of the result needed). |
| `BinaryReadGuardMiddleware` | `read_file` hands the model a base64 block for a binary file; a lossy gateway drops it; the model *claims it read the file* and hallucinates. |
| `ToolResultBudgetMiddleware` | A 200 KB tool dump blows the context window and forces premature summarization. |

Each is **backend-neutral** (depends only on LangChain / LangGraph public
types), small enough to read in one sitting, and composes *with* — rather than
replaces — your existing harness.

## Usage

```python
from langchain.agents import create_agent          # or deepagents.create_deep_agent
from able_harness_kit import (
    ProgressAwareLoopGuardMiddleware,
    BinaryReadGuardMiddleware,
    ToolResultBudgetMiddleware,
)

agent = create_agent(
    model,
    tools=[...],
    middleware=[
        BinaryReadGuardMiddleware(),                   # fail loud on binary read_file
        ToolResultBudgetMiddleware(limit=16_000),      # cap oversized tool results
        ProgressAwareLoopGuardMiddleware(stop_at=3),   # stop loops that make no progress
    ],
)
```

### `ProgressAwareLoopGuardMiddleware`
The cheap loop signal — "same tool, same args, N times" — is wrong about half the
time. Polling a job, walking pagination, or tailing a stream all repeat the call
and *should*: each one returns new information. The difference between progress and
a stuck loop isn't in the arguments, it's in the **output**.

This guard runs the tool, compares the result to the previous same-tool result,
and intervenes only when a call both repeats *and* returns no new information
(`same_args_repeat_count >= stop_at` **and** `new_information_delta < progress_floor`).
Genuine progress passes through; a true spin gets a model-facing directive (not a
synthetic user message). For tools where a content-hash comparison is too lax — a
TODO tool re-sent with a one-character cosmetic edit, say — pass
`delta_overrides={"write_todos": make_set_membership_delta("todos")}`.

The output-delta primitives are exported standalone:
`build_observations(state["messages"])` reconstructs the per-call observations
(`ToolCallObservation` with `new_information_delta`) from history with no instance
state.

### `LoopGuardMiddleware`
The cheaper variant: fingerprints each call (name + normalized args), counts
consecutive identical-argument repeats, and short-circuits once `stop_at` is hit —
without looking at the result. Use it when you only want to catch a model hammering
an identical call and don't need output tracking. The counter resets the moment a
different call is seen. Pass `on_signal=` to observe without changing control flow.

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
*entropy-reducing control system* rather than overhead. The progress-aware loop
signal — keying off whether a repeated call returned new information, not just
whether the arguments repeated — is the open distillation of that controller's
loop detector. The middlewares here are the reusable, framework-neutral core of
that work.

## License

MIT © Hanlin Li
