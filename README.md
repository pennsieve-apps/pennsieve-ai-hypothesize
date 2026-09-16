# pennsieve-ai-hypothesize

Pennsieve compute node processor — **stage 2 of 4** of the `pennsieve-ai`
research pipeline.

Consumes Scout's blackboard, runs the **HypGen ↔ Critic iteration loop**
(both agents request the logical `opus` reasoning role), and produces a
blackboard with locked hypotheses plus per-iteration critic reviews for the
downstream Analyze stage. The governor's effective allow-list determines the
concrete model used for that role at runtime.

## Input

Reads `$INPUT_DIR/blackboard.json` — the output of `pennsieve-ai-scout`.

| Env var | Source | Description |
|---|---|---|
| `INPUT_DIR` | platform-injected | Scout's output: contains `blackboard.json` |
| `OUTPUT_DIR` | platform-injected | Where to write updated `blackboard.json` |
| `EXECUTION_RUN_ID` | platform-injected | Governor budget attribution key |
| `WORKFLOW_INSTANCE_ID` | platform-injected | Workflow execution identifier |
| `LLM_GOVERNOR_FUNCTION_NAME` | platform-injected | Lambda name for the LLM governor |

There are no Pennsieve user parameters for this app. Scientific run
configuration is carried on the blackboard from Scout (notably
`bb.max_iterations`, default 3). Operators can use these optional environment
overrides when diagnosing or deliberately changing a deployment:

| Env var | Default | Purpose |
|---|---|---|
| `VP_MODEL_OPUS` | dynamically resolved | Explicitly pin the reasoning role; pinned models do not silently fall back |
| `VP_MODEL_SONNET` | dynamically resolved | Explicitly pin the synthesis role |
| `VP_MODEL_HAIKU` | dynamically resolved | Explicitly pin the bulk role |
| `VP_LLM_MAX_RETRIES` | `1` | Anthropic SDK retries inside each routed request attempt |
| `VP_BEDROCK_EMBED_ID` | `amazon.titan-embed-text-v2:0` | Embedding model used with the optional RAG layer |
| `VP_BEDROCK_RERANK_ID` | unset | Optional Bedrock reranking model |
| `VP_TITAN_PARALLELISM` | `16` | Parallel embedding requests |
| `VP_RAG_PATH` | `/mnt/layers/pennsieve-ai-rag` | Optional persistent RAG layer mount |
| `AWS_REGION` | `us-east-1` | Region for embedding and reranking calls |

## Output

Writes `$OUTPUT_DIR/blackboard.json` with the original Scout state plus:

- `bb.hypotheses` — N hypotheses per iteration, with claim, rationale, suggested experiment, dataset citations, and per-hypothesis evidence chunks (if the RAG layer is mounted)
- `bb.critic_reviews` — Critic verdicts (`accept` / `revise` / `reject`) with novelty/plausibility/feasibility/alignment scores and attack texts, per hypothesis per iteration
- `bb.iterations_completed`, `bb.stop_reason` — terminal state of the loop
- `bb.papers_enriched` — CrossRef-authoritative author/year/title/journal for every cited DOI (for the Writer stage's bibliography)
- `bb.model_assignments` — concrete model provenance for each logical role used by this stage

The app also writes:

- `$OUTPUT_DIR/notebook.jsonl` — structured scientific notebook events
- `$OUTPUT_DIR/usage.jsonl` — model routing, retry, failure, and token-usage events
- `$OUTPUT_DIR/blackboard.iter{N}.json` — state after each completed HypGen/Critic iteration

## Loop termination

The loop runs `bb.max_iterations` rounds, but exits early on:

- **`all_accepted`** — every hypothesis in the current iteration has verdict `accept` from the Critic.
- **`no_improvement`** — no scores changed between the prior and current iteration (the loop has stalled).

Hypotheses accepted in a prior iteration are **locked** — they're carried forward verbatim rather than re-revised, which prevents score drift and gives the loop a real accept-set to terminate on.

The process exits `0` only when the final iteration contains at least one
accepted hypothesis. It exits `6` when the loop stops without one, after
retaining the blackboard, iteration snapshots, and logs for review. Workflow
orchestration must stop before Analyze on exit `6`; other nonzero statuses are
fatal setup or execution errors.

## Layers consumed

| Layer | Mount | Producer |
|---|---|---|
| `pennsieve-ai-rag` | `/mnt/layers/pennsieve-ai-rag/` (read-only) | `pennsieve-ai-rag-loader` workflow |

When mounted, each generated hypothesis gets top-K paragraph-level evidence chunks attached, which the Critic uses to attack claims with specific citations. When absent, the loop runs without deep evidence — hypotheses are still produced, but grounded only on titles/abstracts from Scout.

## LLM access

All LLM calls route through the Pennsieve LLM Governor Lambda
(`$LLM_GOVERNOR_FUNCTION_NAME`) using the same routing design as
`pennsieve-ai-scout`.
Hypothesis Generator and Critic request the logical `opus` reasoning role.
At startup the service reads the governor's effective allow-list, selects the
strongest permitted model at or above the reasoning floor (Sonnet 4.6), and
makes a one-token call to verify that the governor IAM role can invoke it.
The concrete assignment is printed, saved on `bb.model_assignments`, and
included on every `usage.jsonl` event. `VP_MODEL_OPUS` remains an explicit pin;
preflight fails with the effective allow-list if that pin is not permitted.

Calls also self-heal when the allow-list or governor IAM permissions change
mid-run: permission failures are traced to `usage.jsonl`, the role is
re-resolved, and the request is retried. Real completions use streaming so
large HypGen/Critic prompts do not hit the Lambda transport's read timeout.

### Why the reasoning floor matters

HypGen and Critic both require reasoning-grade behavior. Opus is the preferred
model when it is permitted and invokable, while Sonnet 4.6 is the default
minimum quality floor. For the previously observed Pennsieve Test allow-list,
both logical roles resolved to Sonnet 4.6 automatically. v0.0.9 discovers,
verifies, and logs the current assignments at startup. A model below that
floor is used only through an explicit operator pin and should receive a
side-by-side scientific-quality evaluation before production use.

### Prefill compatibility

Some newer concrete model families, including Sonnet 4.6 and Opus 4.6+, reject
the assistant-prefill `{` technique. Request shaping follows the concrete
model selected at runtime, including after a retry reroutes the request.
`call_json` skips incompatible prefill and relies on the prompt plus
`extract_json` recovery; sampling parameters are also omitted for models that
do not accept them.

## Known limitation

`bb.model_assignments` currently records this stage's assignments and replaces
any assignment map received from Scout. Other upstream blackboard diagnostics,
including `bb.degraded`, are preserved. Cumulative, stage-namespaced model
provenance is intentionally deferred to the shared `pennsieve-ai-core`
blackboard contract so the four apps adopt one consistent representation.

## Status

**v0.0.9 release candidate — locally validated; Pennsieve Test rerun pending.**
This candidate adds iteration snapshots and critic decision logs, and makes a
no-accept result a blocking exit instead of a successful handoff to Analyze.
It requires the Anthropic 1.x-compatible Governor transport from
`pennsieve-llm` 0.6.1. The concrete Bedrock models will be confirmed by the
rerun's preflight log.

## Pipeline

```
[Scout's blackboard.json]
   │
   ▼
iteration loop (up to bb.max_iterations):
   ├─ _carry_forward_accepts → lock prior-iter accepts
   ├─ hypothesis_generator    → propose N hypotheses
   ├─ _populate_per_hypothesis_evidence → RAG chunks (if layer mounted)
   ├─ critic                   → verdict per hypothesis
   └─ check stop conditions    → all_accepted / no_improvement / max_iters
   │
   ▼
_enrich_dois → CrossRef metadata for every cited DOI
   │
   ▼
$OUTPUT_DIR/blackboard.json + blackboard.iterN.json + notebook.jsonl + usage.jsonl
```
