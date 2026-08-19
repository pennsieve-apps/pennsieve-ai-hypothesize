# pennsieve-ai-hypothesize

Pennsieve compute node processor — **stage 2 of 4** of the `pennsieve-ai`
research pipeline.

Consumes Scout's blackboard, runs the **HypGen ↔ Critic iteration loop**
(both agents on Opus), and produces a blackboard with locked hypotheses
plus per-iteration critic reviews for the downstream Analyze stage.

## Input

Reads `$INPUT_DIR/blackboard.json` — the output of `pennsieve-ai-scout`.

| Env var | Source | Description |
|---|---|---|
| `INPUT_DIR` | platform-injected | Scout's output: contains `blackboard.json` |
| `OUTPUT_DIR` | platform-injected | Where to write updated `blackboard.json` |
| `EXECUTION_RUN_ID` | platform-injected | Governor budget attribution key |
| `WORKFLOW_INSTANCE_ID` | platform-injected | Workflow execution identifier |
| `LLM_GOVERNOR_FUNCTION` | platform-injected | Lambda name for the LLM governor |

No user parameters — all configuration is carried on the blackboard from
Scout (notably `bb.max_iterations`, default 3).

## Output

Writes `$OUTPUT_DIR/blackboard.json` with the original Scout state plus:

- `bb.hypotheses` — N hypotheses per iteration, with claim, rationale, suggested experiment, dataset citations, and per-hypothesis evidence chunks (if the RAG layer is mounted)
- `bb.critic_reviews` — Critic verdicts (`accept` / `refine` / `reject`) with novelty/plausibility/feasibility/alignment scores and attack texts, per hypothesis per iteration
- `bb.iterations_completed`, `bb.stop_reason` — terminal state of the loop
- `bb.papers_enriched` — CrossRef-authoritative author/year/title/journal for every cited DOI (for the Writer stage's bibliography)

## Loop termination

The loop runs `bb.max_iterations` rounds, but exits early on:

- **`all_accepted`** — every hypothesis in the current iteration has verdict `accept` from the Critic.
- **`no_improvement`** — no scores changed between the prior and current iteration (the loop has stalled).

Hypotheses accepted in a prior iteration are **locked** — they're carried forward verbatim rather than re-revised, which prevents score drift and gives the loop a real accept-set to terminate on.

## Layers consumed

| Layer | Mount | Producer |
|---|---|---|
| `pennsieve-ai-rag` | `/mnt/layers/pennsieve-ai-rag/` (read-only) | `pennsieve-ai-rag-loader` workflow |

When mounted, each generated hypothesis gets top-K paragraph-level evidence chunks attached, which the Critic uses to attack claims with specific citations. When absent, the loop runs without deep evidence — hypotheses are still produced, but grounded only on titles/abstracts from Scout.

## LLM access

All LLM calls route through the Pennsieve LLM Governor Lambda
(`$LLM_GOVERNOR_FUNCTION`) — same `LLMClient` wrapper as `pennsieve-ai-scout`.
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

### Why Opus matters here

Both HypGen and Critic rely on Opus's extended-thinking capability. The
Critic in particular needs to reason rigorously about whether a hypothesis
is novel, plausible, feasible, and aligned with the research question —
this is what makes the closed loop actually converge instead of generating
plausible-sounding nonsense. Don't downgrade Critic to Sonnet without a
side-by-side eval (see project memory).

### Prefill compatibility

Opus 4.5+ extended-thinking models reject the "prefill `{`" trick. The
`call_json` helper in `agents/_common.py` detects Opus and skips prefill,
relying on the model's prompt + `extract_json`'s regex recovery. Same fix
as Scout.

## Status

**v0.0.6 candidate — locally validated, not yet tested in Pennsieve compute.** Agents and prompts
vendored from the original `virtual-postdoc` research codebase. Import
graph resolves cleanly. Not yet deployed.

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
$OUTPUT_DIR/blackboard.json
```
