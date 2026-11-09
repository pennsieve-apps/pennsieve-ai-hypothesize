"""Entrypoint for pennsieve-ai-hypothesize.

Reads Scout's blackboard from `$INPUT_DIR/blackboard.json`, runs the
HypGen ↔ Critic iteration loop, and writes the updated blackboard to
`$OUTPUT_DIR` for the Analyze stage.
"""
from __future__ import annotations

import sys

from pennsieve_ai_utils.llm import LLMClient
from pennsieve_ai_utils.models import ModelNotAvailable
from pennsieve_ai_utils.processor import (
    ProcessorEnv,
    install_sigterm_handler,
    load_upstream_blackboard,
    write_outputs,
)
from pennsieve_ai_utils.tools.reranker import reranker_mode

from hypothesize.pipeline import run_hypothesize

STAGE = "hypothesize"


def main() -> int:
    install_sigterm_handler()
    env = ProcessorEnv.from_environ()
    env.banner(STAGE)

    bb = load_upstream_blackboard(env.input_dir / "blackboard.json", stage=STAGE,
                                  upstream="Scout", run_id=env.execution_run_id)
    if bb is None:
        return 2
    output_dir = env.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bb.log(STAGE, "start",
           f"loaded blackboard: {len(bb.dataset_summaries)} datasets, "
           f"{len(bb.literature_claims)} claims, "
           f"{len(bb.deep_evidence)} deep evidence chunks, "
           f"max_iterations={bb.max_iterations}")
    print(f"  research_question: {bb.research_question[:200]}")
    print(f"  datasets: {len(bb.dataset_summaries)} | "
          f"literature claims: {len(bb.literature_claims)} | "
          f"deep evidence: {len(bb.deep_evidence)} | "
          f"max_iterations: {bb.max_iterations}")
    bb.save(output_dir / "blackboard.json")

    # HypGen and the Critic reason; dataset compatibility synthesizes. The bulk
    # role is checked only when the Claude reranker (Haiku 4.5) will use it.
    roles = ("reasoning", "synthesis") + (("bulk",) if reranker_mode() == "claude" else ())
    client = LLMClient(stage=STAGE, roles=roles)
    try:
        assignments = client.preflight()
    except ModelNotAvailable as e:
        bb.log(STAGE, "fatal", f"model preflight failed: {e}")
        write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
        print(f"ERROR: {e}", file=sys.stderr)
        return 5
    for role, assignment in assignments.items():
        bb.record_model_assignment(stage=STAGE, role=role, assignment=assignment)
    bb.log(STAGE, "models",
           ", ".join(f"{role}={a['model_id']}" for role, a in sorted(assignments.items())))

    try:
        run_hypothesize(bb, client)
    except (Exception, KeyboardInterrupt) as e:
        bb.log(STAGE, "fatal", repr(e))
        write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
        raise

    accepted = {r.hypothesis_id for r in bb.reviews_at(bb.iterations_completed)
                if r.verdict == "accept"}
    accepted.intersection_update(h.id for h in bb.hypotheses_at(bb.iterations_completed))
    bb.log(STAGE, "done" if accepted else "blocked",
           f"{len(accepted)} accepted hypotheses; stop_reason={bb.stop_reason}")
    write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0 if accepted else 6


if __name__ == "__main__":
    raise SystemExit(main())
