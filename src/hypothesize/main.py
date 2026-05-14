"""Entrypoint for pennsieve-ai-hypothesize.

Reads Scout's blackboard from `$INPUT_DIR/blackboard.json`, runs the
HypGen ↔ Critic iteration loop, and writes the updated blackboard to
`$OUTPUT_DIR` for the Analyze stage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hypothesize.blackboard import Blackboard
from hypothesize.llm import LLMClient
from hypothesize.pipeline import run_hypothesize


def main() -> int:
    input_dir = Path(os.environ.get("INPUT_DIR", "/mnt/efs/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/mnt/efs/output"))
    execution_run_id = os.environ.get("EXECUTION_RUN_ID", "local-dev")
    workflow_instance_id = os.environ.get("WORKFLOW_INSTANCE_ID", "local-dev")

    print("=== pennsieve-ai-hypothesize ===")
    print(f"  INPUT_DIR={input_dir}")
    print(f"  OUTPUT_DIR={output_dir}")
    print(f"  EXECUTION_RUN_ID={execution_run_id}")
    print(f"  WORKFLOW_INSTANCE_ID={workflow_instance_id}")

    bb_path = input_dir / "blackboard.json"
    if not bb_path.exists():
        print(f"ERROR: no blackboard.json at {bb_path} — Scout did not produce "
              f"its expected output, or the workflow wiring is wrong.",
              file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- loading blackboard from {bb_path} ---")
    bb = Blackboard.load(bb_path)
    bb.run_id = execution_run_id  # adopt the platform's run identifier
    bb.log("hypothesize", "start",
           f"loaded blackboard: {len(bb.dataset_summaries)} datasets, "
           f"{len(bb.literature_claims)} claims, "
           f"{len(bb.deep_evidence)} deep evidence chunks, "
           f"max_iterations={bb.max_iterations}")
    print(f"  research_question: {bb.research_question[:200]}")
    print(f"  datasets: {len(bb.dataset_summaries)} | "
          f"literature claims: {len(bb.literature_claims)} | "
          f"deep evidence: {len(bb.deep_evidence)} | "
          f"max_iterations: {bb.max_iterations}")

    client = LLMClient()

    try:
        run_hypothesize(bb, client)
    except Exception as e:
        bb.log("hypothesize", "fatal", repr(e))
        _write_outputs(bb, client, output_dir)
        raise

    bb.log("hypothesize", "done", "")
    _write_outputs(bb, client, output_dir)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0


def _write_outputs(bb: Blackboard, client: LLMClient, output_dir: Path) -> None:
    """Persist blackboard + audit logs to OUTPUT_DIR for the next stage."""
    bb.save(output_dir / "blackboard.json")
    nb = "\n".join(json.dumps(e) for e in bb.notebook)
    (output_dir / "notebook.jsonl").write_text(nb + "\n" if nb else "")
    ul = "\n".join(json.dumps(e) for e in client.usage_log)
    (output_dir / "usage.jsonl").write_text(ul + "\n" if ul else "")


if __name__ == "__main__":
    raise SystemExit(main())
