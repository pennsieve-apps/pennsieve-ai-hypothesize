"""Hypothesize pipeline — Stage 2 of the pennsieve-ai research pipeline.

Reads Scout's blackboard from INPUT_DIR, runs the HypGen ↔ Critic iteration
loop, writes the updated blackboard to OUTPUT_DIR for the Analyze stage.

Adapted from the closed-loop section (lines 828-872) of the original
research-codebase orchestrator. The loop terminates on `all_accepted`,
`no_improvement`, or reaching `bb.max_iterations`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
from typing import TYPE_CHECKING

from hypothesize.agents import critic, hypothesis_generator
from pennsieve_ai_utils.blackboard import Blackboard
from pennsieve_ai_utils.tools import crossref

if TYPE_CHECKING:
    from pennsieve_ai_utils.llm import LLMClient


LAYER_RAG_PATH = Path(os.environ.get("VP_RAG_PATH", "/mnt/layers/pennsieve-ai-rag"))
PER_HYPOTHESIS_K = 8
PER_HYPOTHESIS_POOL = 32


def run_hypothesize(bb: Blackboard, client: "LLMClient") -> None:
    """Run the HypGen ↔ Critic iteration loop in place on `bb`."""
    iteration = 0
    while iteration < bb.max_iterations:
        iteration += 1
        print(f"\n--- iteration {iteration}/{bb.max_iterations} ---")
        locked_ids = _carry_forward_accepts(bb, iteration)

        print("  hypothesis_generator")
        hypothesis_generator.run(bb, client, iteration=iteration, locked_ids=locked_ids)
        _populate_per_hypothesis_evidence(bb, iteration)

        print("  critic")
        critic.run(bb, client, iteration=iteration, locked_ids=locked_ids)
        bb.iterations_completed = iteration
        for review in bb.reviews_at(iteration):
            bb.log("critic", "verdict", f"iter={iteration} hypothesis={review.hypothesis_id} "
                   f"verdict={review.verdict} feasibility={review.feasibility} "
                   f"attacks={review.attacks}")
        if os.environ.get("OUTPUT_DIR"):
            directory = Path(os.environ["OUTPUT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            bb.save(directory / f"blackboard.iter{iteration}.json")

        if _all_accepted(bb, iteration):
            bb.stop_reason = f"all_accepted at iteration {iteration}"
            bb.log("pipeline", "stop", bb.stop_reason)
            break
        if _no_improvement(bb, iteration):
            bb.stop_reason = f"no_improvement at iteration {iteration}"
            bb.log("pipeline", "stop", bb.stop_reason)
            break
    else:
        bb.stop_reason = f"max_iterations ({bb.max_iterations}) reached"
        bb.log("pipeline", "stop", bb.stop_reason)

    bb.iterations_completed = iteration
    _enrich_dois(bb)

    final_iter = iteration
    n_hyps = len(bb.hypotheses_at(final_iter))
    n_accepts = sum(1 for r in bb.reviews_at(final_iter) if r.verdict == "accept")
    print(f"\nHypothesize complete: {final_iter} iterations, "
          f"{n_hyps} hypotheses at final iter, {n_accepts} accepted. "
          f"Stop: {bb.stop_reason}")


# ---------------------------------------------------------------------------
# Loop control helpers (lifted from orchestrator.py:53-93 with imports adapted)
# ---------------------------------------------------------------------------

def _carry_forward_accepts(bb: Blackboard, iteration: int) -> set[str]:
    """At iteration > 1, copy accepted hypotheses (and their reviews) from the
    previous iteration into the current iteration verbatim. Returns the set of
    locked hypothesis IDs so HypGen/Critic can skip them.

    Rationale: re-revising and re-critiquing an already-accepted hypothesis
    tends to drift its scores down across iterations (Critic becomes more
    demanding, HypGen tweaks unnecessarily). Locking accepts gives the loop
    a real accept-set to terminate on."""
    if iteration < 2:
        return set()
    prev_reviews = {r.hypothesis_id: r for r in bb.reviews_at(iteration - 1)}
    locked: set[str] = set()
    for h in bb.hypotheses_at(iteration - 1):
        r = prev_reviews.get(h.id)
        if r and r.verdict == "accept":
            bb.hypotheses.append(replace(h, iteration=iteration))
            bb.critic_reviews.append(replace(r, iteration=iteration))
            locked.add(h.id)
    if locked:
        bb.log("pipeline", f"locked_iter{iteration}",
               f"{len(locked)} accepted hypotheses carried forward: {','.join(sorted(locked))}")
    return locked


def _all_accepted(bb: Blackboard, iteration: int) -> bool:
    hypotheses = bb.hypotheses_at(iteration)
    reviews = {r.hypothesis_id: r for r in bb.reviews_at(iteration)}
    return bool(hypotheses) and all(
        h.id in reviews and reviews[h.id].verdict == "accept" for h in hypotheses
    )


def _no_improvement(bb: Blackboard, iteration: int) -> bool:
    """Stop if no scores changed between the prior and current iteration —
    the loop has stalled and further iterations are unlikely to help."""
    if iteration < 2:
        return False
    cur = {r.hypothesis_id: r.novelty + r.plausibility + r.feasibility + r.alignment
           for r in bb.reviews_at(iteration)}
    prev = {r.hypothesis_id: r.novelty + r.plausibility + r.feasibility + r.alignment
            for r in bb.reviews_at(iteration - 1)}
    shared = set(cur) & set(prev)
    if not shared:
        return False
    return all(cur[k] == prev[k] for k in shared)


# ---------------------------------------------------------------------------
# Deep evidence retrieval — per-hypothesis chunks from the persistent RAG
# layer. No-op when the layer isn't mounted or is empty.
# ---------------------------------------------------------------------------

def _populate_per_hypothesis_evidence(bb: Blackboard, iteration: int) -> None:
    """Retrieve top-K RAG chunks specific to each hypothesis at this iteration.
    Populates `Hypothesis.evidence_chunks`. Idempotent — skips already-populated
    hypotheses. No-ops cleanly when the RAG layer isn't mounted."""
    if not LAYER_RAG_PATH.exists():
        return
    try:
        from pennsieve_ai_utils.tools.rag_store import RAGStore
        store = RAGStore(LAYER_RAG_PATH, read_only=True, dataset_doi=next(
                (d.doi for d in bb.dataset_summaries if d.id == bb.subsystem), ""))
    except Exception as e:
        bb.log("pipeline", "rag_store_init_warn", f"{e!r}")
        return
    if store.stats().get("chunks", 0) == 0:
        return

    topics_filter = _run_topics_filter(bb)
    populated = 0
    for h in bb.hypotheses_at(iteration):
        if h.evidence_chunks:
            continue
        query = f"{h.claim}\n{h.rationale}\n{h.suggested_experiment}"
        try:
            chunks = store.retrieve(
                query, k=PER_HYPOTHESIS_K, candidate_pool=PER_HYPOTHESIS_POOL,
                topics_filter=topics_filter,
            )
        except Exception as e:
            bb.log("pipeline", "per_hyp_retrieval_warn", f"{h.id}: {e!r}")
            continue
        h.evidence_chunks = [
            {"chunk_id": c.chunk_id, "doi": c.doi, "section": c.section,
             "text": c.text, "score": round(c.score, 4)}
            for c in chunks
        ]
        populated += 1
    if populated:
        bb.log(
            "pipeline", "per_hyp_retrieval",
            f"iter {iteration}: populated per-hypothesis evidence for "
            f"{populated}/{len(bb.hypotheses_at(iteration))} hypotheses "
            f"(k={PER_HYPOTHESIS_K} each)",
        )


_GENERIC_TAG_STOPWORDS = {
    "sparc", "dandi", "pennsieve", "human", "humans", "mouse", "mice",
    "rat", "rats", "data", "dataset", "modeling", "imaging",
    "electrophysiology", "open", "oad",
}


def _run_topics_filter(bb: Blackboard) -> set[str] | None:
    """Topical filter for retrieval — prefers extracted topics from the seed
    dataset summary, falls back to filtered measurements tags."""
    if not bb.dataset_summaries:
        return None
    seed = bb.dataset_summaries[0]
    extracted = getattr(seed, "extracted_topics", None) or []
    if extracted:
        return {t.strip().lower() for t in extracted if t and t.strip()}
    topics: set[str] = set()
    for tag in (seed.measurements or []):
        t = (tag or "").strip().lower().replace(" ", "-").replace("_", "-")
        if t and t not in _GENERIC_TAG_STOPWORDS:
            topics.add(t)
    return topics or None


# ---------------------------------------------------------------------------
# DOI enrichment — CrossRef authoritative author/year/title/journal metadata
# ---------------------------------------------------------------------------

def _enrich_dois(bb: Blackboard) -> None:
    """Populate `bb.papers_enriched` with authoritative metadata for every
    DOI cited across literature claims, papers_seen, deep_evidence, and
    per-hypothesis evidence. Used by the Writer stage to render inline
    citations without hallucinated authors."""
    cited: set[str] = set()

    def _add(raw: str) -> None:
        d = (raw or "").strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if d.startswith(prefix):
                d = d[len(prefix):]
        if d:
            cited.add(d)

    for c in bb.literature_claims:
        _add(getattr(c, "source", "") or "")
    for p in bb.papers_seen:
        _add(p.get("doi") or "")
    for c in bb.deep_evidence:
        _add(c.get("doi") or "")
    for h in bb.hypotheses:
        for c in (h.evidence_chunks or []):
            _add(c.get("doi") or "")

    pending = [doi for doi in cited if doi not in bb.papers_enriched]
    if not pending:
        return

    def fetch(doi: str) -> tuple[str, dict | None]:
        try:
            return doi, crossref.fetch_doi_metadata(doi)
        except Exception:
            return doi, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for doi, rec in pool.map(fetch, pending):
            if rec:
                bb.papers_enriched[doi] = {
                    "title": rec.get("title", ""),
                    "authors": rec.get("authors", []),
                    "year": rec.get("year", ""),
                    "journal": rec.get("journal", ""),
                }
    bb.log("pipeline", "enrich_dois",
           f"enriched {len(pending)} new DOIs ({len(bb.papers_enriched)} total)")
