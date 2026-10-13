from __future__ import annotations

from hypothesize.agents._common import (
    call_json,
    format_dataset_evidence,
    format_datasets,
    load_prompt,
)
from pennsieve_ai_utils.blackboard import Blackboard, CriticReview
from pennsieve_ai_utils.llm import LLMClient

MODEL = "opus"
AGENT = "critic"


def _format_hypothesis_evidence(h) -> str:
    """Per-hypothesis RAG chunks shown to the Critic so it can attack with deeper detail."""
    if not getattr(h, "evidence_chunks", None):
        return ""
    lines = ["  Hypothesis-specific evidence (focused RAG retrieval):"]
    for i, c in enumerate(h.evidence_chunks):
        lines.append(
            f"    [E{i}] ({c.get('section','')}, {c.get('doi','')}, score={c.get('score',0):.2f})\n"
            f"        \"{(c.get('text','') or '')[:400]}\""
        )
    return "\n".join(lines)


def _format_hypotheses(hyps) -> str:
    out = []
    for h in hyps:
        block = (
            f"{h.id}: {h.claim}\n"
            f"  Rationale: {h.rationale}\n"
            f"  Supporting claims: {h.supporting_evidence}\n"
            f"  Predicted: {h.predicted_observation}\n"
            f"  Experiment: {h.suggested_experiment}"
        )
        evidence = _format_hypothesis_evidence(h)
        if evidence:
            block = block + "\n" + evidence
        files = format_dataset_evidence(
            getattr(h, "dataset_evidence", None) or [], indent="  ",
            heading="What the dataset's files contain, for this hypothesis (from file headers and listings):")
        if files:
            block = block + "\n" + files
        out.append(block)
    return "\n\n".join(out)


def _claim_tag(c) -> str:
    if c.entailment == "supports":
        return "VERIFIED"
    if c.entailment == "contradicts":
        return "CONTRADICTED"
    if c.entailment == "not_addressed":
        return "NOT-ADDRESSED"
    if c.quote_match == "missing":
        return "UNVERIFIED-no-quote"
    if c.quote_match == "partial":
        return "UNVERIFIED-paraphrase"
    return "UNVERIFIED"


def _format_claims(bb: Blackboard) -> str:
    return "\n".join(
        f"[{i}] ({c.evidence_strength}, {_claim_tag(c)}) {c.text}"
        for i, c in enumerate(bb.literature_claims)
    )


def _format_deep_evidence(bb: Blackboard) -> str:
    if not bb.deep_evidence:
        return ""
    lines = ["## Deep evidence (full-text chunks for this question)\n"]
    for i, c in enumerate(bb.deep_evidence):
        lines.append(
            f"[D{i}] ({c.get('section','')}, {c.get('doi','')}, score={c.get('score',0):.2f})\n"
            f"  \"{c.get('text','')[:600]}\""
        )
    return "\n".join(lines)


def _format_prior_papers(bb: Blackboard) -> str:
    """Render `bb.prior_papers` for the Critic. Critic uses these to attack
    hypotheses that essentially replicate already-published findings on this
    dataset (linked papers from the depositor or user-supplied via
    --prior-paper). Same content as the HypGen rendering."""
    pp = getattr(bb, "prior_papers", None) or []
    if not pp:
        return ""
    lines = [
        "## Prior analyses on this dataset (use these to attack `redundant_with_prior_paper`)",
        "",
        "These papers have already been published using this dataset. A "
        "hypothesis that essentially repeats one of these analyses is "
        "redundant and should be rejected with a `redundant_with_prior_paper` "
        "attack naming the specific [P#] it overlaps with.",
        "",
    ]
    for i, p in enumerate(pp, 1):
        authors = (p.get("authors") or "").strip()
        first_author = authors.split(",")[0].strip() if authors else ""
        year = (p.get("year") or "").strip()
        cite = (
            f"{first_author.split()[-1] if first_author else '?'} et al., {year}"
            if first_author and year else f"({p.get('source_label','?')})"
        )
        lines.append(f"### [P{i}] {cite} — doi:{p.get('doi','')} ({p.get('source_label','?')})")
        title = (p.get("title") or "").strip()
        if title:
            lines.append(f"**Title:** {title}")
        abstract = (p.get("abstract") or "").strip()
        if abstract:
            lines.append(f"**Abstract:** {abstract[:1000]}{'...' if len(abstract) > 1000 else ''}")
        lines.append("")
    return "\n".join(lines)


def _format_director_brief(bb: Blackboard) -> str:
    if not bb.director_brief:
        return ""
    b = bb.director_brief
    parts = [
        "## Director's brief (use this to score the alignment axis)",
        "",
        f"### Strategy\n{b.research_strategy}",
        "",
        "### Focus areas",
        "\n".join(f"- {fa}" for fa in b.focus_areas),
        "",
        "### Lab constraints",
        "\n".join(f"- {c}" for c in b.lab_constraints),
        "",
        "### Acceptance criteria",
        "\n".join(f"- {c}" for c in b.acceptance_criteria),
    ]
    return "\n".join(parts)


def run(bb: Blackboard, client: LLMClient, *, iteration: int = 1,
        locked_ids: set[str] | None = None) -> None:
    locked_ids = set(locked_ids or ())
    current_hyps = [h for h in bb.hypotheses_at(iteration) if h.id not in locked_ids]
    if not current_hyps:
        if locked_ids:
            bb.log(AGENT, f"skip_iter{iteration}",
                   f"all {len(locked_ids)} hypotheses locked, no fresh reviews needed")
        else:
            bb.log(AGENT, "skip", f"no hypotheses to review at iteration {iteration}")
        return

    system = load_prompt("critic")
    director_section = _format_director_brief(bb)
    deep_section = _format_deep_evidence(bb)
    files_section = format_dataset_evidence(
        bb.dataset_evidence, heading="## What the seed dataset's files contain (from file headers and listings)\n")
    prior_section = _format_prior_papers(bb)
    from hypothesize.agents.dataset_compatibility import format_for_hypgen
    compatibility_section = format_for_hypgen(bb)
    datasets_section = (
        f"Available datasets ({len(bb.dataset_summaries)}):\n{format_datasets(bb)}\n\n"
        if bb.dataset_summaries else ""
    )
    user = (
        f"Research question: {bb.research_question}\n"
        f"Iteration: {iteration} of up to {bb.max_iterations}\n\n"
        + (director_section + "\n\n" if director_section else "")
        + (prior_section + "\n\n" if prior_section else "")
        + f"Available evidence (literature claims, indexed):\n{_format_claims(bb)}\n\n"
        + (deep_section + "\n\n" if deep_section else "")
        + (files_section + "\n\n" if files_section else "")
        + datasets_section
        + (compatibility_section + "\n\n" if compatibility_section else "")
        + f"Hypotheses to review:\n{_format_hypotheses(current_hyps)}"
    )

    parsed = call_json(
        client,
        agent=AGENT,
        system=system,
        user=user,
        run_id=bb.run_id,
        model=MODEL,
    )

    for r in parsed.get("reviews", []):
        bb.critic_reviews.append(CriticReview(
            hypothesis_id=r["hypothesis_id"],
            novelty=r["novelty"],
            plausibility=r["plausibility"],
            feasibility=r["feasibility"],
            alignment=r.get("alignment", 3),  # tolerate older outputs that lack alignment
            attacks=r.get("attacks", []),
            verdict=r["verdict"],
            iteration=iteration,
        ))
    bb.log(AGENT, f"review_iter{iteration}", f"{len(parsed.get('reviews', []))} reviews")
