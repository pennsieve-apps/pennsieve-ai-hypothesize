from __future__ import annotations

from hypothesize.agents._common import call_json, format_datasets, load_prompt
from pennsieve_ai_utils.blackboard import Blackboard, CriticReview, Hypothesis
from pennsieve_ai_utils.llm import LLMClient

MODEL = "opus"
AGENT = "hypothesis_generator"


def _claim_tag(c) -> str:
    """Compact verification tag the model can read at a glance."""
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
        f"[{i}] ({c.evidence_strength}, {_claim_tag(c)}) {c.text} — {c.source}"
        for i, c in enumerate(bb.literature_claims)
    )




def _format_deep_evidence(bb: Blackboard) -> str:
    if not bb.deep_evidence:
        return ""
    lines = ["## Deep evidence (retrieved from full-text papers)\n"]
    for i, c in enumerate(bb.deep_evidence):
        lines.append(
            f"[D{i}] ({c.get('section','')}, {c.get('doi','')}, score={c.get('score',0):.2f})\n"
            f"  \"{c.get('text','')[:600]}\""
        )
    return "\n".join(lines)


def _format_prior_papers(bb: Blackboard) -> str:
    """Render `bb.prior_papers` — papers already published using this dataset
    (linked from depositor metadata) plus user-supplied DOIs (via
    --prior-paper). The Hypothesis Generator must NOT propose hypotheses
    whose claim/predicted-observation collapses to what these papers
    already demonstrated. The Critic enforces this with a `redundant_with_
    prior_paper` attack class."""
    pp = getattr(bb, "prior_papers", None) or []
    if not pp:
        return ""
    lines = [
        "## Prior analyses on this dataset (DO NOT REPLICATE)",
        "",
        "These papers have already been published using this dataset (or you "
        "were told to treat them as prior art). Hypotheses that essentially "
        "repeat what these analyses showed will be rejected by the Critic. "
        "Use them as the floor — propose what is NOT yet been tested.",
        "",
    ]
    for i, p in enumerate(pp, 1):
        authors = (p.get("authors") or "").strip()
        first_author = authors.split(",")[0].strip() if authors else ""
        year = (p.get("year") or "").strip()
        cite_token = (
            f"{first_author.split()[-1] if first_author else '?'} et al., {year}"
            if first_author and year else f"({p.get('source_label','?')})"
        )
        lines.append(f"### [P{i}] {cite_token} — doi:{p.get('doi','')} ({p.get('source_label','?')})")
        title = (p.get("title") or "").strip()
        if title:
            lines.append(f"**Title:** {title}")
        abstract = (p.get("abstract") or "").strip()
        if abstract:
            lines.append(f"**Abstract:** {abstract[:1200]}{'...' if len(abstract) > 1200 else ''}")
        lines.append("")
    return "\n".join(lines)


def _format_director_brief(bb: Blackboard) -> str:
    if not bb.director_brief:
        return ""
    b = bb.director_brief
    parts = [
        "## Director's brief (your hypotheses MUST align with this)",
        "",
        f"### Strategy\n{b.research_strategy}",
        "",
        "### Focus areas",
        "\n".join(f"- {fa}" for fa in b.focus_areas),
        "",
        "### Lab constraints (do not propose anything that violates these)",
        "\n".join(f"- {c}" for c in b.lab_constraints),
        "",
        "### Acceptance criteria (what makes a hypothesis worth pursuing here)",
        "\n".join(f"- {c}" for c in b.acceptance_criteria),
    ]
    return "\n".join(parts)


def _format_hypothesis_evidence(h: Hypothesis) -> str:
    """Per-hypothesis RAG chunks, formatted compactly for the revision prompt."""
    if not h.evidence_chunks:
        return ""
    lines = ["Hypothesis-specific evidence (focused RAG retrieval for this hypothesis):"]
    for i, c in enumerate(h.evidence_chunks):
        lines.append(
            f"  [E{i}] ({c.get('section','')}, {c.get('doi','')}, score={c.get('score',0):.2f})\n"
            f"      \"{(c.get('text','') or '')[:400]}\""
        )
    return "\n".join(lines)


def _format_revision_context(prev_hyps: list[Hypothesis], prev_reviews: list[CriticReview]) -> str:
    by_id = {r.hypothesis_id: r for r in prev_reviews}
    chunks = []
    for h in prev_hyps:
        r = by_id.get(h.id)
        if r is None:
            scores = "(no review available)"
            attacks: list[str] = []
            verdict = "unknown"
        else:
            scores = (
                f"novelty={r.novelty}/5, plausibility={r.plausibility}/5, "
                f"feasibility={r.feasibility}/5, alignment={r.alignment}/5"
            )
            attacks = r.attacks
            verdict = r.verdict
        per_hyp_evidence = _format_hypothesis_evidence(h)
        chunk = (
            f"### {h.id}  ({verdict.upper()}, {scores})\n"
            f"Claim: {h.claim}\n"
            f"Rationale: {h.rationale}\n"
            f"Predicted: {h.predicted_observation}\n"
            f"Experiment: {h.suggested_experiment}\n"
            f"Critic attacks:\n" + ("\n".join(f"  - {a}" for a in attacks) if attacks else "  (none)")
            + (f"\n{per_hyp_evidence}" if per_hyp_evidence else "")
        )
        chunks.append(chunk)
    return "\n\n".join(chunks)


def _format_locked_hypotheses(locked_hyps: list[Hypothesis]) -> str:
    if not locked_hyps:
        return ""
    lines = [
        "## Locked hypotheses (already ACCEPTED in a prior iteration — DO NOT regenerate or modify)",
        "",
        "These IDs are taken. Do not include them in your output. They are shown only so you can "
        "ensure your new hypotheses cover *different* angles and don't duplicate them.",
        "",
    ]
    for h in locked_hyps:
        lines.append(f"- **{h.id}**: {h.claim}")
    return "\n".join(lines)


def run(bb: Blackboard, client: LLMClient, *, iteration: int = 1,
        locked_ids: set[str] | None = None) -> None:
    if not bb.literature_claims:
        bb.log(AGENT, "skip", "no literature claims to ground hypotheses")
        return

    locked_ids = set(locked_ids or ())
    target_total = 5
    n_to_generate = target_total - len(locked_ids)
    if n_to_generate <= 0:
        bb.log(AGENT, f"skip_iter{iteration}",
               f"all {len(locked_ids)} hypotheses locked, nothing to generate")
        return

    system = load_prompt("hypothesis_generator")
    director_section = _format_director_brief(bb)
    deep_section = _format_deep_evidence(bb)
    prior_section = _format_prior_papers(bb)
    from hypothesize.agents.dataset_compatibility import format_for_hypgen
    compatibility_section = format_for_hypgen(bb)

    base = (
        f"Research question: {bb.research_question}\n"
        f"Subsystem: {bb.subsystem}\n\n"
        + (director_section + "\n\n" if director_section else "")
        + (prior_section + "\n\n" if prior_section else "")
        + f"Literature claims ({len(bb.literature_claims)}):\n{_format_claims(bb)}\n\n"
        + (deep_section + "\n\n" if deep_section else "")
        + f"Available datasets ({len(bb.dataset_summaries)}):\n{format_datasets(bb)}\n\n"
        + (compatibility_section + "\n\n" if compatibility_section else "")
    )

    locked_hyps = [h for h in bb.hypotheses_at(iteration) if h.id in locked_ids]
    locked_section = _format_locked_hypotheses(locked_hyps)

    if iteration == 1:
        user = base + f"Generate exactly {n_to_generate} hypotheses, all aligned with the Director's brief."
    else:
        # The slate to revise = previous-iteration hypotheses MINUS the ones locked
        # (locked ones are already carried forward; HG must not see them as revisable).
        prev_hyps = [h for h in bb.hypotheses_at(iteration - 1) if h.id not in locked_ids]
        prev_reviews = bb.reviews_at(iteration - 1)
        revision = _format_revision_context(prev_hyps, prev_reviews)
        user = (
            base
            + (locked_section + "\n\n" if locked_section else "")
            + f"## REVISION TASK (iteration {iteration} of up to {bb.max_iterations})\n\n"
            + f"Generate **exactly {n_to_generate} hypotheses** — no more, no fewer. Do not include "
            + "any locked IDs above. For each item below:\n"
            + "  - If `revise`, address each listed attack specifically — keep the same id and set "
            + "`revises` to the same id.\n"
            + "  - If `reject`, replace it with a stronger alternative on a different angle. Use a NEW "
            + "id (e.g. H6, H7); set `revises` to the rejected id.\n\n"
            + "Stay aligned with the Director's strategy.\n\n"
            + f"### Slate to revise (previous-iteration hypotheses needing work)\n\n{revision}"
        )

    parsed = call_json(
        client,
        agent=AGENT,
        system=system,
        user=user,
        run_id=bb.run_id,
        model=MODEL,
    )

    n_kept = 0
    for h in parsed.get("hypotheses", []):
        if h["id"] in locked_ids:
            # Defensive: model ignored the lock and tried to regenerate an accepted hyp.
            # Drop the duplicate — the locked version is authoritative.
            bb.log(AGENT, f"drop_iter{iteration}",
                   f"dropped duplicate of locked id {h['id']}")
            continue
        bb.hypotheses.append(Hypothesis(
            id=h["id"],
            claim=h["claim"],
            rationale=h["rationale"],
            supporting_evidence=h.get("supporting_evidence", []),
            predicted_observation=h["predicted_observation"],
            suggested_experiment=h["suggested_experiment"],
            iteration=iteration,
            revises=h.get("revises"),
        ))
        n_kept += 1
    locked_note = f" ({len(locked_ids)} locked)" if locked_ids else ""
    bb.log(AGENT, f"generate_iter{iteration}", f"{n_kept} hypotheses{locked_note}")
