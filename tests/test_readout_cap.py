"""The `[READOUT]` rule is enforced in code, not left to the critic model.

In a run on `pennsieve:405` the critic filed a `[READOUT]` attack and then
scored Feasibility 3 and accepted the hypothesis anyway, three times.
"""
from __future__ import annotations

from hypothesize.agents.critic import _apply_readout_cap
from pennsieve_ai_utils.blackboard import Blackboard

READOUT = "[READOUT] The ADCP folders cover only 3 of the 12 clones the claim needs."
REASONING = "[REASONING] The threshold is arbitrary."


def bb():
    return Blackboard("Does H15K6 bind?", "pennsieve:405")


def test_readout_attack_caps_feasibility_and_blocks_accept():
    board = bb()
    feasibility, verdict = _apply_readout_cap(board, "H3", 3, "accept", [REASONING, READOUT])
    assert feasibility == 2
    assert verdict == "revise"
    assert any(e["action"] == "readout_cap" for e in board.notebook)


def test_readout_attack_leaves_a_low_score_alone():
    feasibility, verdict = _apply_readout_cap(bb(), "H6", 2, "revise", [READOUT])
    assert (feasibility, verdict) == (2, "revise")


def test_reject_stays_reject():
    feasibility, verdict = _apply_readout_cap(bb(), "H4", 3, "reject", [READOUT])
    assert (feasibility, verdict) == (2, "reject")


def test_no_readout_attack_changes_nothing():
    board = bb()
    feasibility, verdict = _apply_readout_cap(board, "H2", 4, "accept", [REASONING])
    assert (feasibility, verdict) == (4, "accept")
    assert not [e for e in board.notebook if e["action"] == "readout_cap"]


def test_a_readout_tag_in_the_middle_of_an_attack_does_not_fire():
    # Only the tag the prompt asks for, at the start of the attack, counts.
    attacks = ["[REASONING] A better [READOUT] would be the functional assay."]
    assert _apply_readout_cap(bb(), "H5", 4, "accept", attacks) == (4, "accept")
