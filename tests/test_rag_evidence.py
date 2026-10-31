from types import SimpleNamespace

import pytest

from hypothesize import pipeline
from hypothesize.agents._common import format_dataset_evidence
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis


class FakeStore:
    def __init__(self, papers=0):
        self.papers = papers
        self.queries = []

    def stats(self):
        return {"chunks": self.papers}

    def retrieve(self, query, **kwargs):
        return [SimpleNamespace(chunk_id="p0", doi="10.1/x", section="Results", text="paper", score=0.5)]

    def retrieve_dataset(self, query, k=6, **kwargs):
        self.queries.append(query)
        return [SimpleNamespace(chunk_id="d0", doi="files/CDC", section="dataset_data_dictionary",
                                text="FITC-A (stain: anti-human IgG)", score=0.4)]


def bb_with_hypothesis():
    bb = Blackboard("Does H15K6 bind?", "pennsieve:405")
    bb.hypotheses = [Hypothesis("H1", "claim", "why", [], "pred", "test in CDC files")]
    return bb


def test_dataset_evidence_is_filled_even_without_paper_passages():
    bb, store = bb_with_hypothesis(), FakeStore(papers=0)
    pipeline._populate_run_evidence(bb, store)
    pipeline._populate_per_hypothesis_evidence(bb, 1, store)
    assert bb.deep_evidence == []
    assert bb.dataset_evidence[0]["source_ref"] == "files/CDC"
    h = bb.hypotheses[0]
    assert h.evidence_chunks == [] and h.dataset_evidence[0]["section"] == "dataset_data_dictionary"
    assert "test in CDC files" in store.queries[-1]


def test_paper_passages_are_used_when_the_index_has_them():
    bb, store = bb_with_hypothesis(), FakeStore(papers=3)
    pipeline._populate_run_evidence(bb, store)
    pipeline._populate_per_hypothesis_evidence(bb, 1, store)
    assert bb.deep_evidence[0]["doi"] == "10.1/x"
    assert bb.hypotheses[0].evidence_chunks[0]["text"] == "paper"


def test_required_full_text_stops_the_stage(monkeypatch):
    monkeypatch.setenv("VP_REQUIRE_DEEP_EVIDENCE", "1")
    with pytest.raises(RuntimeError, match="no full-text"):
        pipeline._check_required_evidence(bb_with_hypothesis())


def test_dataset_evidence_is_labelled_as_files_not_papers():
    text = format_dataset_evidence(
        [{"section": "dataset_data_dictionary", "source_ref": "files/CDC", "text": "FITC-A"}],
        heading="## Files")
    assert "[F0] (dataset_data_dictionary, files/CDC)" in text and "FITC-A" in text
    assert format_dataset_evidence([], heading="x") == ""


def test_a_requested_index_that_is_missing_stops_the_stage(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_RAG_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(pipeline, "LAYER_RAG_PATH", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="does not exist"):
        pipeline._open_rag_store(bb_with_hypothesis())


def test_an_index_for_another_dataset_stops_the_stage(monkeypatch, tmp_path):
    import gzip, json
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"dataset_doi": "10.1/other", "embedding_model_id": "amazon.titan-embed-text-v2:0",
         "dim": 256, "chunk_count": 0}))
    with gzip.open(tmp_path / "chunks.jsonl.gz", "wt") as stream:
        stream.write("")
    monkeypatch.setenv("VP_RAG_PATH", str(tmp_path))
    monkeypatch.setattr(pipeline, "LAYER_RAG_PATH", tmp_path)
    bb = bb_with_hypothesis()
    from pennsieve_ai_utils.blackboard import DatasetSummary
    bb.dataset_summaries = [DatasetSummary(id="pennsieve:405", title="t", description="",
                                           measurements=[], species="", n_subjects=0, doi="10.1/seed")]
    with pytest.raises(ValueError, match="must match"):
        pipeline._open_rag_store(bb)


def test_the_stage_client_reaches_the_rag_reader_for_reranking(monkeypatch, tmp_path):
    """The Claude reranker runs through the stage's own model client."""
    import gzip, json
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"dataset_doi": "10.1/seed", "embedding_model_id": "amazon.titan-embed-text-v2:0",
         "dim": 256, "chunk_count": 0}))
    with gzip.open(tmp_path / "chunks.jsonl.gz", "wt") as stream:
        stream.write("")
    monkeypatch.setenv("VP_RERANKER", "claude")
    monkeypatch.setattr(pipeline, "LAYER_RAG_PATH", tmp_path)
    bb = bb_with_hypothesis()
    from pennsieve_ai_utils.blackboard import DatasetSummary
    bb.dataset_summaries = [DatasetSummary(id="pennsieve:405", title="t", description="",
                                           measurements=[], species="", n_subjects=0, doi="10.1/seed")]
    client = object()
    store = pipeline._open_rag_store(bb, client)
    assert store.llm_client is client and store.reranker.enabled
