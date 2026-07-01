from eval.retrieval_rankers import (
    BM25Retriever,
    reciprocal_rank_fusion,
)

from eval.schemas import RetrievalResult


def test_bm25_returns_result():
    docs = {
        "doc1": "dream rag retrieval evaluation",
        "doc2": "cats and dogs",
    }

    retriever = BM25Retriever(docs)

    result = retriever.search(
        query_id="q1",
        query="dream retrieval",
        top_k=2,
    )

    assert result.query_id == "q1"
    assert len(result.retrieved_ids) > 0
    assert result.retrieved_ids[0] == "doc1"


def test_rrf_fuses_rankings():
    r1 = RetrievalResult(
        query_id="q1",
        retrieved_ids=["a", "b", "c"],
    )

    r2 = RetrievalResult(
        query_id="q1",
        retrieved_ids=["a", "c", "d"],
    )

    fused = reciprocal_rank_fusion(
        query_id="q1",
        ranked_results=[r1, r2],
        top_k=3,
    )

    assert fused.query_id == "q1"
    assert len(fused.retrieved_ids) == 3
    assert fused.retrieved_ids[0] == "a"