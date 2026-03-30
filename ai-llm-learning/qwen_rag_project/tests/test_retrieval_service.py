from services import retrieval_service


def test_retrieve_chunks_returns_top_k_sorted_by_similarity(monkeypatch):
    def fake_embed_texts(texts):
        # 只会传入 query
        return [[1.0, 0.0]], {"usage": None}

    monkeypatch.setattr(retrieval_service, "embed_texts", fake_embed_texts)

    embedded_chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source": "bert.md",
            "text": "chunk 1",
            "embedding": [0.9, 0.1],
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "source": "gpt.md",
            "text": "chunk 2",
            "embedding": [0.1, 0.9],
        },
        {
            "chunk_id": "c3",
            "doc_id": "d3",
            "source": "vit.md",
            "text": "chunk 3",
            "embedding": [0.8, 0.2],
        },
    ]

    results = retrieval_service.retrieve_chunks("BERT", embedded_chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[1]["chunk_id"] == "c3"
    assert results[0]["score"] >= results[1]["score"]