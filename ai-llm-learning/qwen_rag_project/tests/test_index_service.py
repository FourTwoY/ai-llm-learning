from services.index_service import build_chunks


def test_build_chunks_preserves_doc_info_and_generates_chunk_ids():
    docs = [
        {
            "doc_id": "doc1",
            "source": "doc1.md",
            "text": "abcdefghijklmnopqrstuvwxyz"
        }
    ]

    chunks = build_chunks(docs, chunk_size=10, overlap=2)

    assert len(chunks) >= 2

    assert chunks[0]["chunk_id"] == "doc1_chunk_0"
    assert chunks[0]["doc_id"] == "doc1"
    assert chunks[0]["source"] == "doc1.md"

    assert chunks[1]["chunk_id"] == "doc1_chunk_1"
    assert chunks[1]["doc_id"] == "doc1"
    assert chunks[1]["source"] == "doc1.md"

    # 检查有 overlap：后一块前缀应该和前一块尾部有重叠
    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]