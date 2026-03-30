from pathlib import Path

from services.document_service import read_raw_documents


def test_read_raw_documents_only_loads_txt_and_md_and_skips_empty(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "bert.md").write_text("  BERT is bidirectional.  ", encoding="utf-8")
    (raw_dir / "gpt.txt").write_text("GPT is autoregressive.", encoding="utf-8")
    (raw_dir / "empty.md").write_text("   ", encoding="utf-8")
    (raw_dir / "ignore.pdf").write_text("should be ignored", encoding="utf-8")

    docs = read_raw_documents(str(raw_dir))

    assert len(docs) == 2

    sources = {doc["source"] for doc in docs}
    assert sources == {"bert.md", "gpt.txt"}

    bert_doc = next(doc for doc in docs if doc["source"] == "bert.md")
    assert bert_doc["doc_id"] == "bert"
    assert bert_doc["text"] == "BERT is bidirectional."

    gpt_doc = next(doc for doc in docs if doc["source"] == "gpt.txt")
    assert gpt_doc["doc_id"] == "gpt"
    assert gpt_doc["text"] == "GPT is autoregressive."