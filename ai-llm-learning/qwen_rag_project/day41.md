## 新增接口：POST /rebuild_index

这个接口用于重建本地知识库索引。

### 功能
- 重新读取 `data/raw` 下的原始文档
- 重新切分文本
- 重新生成 embedding
- 刷新本地索引文件

### 使用场景
当你往知识库中新增或修改文档后，可以调用这个接口来更新索引，而不需要手动逐步运行多个脚本。

### 返回示例

```json
{
  "message": "索引重建成功",
  "doc_count": 3,
  "chunk_count": 28,
  "embedding_count": 28,
  "processed_file": "data/processed/docs.json",
  "chunks_file": "data/chunks/chunks.json",
  "embeddings_file": "data/embeddings/all_embeddings.json"
}