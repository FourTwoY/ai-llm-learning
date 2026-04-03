# Rewrite Comparison Examples

## Example 1

- 原问题：帮我看一下 RAG 里面 chunk 是怎么切的
- 改写后 query：RAG chunk 切分方式
- 无 rewrite 的检索简述：容易把“帮我看一下”这类口语一起带进检索，命中不够集中
- 有 rewrite 的检索简述：关键词更聚焦在 `RAG`、`chunk`、`切分`
- 简单结论：更好

## Example 2

- 原问题：你能不能帮我找一下 query rewrite 在这个项目里是干嘛的
- 改写后 query：query rewrite 作用
- 无 rewrite 的检索简述：会混入“你能不能帮我找一下”这种无效前缀
- 有 rewrite 的检索简述：更像知识库检索词，通常更容易命中 rewrite 相关实现和说明
- 简单结论：更好

## Example 3

- 原问题：我想问一下 embeddings 是怎么准备好的
- 改写后 query：embeddings 准备流程
- 无 rewrite 的检索简述：口语成分多，召回可能分散
- 有 rewrite 的检索简述：更容易命中 `ensure_embeddings_ready`、`load_embeddings`、`build_chunk_embeddings`
- 简单结论：更好

## Example 4

- 原问题：麻烦帮我看看这个 RAG 项目最后怎么生成答案
- 改写后 query：RAG generate_answer 生成答案流程
- 无 rewrite 的检索简述：容易命中泛化的 README 或上下文不强的片段
- 有 rewrite 的检索简述：更容易把检索集中到 `generation_service.py`
- 简单结论：更好

## Example 5

- 原问题：retrieval top_k 默认是多少来着
- 改写后 query：retrieval top_k 默认值
- 无 rewrite 的检索简述：已经比较短，通常也能检索到配置
- 有 rewrite 的检索简述：会略微更规范，但提升不一定特别明显
- 简单结论：略好

## Example 6

- 原问题：请问一下混合检索和普通检索有什么区别
- 改写后 query：混合检索 与 普通检索 区别
- 无 rewrite 的检索简述：可能命中到零散说明，但表达不够聚焦
- 有 rewrite 的检索简述：更有利于命中 hybrid retrieval 相关实现和配置
- 简单结论：更好
