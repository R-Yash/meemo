from retrieval.rag import RetrievedChunk

RRF_K = 60    
FINAL_TOP_N = 5     
SEMANTIC_WEIGHT = 0.6   
GRAPH_WEIGHT = 0.4 

def reciprocal_rank_fusion(
    semantic_chunks: list[RetrievedChunk],
    graph_chunks: list[RetrievedChunk],
    top_n: int   = FINAL_TOP_N,
    semantic_weight: float = SEMANTIC_WEIGHT,
    graph_weight:float = GRAPH_WEIGHT,
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(semantic_chunks):
        fused_scores[chunk.chunk_id] = (fused_scores.get(chunk.chunk_id, 0.0) + semantic_weight / (RRF_K + rank + 1))
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(graph_chunks):
        rrf = graph_weight / (RRF_K + rank + 1)
        if chunk.chunk_id in fused_scores:
            fused_scores[chunk.chunk_id] += rrf
            chunk_map[chunk.chunk_id].source = "both"   # confirmed by both paths
        else:
            fused_scores[chunk.chunk_id] = rrf
            chunk_map[chunk.chunk_id] = chunk

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_id, score in ranked[:top_n]:
        chunk       = chunk_map[chunk_id]
        chunk.score = round(score, 6)
        results.append(chunk)

    return results

def explain_ranking(chunks: list[RetrievedChunk]) -> str:
    """Returns a human-readable breakdown of the fused ranking."""
    lines = ["=== Fused Ranking ==="]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] score={c.score:.5f}  source={c.source:<8}  "
            f"doc={c.document_id[:8]}…  title={c.title!r}"
        )
    return "\n".join(lines)