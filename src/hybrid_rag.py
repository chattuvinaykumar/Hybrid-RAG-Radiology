def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    scores = {}

    # FAISS scores
    for rank, doc in enumerate(faiss_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # BM25 scores
    for rank, doc in enumerate(bm25_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked]
