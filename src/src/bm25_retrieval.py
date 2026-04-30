from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.texts = [d["findings"] for d in documents]
        tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [self.texts[i] for i in ranked_indices[:top_k]]
