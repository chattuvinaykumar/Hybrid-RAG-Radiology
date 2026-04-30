from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.texts = [d["findings"] for d in documents]

        embeddings = self.model.encode(self.texts)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, top_k=5):
        query_vec = self.model.encode([query])
        _, indices = self.index.search(
            np.array(query_vec).astype("float32"),
            top_k
        )

        return [self.texts[i] for i in indices[0]]
