from src.preprocess import load_reports
from src.faiss_retrieval import FaissRetriever
from src.bm25_retrieval import BM25Retriever
from src.hybrid_rag import reciprocal_rank_fusion
from src.prompt_engineering import build_prompt


def main():
    print("Loading MIMIC-CXR reports...")
    data = load_reports()

    if len(data) == 0:
        print("No reports found. Check dataset path.")
        return

    print(f"Loaded {len(data)} reports")

    # Build retrievers
    print("Building FAISS index...")
    faiss = FaissRetriever(data)

    print("Building BM25 index...")
    bm25 = BM25Retriever(data)

    # Sample query (you can change this)
    query = "opacity in lungs with no pleural effusion"

    print("\nQuery:", query)

    # Retrieve results
    faiss_results = faiss.retrieve(query)
    bm25_results = bm25.retrieve(query)

    # Hybrid fusion
    hybrid
