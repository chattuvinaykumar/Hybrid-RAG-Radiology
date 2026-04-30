from src.preprocess import load_reports
from src.faiss_retrieval import FaissRetriever
from src.bm25_retrieval import BM25Retriever
from src.hybrid_rag import reciprocal_rank_fusion
from src.prompt_engineering import build_prompt


def main():
    print("Loading reports...")
    data = load_reports()

    if len(data) == 0:
        print("No data found")
        return

    print(f"Loaded {len(data)} reports")

    faiss = FaissRetriever(data)
    bm25 = BM25Retriever(data)

    query = "lung opacity"

    f_results = faiss.retrieve(query)
    b_results = bm25.retrieve(query)

    hybrid = reciprocal_rank_fusion(f_results, b_results)

    context = "\n".join(hybrid[:3])

    prompt = build_prompt(query, context)

    print("\nGenerated Prompt:\n")
    print(prompt)


if __name__ == "__main__":
    main()
