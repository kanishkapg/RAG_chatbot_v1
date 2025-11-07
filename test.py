"""Thin runner that uses refactored modules under src/ to run the same pipeline as before."""

import logging
from src.rag_system import build_and_initialize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    rag_system = build_and_initialize()

    query = "What is the threshold value above which the SLT Board must approve a procurement for Non-Standardized Goods and Services, as per the most recent circular?"
    result = rag_system.query(query)

    print("=" * 100)
    print(f"\nQuestion: {result['question']}")
    print("=" * 100)
    print(f"Answer: {result['answer']}")
    print("=" * 100)
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
     
        print(f"\nSource: {doc['source']}")
        print(f"ID: {doc['id']}")
        print(f"Score: {doc['score']:.4f}")
        print(f"Effective Date: {doc['metadata']['effective_date']}")
        print(f"Issue Date: {doc['metadata']['issued_date']}")
        print(f"Category_Path: {doc['metadata']['category_path']}")
        print("-" * 100)


    # Generate visualization
    viz_path = rag_system.generate_visualization()
    print(f"\nDocument relationship visualization generated at: {viz_path}")


if __name__ == "__main__":
    main()