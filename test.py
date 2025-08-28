"""Thin runner that uses refactored modules under src/ to run the same pipeline as before."""

import logging
from src.rag_system import build_and_initialize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    rag_system = build_and_initialize()

    query = "Give me a summary of maternity leave policy after 2021 on each year?"
    result = rag_system.query(query)

    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
        print(f"\nID: {doc['id']}, Score: {doc['score']:.4f}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"Metadata: {doc['metadata']}")

    # Test graph functionality - find if a circular is still effective
    print("\nTesting graph database functionality:")
    # Let's check if Circular No. 10/2022 is still effective
    circular_status = rag_system.find_effective_circular("10/2022")
    if circular_status:
        if circular_status["is_effective"]:
            print(f"Circular {circular_status['original']} is still effective")
        else:
            print(f"Circular {circular_status['original']} has been replaced by: {', '.join(circular_status['replaced_by'])}")
    else:
        print("Circular not found in the database")

    # Generate visualization
    viz_path = rag_system.generate_visualization()
    print(f"\nDocument relationship visualization generated at: {viz_path}")


if __name__ == "__main__":
    main()