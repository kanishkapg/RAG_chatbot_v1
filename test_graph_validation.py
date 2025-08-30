#!/usr/bin/env python3
"""
Test script to demonstrate the graph-validated retrieval functionality.
This script compares the results between standard search and graph-validated search.
"""

import logging
from src.rag_system import RAGSystem
from config import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_graph_validation():
    """Test the graph validation functionality."""
    
    print("=" * 80)
    print("Testing Graph-Validated Document Retrieval")
    print("=" * 80)
    
    # Initialize RAG system
    print("\n1. Initializing RAG System...")
    rag_system = RAGSystem(data_dir=DATA_DIR)
    
    # Test queries
    test_queries = [
        "What are the working hours for employees?",
        "What is the policy on leave applications?", 
        "Give me a summary of maternity leave policy after 2021 on each year?",
        "How has the dress code policy evolved over time?",
        "What was the original working hours policy and how has it changed?",
        "What are the current performance evaluation requirements?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 60)
        
        # Test with graph validation
        print("\n🔍 With Graph Validation:")
        try:
            result_validated = rag_system.query(query, use_graph_validation=True)
            print(f"Answer: {result_validated['answer'][:200]}...")
            print(f"Sources: {', '.join(result_validated['sources'])}")
            print(f"Number of chunks: {len(result_validated['relevant_documents'])}")
            
            # Show circular numbers of retrieved documents
            circular_numbers = set()
            for doc in result_validated['relevant_documents']:
                if 'circular_number' in doc['metadata']:
                    circular_numbers.add(doc['metadata']['circular_number'])
            if circular_numbers:
                print(f"Circular Numbers: {', '.join(sorted(circular_numbers))}")
                
        except Exception as e:
            print(f"Error with validation: {e}")
        
        # Test without graph validation
        print("\n📄 Without Graph Validation:")
        try:
            result_standard = rag_system.query(query, use_graph_validation=False)
            print(f"Answer: {result_standard['answer'][:200]}...")
            print(f"Sources: {', '.join(result_standard['sources'])}")
            print(f"Number of chunks: {len(result_standard['relevant_documents'])}")
            
            # Show circular numbers of retrieved documents
            circular_numbers = set()
            for doc in result_standard['relevant_documents']:
                if 'circular_number' in doc['metadata']:
                    circular_numbers.add(doc['metadata']['circular_number'])
            if circular_numbers:
                print(f"Circular Numbers: {', '.join(sorted(circular_numbers))}")
                
        except Exception as e:
            print(f"Error without validation: {e}")
        
        print()

def test_effectiveness_check():
    """Test individual document effectiveness checking."""
    
    print("\n" + "=" * 80)
    print("Testing Document Effectiveness Check")
    print("=" * 80)
    
    rag_system = RAGSystem(data_dir=DATA_DIR)
    
    # Test some circular numbers (you may need to adjust these based on your data)
    test_circulars = ["01/2023", "05/2022", "12/2021", "19/2020", "26/2018"]
    
    for circular in test_circulars:
        print(f"\nChecking circular: {circular}")
        try:
            status = rag_system.find_effective_circular(circular)
            if status:
                if status["is_effective"]:
                    print(f"✅ {circular} is still effective")
                else:
                    print(f"❌ {circular} has been replaced by: {', '.join(status['replaced_by'])}")
            else:
                print(f"❓ {circular} not found in graph database")
        except Exception as e:
            print(f"Error checking {circular}: {e}")

if __name__ == "__main__":
    try:
        test_graph_validation()
        test_effectiveness_check()
        print("\n" + "=" * 80)
        print("✅ Testing completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
