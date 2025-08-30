#!/usr/bin/env python3
"""
Demo script showing the enhanced RAG system with graph validation.
Run this script to see the system in action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import RAGSystem
from config import DATA_DIR

def main():
    print("🚀 Enhanced RAG System with Graph Validation Demo")
    print("=" * 60)
    
    # Initialize the system
    print("\n📚 Loading RAG System...")
    rag_system = RAGSystem(data_dir=DATA_DIR)
    
    # Show vector database stats
    print("\n📊 Vector Database Statistics:")
    stats = rag_system.vector_db.get_collection_stats()
    if "error" not in stats:
        print(f"   • Total vectors: {stats['total_vectors']}")
        print(f"   • Unique circulars: {stats['unique_circulars']}")
        print(f"   • Top circulars: {list(stats['circular_distribution'].keys())[:5]}")
    else:
        print(f"   • Error getting stats: {stats['error']}")
    
    # Interactive demo
    print("\n🤖 Ready for questions! (Type 'quit' to exit, 'compare' to see validation comparison)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question.lower() == 'compare':
                demo_comparison()
                continue
                
            if not question:
                continue
            
            print("\n🔍 Searching with graph validation...")
            result = rag_system.query(question, use_graph_validation=True)
            
            print(f"\n💡 Answer: {result['answer']}")
            print(f"\n📚 Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
            
            if result['relevant_documents']:
                print(f"\n📄 Retrieved {len(result['relevant_documents'])} document chunks:")
                for i, doc in enumerate(result['relevant_documents'], 1):
                    circular = doc['metadata'].get('circular_number', 'Unknown')
                    score = doc.get('score', 0)
                    print(f"   {i}. Circular {circular} (score: {score:.3f})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def demo_comparison():
    """Demo comparing standard vs graph-validated search."""
    print("\n" + "=" * 60)
    print("🔬 Comparison: Standard vs Graph-Validated Search")
    print("=" * 60)
    
    rag_system = RAGSystem(data_dir=DATA_DIR)
    
    demo_question = "Give me a summary of maternity leave policy changes over the years"
    print(f"\n📝 Demo Question: '{demo_question}'")
    
    # Standard search
    print("\n📄 Standard Search Results:")
    print("-" * 30)
    try:
        result_standard = rag_system.query(demo_question, use_graph_validation=False)
        print(f"Answer: {result_standard['answer'][:200]}...")
        
        if result_standard['relevant_documents']:
            circulars = []
            for doc in result_standard['relevant_documents']:
                circular = doc['metadata'].get('circular_number', 'Unknown')
                date = doc['metadata'].get('effective_date', 'Unknown')
                circulars.append(f"{circular} ({date})")
            print(f"Documents found: {', '.join(set(circulars))}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Graph-validated search
    print("\n🔍 Graph-Validated Search Results:")
    print("-" * 35)
    try:
        result_validated = rag_system.query(demo_question, use_graph_validation=True)
        print(f"Answer: {result_validated['answer'][:200]}...")
        
        if result_validated['relevant_documents']:
            current_docs = []
            historical_docs = []
            
            for doc in result_validated['relevant_documents']:
                circular = doc['metadata'].get('circular_number', 'Unknown')
                date = doc['metadata'].get('effective_date', 'Unknown')
                is_superseded = doc['metadata'].get('is_superseded', False)
                
                doc_info = f"{circular} ({date})"
                if is_superseded:
                    historical_docs.append(doc_info)
                else:
                    current_docs.append(doc_info)
            
            if current_docs:
                print(f"Current documents: {', '.join(set(current_docs))}")
            if historical_docs:
                print(f"Historical documents: {', '.join(set(historical_docs))}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n💡 The graph-validated search now includes BOTH:")
    print("   ✅ Current, effective documents for latest policies")
    print("   📚 Historical documents for policy evolution context")
    print("   🎯 This provides complete answers for historical questions!")

if __name__ == "__main__":
    main()
