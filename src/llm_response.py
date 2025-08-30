from groq import Groq
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class LLMResponseGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_response(self, query: str, context: str, sources: List[str], metadata_list: List[Dict]) -> Dict:
        try:
            prompt = self._build_prompt(query, context, sources, metadata_list)
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=512,
                temperature=0.3
            )
            
            return {
                "answer": response.choices[0].message.content.strip(),
                "sources": sources,
                "metadata": metadata_list
            }
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return {
                "answer": "Error generating response",
                "sources": sources,
                "metadata": metadata_list
            }

    def _build_prompt(self, query: str, context: str, sources: List[str], metadata_list: List[Dict]) -> str:
        # Separate current and historical documents
        current_docs = []
        historical_docs = []
        
        for meta in metadata_list:
            doc_info = f"\nDocument from {meta['source_file']}:\n"
            if meta.get('circular_number'):
                doc_info += f"Circular Number: {meta['circular_number']}\n"
            if meta.get('title'):
                doc_info += f"Title: {meta['title']}\n"
            if meta.get('issued_date'):
                doc_info += f"Issued Date: {meta['issued_date']}\n"
            if meta.get('effective_date'):
                doc_info += f"Effective Date: {meta['effective_date']}\n"
            
            # Check if document is superseded
            is_superseded = meta.get('is_superseded', False)
            if is_superseded:
                doc_info += f"Status: SUPERSEDED/HISTORICAL\n"
                historical_docs.append(doc_info)
            else:
                doc_info += f"Status: CURRENT/EFFECTIVE\n"
                current_docs.append(doc_info)
            
            # Add relationship information
            if meta.get('document_relationships'):
                relationships = meta['document_relationships']
                if relationships.get('repeals') and relationships['repeals']:
                    doc_info += f"Repeals: {', '.join(relationships['repeals'])}\n"
                if relationships.get('amends') and relationships['amends']:
                    doc_info += f"Amends: {', '.join(relationships['amends'])}\n"
                if relationships.get('supersedes') and relationships['supersedes']:
                    doc_info += f"Supersedes: {', '.join(relationships['supersedes'])}\n"
                if relationships.get('references') and relationships['references']:
                    doc_info += f"References: {', '.join(relationships['references'])}\n"
        
        # Build metadata sections
        current_metadata = "".join(current_docs) if current_docs else "None available"
        historical_metadata = "".join(historical_docs) if historical_docs else "None available"
        
        return f"""
        Answer the question based on the context provided below. You have access to both current and historical documents.

        CURRENT/EFFECTIVE DOCUMENTS (prioritize for current policies):
        {current_metadata}

        HISTORICAL/SUPERSEDED DOCUMENTS (use for historical context and evolution):
        {historical_metadata}

        Context: {context}
        
        Question: {query}
        
        Instructions:
        1. For questions about current policies, prioritize CURRENT/EFFECTIVE documents
        2. For questions about historical policies or policy evolution, use HISTORICAL documents as well
        3. When showing policy changes over time, clearly indicate the timeframe and status of each policy
        4. Always cite the circular number, source file, effective date, and document status
        5. If a document is superseded, mention what replaced it if available
        
        Provide a comprehensive answer that addresses the question appropriately using both current and historical context where relevant.
        """
