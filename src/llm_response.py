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
        metadata_info = ""
        for meta in metadata_list:
            metadata_info += f"\nDocument from {meta['source_file']}:\n"
            if meta.get('circular_number'):
                metadata_info += f"Circular Number: {meta['circular_number']}\n"
            if meta.get('title'):
                metadata_info += f"Title: {meta['title']}\n"
            if meta.get('issued_date'):
                metadata_info += f"Issued Date: {meta['issued_date']}\n"
            if meta.get('effective_date'):
                metadata_info += f"Effective Date: {meta['effective_date']}\n"
            if meta.get('document_relationships'):
                relationships = meta['document_relationships']
                if relationships.get('repeals') and relationships['repeals']:
                    metadata_info += f"Repeals: {', '.join(relationships['repeals'])}\n"
                if relationships.get('amends') and relationships['amends']:
                    metadata_info += f"Amends: {', '.join(relationships['amends'])}\n"
                if relationships.get('supersedes') and relationships['supersedes']:
                    metadata_info += f"Supersedes: {', '.join(relationships['supersedes'])}\n"
                if relationships.get('references') and relationships['references']:
                    metadata_info += f"References: {', '.join(relationships['references'])}\n"
                
        return f"""
        Answer the question based on the context provided below. Prioritize information from documents that are effective 
        (not listed in any 'repealed_circulars'). If multiple documents are relevant, prefer the one with the most recent effective_date. 
        Cite the circular number, source file, and effective date in your response.

        Document Metadata:
        {metadata_info}

        Context: {context}
        
        Question: {query}
        
        Provide a concise answer and include citations with circular numbers, source files, and effective dates where applicable.
        """
