"""
Core RAG System for Customer Support
Handles knowledge base processing, vector storage, and retrieval
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for customer support
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "customer_support_kb"
    ):
        """
        Initialize the RAG system
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_model_name = embedding_model_name
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Load knowledge base if it exists
        self.knowledge_base = []
        self._load_knowledge_base()
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.chroma_persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_directory,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support knowledge base"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _load_knowledge_base(self):
        """Load knowledge base from JSON file"""
        kb_path = Path("data/knowledge_base.json")
        if kb_path.exists():
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded {len(self.knowledge_base)} articles from knowledge base")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                self.knowledge_base = []
        else:
            logger.warning(f"Knowledge base file not found: {kb_path}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the knowledge base and vector store
        
        Args:
            documents: List of document dictionaries with keys like 'title', 'content', 'category', etc.
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        try:
            # Prepare documents for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f"doc_{len(self.knowledge_base) + i}")
                
                # Combine title and content for better retrieval
                text = f"Title: {doc.get('title', '')}\n\nContent: {doc.get('content', '')}"
                
                # Prepare metadata
                metadata = {
                    'title': doc.get('title', ''),
                    'category': doc.get('category', 'general'),
                    'tags': str(doc.get('tags', [])),
                    'type': doc.get('type', 'article'),
                    'priority': doc.get('priority', 'medium')
                }
                
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update local knowledge base
            self.knowledge_base.extend(documents)
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_category: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_category: Optional category filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of relevant documents with similarity scores
        """
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filter_category:
                where_clause = {"category": filter_category}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i] if include_metadata else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_context(
        self,
        query: str,
        max_context_length: int = 2000,
        n_results: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get context for RAG response generation
        
        Args:
            query: User query
            max_context_length: Maximum length of context text
            n_results: Number of documents to retrieve
            
        Returns:
            Tuple of (context_text, source_documents)
        """
        # Search for relevant documents
        results = self.search(query, n_results=n_results)
        
        if not results:
            return "No relevant information found in knowledge base.", []
        
        # Build context from top results
        context_parts = []
        total_length = 0
        source_docs = []
        
        for result in results:
            content = result['content']
            metadata = result['metadata']
            
            # Add document info
            doc_info = f"Source: {metadata.get('title', 'Unknown')} (Category: {metadata.get('category', 'General')})\n"
            doc_content = f"{doc_info}{content}\n\n"
            
            # Check if adding this document would exceed max length
            if total_length + len(doc_content) > max_context_length:
                # Add truncated version if possible
                remaining_space = max_context_length - total_length
                if remaining_space > 100:  # Only add if meaningful space left
                    truncated_content = doc_content[:remaining_space-3] + "..."
                    context_parts.append(truncated_content)
                break
            
            context_parts.append(doc_content)
            total_length += len(doc_content)
            
            # Add to source documents
            source_docs.append({
                'title': metadata.get('title', 'Unknown'),
                'category': metadata.get('category', 'General'),
                'score': result['score'],
                'content_preview': content[:200] + "..." if len(content) > 200 else content
            })
        
        context_text = "".join(context_parts)
        
        return context_text, source_docs
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base collection"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze categories
            sample_results = self.collection.query(
                query_texts=["general"],
                n_results=min(100, count) if count > 0 else 0,
                include=['metadatas']
            )
            
            categories = {}
            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                for metadata in sample_results['metadatas'][0]:
                    category = metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
            
            return {
                'total_documents': count,
                'categories': categories,
                'embedding_model': self.embedding_model_name,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def reset_collection(self):
        """Reset the ChromaDB collection (useful for development)"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support knowledge base"}
            )
            self.knowledge_base = []
            logger.info("Collection reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def update_document(self, doc_id: str, updated_content: Dict[str, Any]):
        """Update a specific document in the knowledge base"""
        try:
            # Prepare updated text and metadata
            text = f"Title: {updated_content.get('title', '')}\n\nContent: {updated_content.get('content', '')}"
            metadata = {
                'title': updated_content.get('title', ''),
                'category': updated_content.get('category', 'general'),
                'tags': str(updated_content.get('tags', [])),
                'type': updated_content.get('type', 'article'),
                'priority': updated_content.get('priority', 'medium')
            }
            
            # Update in ChromaDB
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated document: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise 