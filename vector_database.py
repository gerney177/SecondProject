import os
import json
import uuid
from typing import List, Dict, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
from datetime import datetime


class EnsembleVectorDatabase:
    """
    Advanced vector database with ensemble search capabilities.
    Combines BM25 (sparse) and dense vector search with configurable weights.
    Uses ChromaDB for vector storage and retrieval.
    """
    
    def __init__(self, 
                 collection_name: str = "investment_knowledge",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 persist_directory: str = "./chroma_db",
                 bm25_weight: float = 0.4,
                 dense_weight: float = 0.6):
        """
        Initialize EnsembleVectorDatabase.
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            embedding_model (str): Sentence transformer model for embeddings
            persist_directory (str): Directory to persist ChromaDB data
            bm25_weight (float): Weight for BM25 scores (should sum to 1.0 with dense_weight)
            dense_weight (float): Weight for dense search scores
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Ensure weights sum to 1.0
        total_weight = bm25_weight + dense_weight
        self.bm25_weight = bm25_weight / total_weight
        self.dense_weight = dense_weight / total_weight
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self._initialize_chromadb()
        
        # Initialize embedding model
        self._initialize_embedding_model(embedding_model)
        
        # Initialize BM25 (will be loaded when documents are added)
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        
        self.logger.info(f"EnsembleVectorDatabase initialized with weights - BM25: {self.bm25_weight:.2f}, Dense: {self.dense_weight:.2f}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Investment knowledge base with ensemble search"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _initialize_embedding_model(self, model_name: str):
        """Initialize sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            # Fallback to default model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Using fallback embedding model: all-MiniLM-L6-v2")
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> bool:
        """
        Add documents to the vector database and update BM25 index.
        
        Args:
            documents (List[str]): List of document texts
            metadatas (Optional[List[Dict]]): List of metadata dictionaries
            ids (Optional[List[str]]): List of document IDs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                self.logger.warning("No documents provided")
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"timestamp": datetime.now().isoformat()} for _ in documents]
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update BM25 index
            self._update_bm25_index()
            
            self.logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def _update_bm25_index(self):
        """Update BM25 index with all documents in the collection."""
        try:
            # Get all documents from ChromaDB
            result = self.collection.get()
            
            if result['documents']:
                self.documents = result['documents']
                self.document_ids = result['ids']
                
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() for doc in self.documents]
                
                # Create BM25 index
                self.bm25 = BM25Okapi(tokenized_docs)
                
                self.logger.info(f"Updated BM25 index with {len(self.documents)} documents")
            else:
                self.logger.warning("No documents found in collection")
                
        except Exception as e:
            self.logger.error(f"Error updating BM25 index: {e}")
    
    def search_dense(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform dense vector search using embeddings.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of search results with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'search_type': 'dense'
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in dense search: {e}")
            return []
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform BM25 sparse search.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of search results with scores
        """
        try:
            if self.bm25 is None:
                self.logger.warning("BM25 index not available")
                return []
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Format results
            formatted_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    result = {
                        'id': self.document_ids[idx],
                        'document': self.documents[idx],
                        'score': float(scores[idx]),
                        'metadata': {},  # BM25 doesn't have metadata directly
                        'search_type': 'bm25'
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in BM25 search: {e}")
            return []
    
    def search_ensemble(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform ensemble search combining BM25 and dense search.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of search results with combined scores
        """
        try:
            # Perform both searches
            dense_results = self.search_dense(query, top_k * 2)  # Get more results for better ensemble
            bm25_results = self.search_bm25(query, top_k * 2)
            
            # Normalize scores
            dense_results = self._normalize_scores(dense_results)
            bm25_results = self._normalize_scores(bm25_results)
            
            # Combine results
            combined_scores = {}
            
            # Add dense scores
            for result in dense_results:
                doc_id = result['id']
                combined_scores[doc_id] = {
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'dense_score': result['score'],
                    'bm25_score': 0.0,
                    'combined_score': 0.0
                }
            
            # Add BM25 scores
            for result in bm25_results:
                doc_id = result['id']
                if doc_id in combined_scores:
                    combined_scores[doc_id]['bm25_score'] = result['score']
                else:
                    combined_scores[doc_id] = {
                        'document': result['document'],
                        'metadata': result.get('metadata', {}),
                        'dense_score': 0.0,
                        'bm25_score': result['score'],
                        'combined_score': 0.0
                    }
            
            # Calculate combined scores
            for doc_id in combined_scores:
                dense_score = combined_scores[doc_id]['dense_score']
                bm25_score = combined_scores[doc_id]['bm25_score']
                combined_scores[doc_id]['combined_score'] = (
                    self.dense_weight * dense_score + 
                    self.bm25_weight * bm25_score
                )
            
            # Sort by combined score and return top-k
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['combined_score'],
                reverse=True
            )[:top_k]
            
            # Format final results
            final_results = []
            for doc_id, scores in sorted_results:
                result = {
                    'id': doc_id,
                    'document': scores['document'],
                    'score': scores['combined_score'],
                    'dense_score': scores['dense_score'],
                    'bm25_score': scores['bm25_score'],
                    'metadata': scores['metadata'],
                    'search_type': 'ensemble'
                }
                final_results.append(result)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in ensemble search: {e}")
            return []
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            results (List[Dict]): Search results with scores
            
        Returns:
            List[Dict]: Results with normalized scores
        """
        if not results:
            return results
        
        scores = [result['score'] for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result['score'] = 1.0
        else:
            for result in results:
                result['score'] = (result['score'] - min_score) / (max_score - min_score)
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            result = self.collection.get()
            
            stats = {
                'total_documents': len(result['documents']) if result['documents'] else 0,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight,
                'has_bm25_index': self.bm25 is not None
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the collection.
        
        Args:
            ids (List[str]): List of document IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=ids)
            
            # Update BM25 index
            self._update_bm25_index()
            
            self.logger.info(f"Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all document IDs
            result = self.collection.get()
            if result['ids']:
                self.collection.delete(ids=result['ids'])
            
            # Reset BM25
            self.bm25 = None
            self.documents = []
            self.document_ids = []
            
            self.logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False


def main():
    """Example usage of EnsembleVectorDatabase"""
    # Initialize database
    db = EnsembleVectorDatabase(
        collection_name="test_investment_db",
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    # Sample investment-related documents
    sample_documents = [
        "삼성전자는 한국의 대표적인 반도체 회사로 글로벌 시장에서 경쟁력을 가지고 있습니다.",
        "Apple's strong financial performance in Q3 2023 shows promising growth in services revenue.",
        "Tesla의 전기차 시장 점유율이 증가하고 있으며, 자율주행 기술 개발도 지속되고 있습니다.",
        "NVIDIA's AI chip demand continues to surge as companies invest heavily in artificial intelligence.",
        "금리 인상이 부동산 시장에 미치는 영향을 분석해보면 투자 전략을 수립할 수 있습니다.",
        "The Federal Reserve's monetary policy decisions significantly impact stock market volatility.",
        "ESG 투자가 장기적으로 안정적인 수익을 제공할 수 있다는 연구 결과가 나오고 있습니다."
    ]
    
    metadatas = [
        {"source": "korean_market_analysis", "category": "semiconductor"},
        {"source": "earnings_report", "category": "technology"},
        {"source": "market_research", "category": "automotive"},
        {"source": "tech_news", "category": "ai_hardware"},
        {"source": "economic_analysis", "category": "real_estate"},
        {"source": "fed_report", "category": "monetary_policy"},
        {"source": "esg_research", "category": "sustainable_investing"}
    ]
    
    print("=== EnsembleVectorDatabase Test ===")
    
    # Add documents
    print("Adding documents to database...")
    success = db.add_documents(sample_documents, metadatas)
    print(f"Documents added: {success}")
    
    # Get collection stats
    stats = db.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
    
    # Test different search methods
    test_queries = [
        "삼성전자 투자 분석",
        "AI technology investment opportunities",
        "Federal Reserve policy impact"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing Query: '{query}' ===")
        
        # Dense search
        dense_results = db.search_dense(query, top_k=3)
        print(f"\nDense Search Results ({len(dense_results)}):")
        for i, result in enumerate(dense_results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['document'][:100]}...")
        
        # BM25 search
        bm25_results = db.search_bm25(query, top_k=3)
        print(f"\nBM25 Search Results ({len(bm25_results)}):")
        for i, result in enumerate(bm25_results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['document'][:100]}...")
        
        # Ensemble search
        ensemble_results = db.search_ensemble(query, top_k=3)
        print(f"\nEnsemble Search Results ({len(ensemble_results)}):")
        for i, result in enumerate(ensemble_results):
            print(f"{i+1}. Combined: {result['score']:.3f} (Dense: {result['dense_score']:.3f}, BM25: {result['bm25_score']:.3f})")
            print(f"   Text: {result['document'][:100]}...")


if __name__ == "__main__":
    main()
