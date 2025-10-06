import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import custom modules
from youtube_subtitle_extractor import YouTubeSubtitleExtractor
from text_chunker import TextChunker
from vector_database import EnsembleVectorDatabase
from llm_service import GPTOSSService
from financial_data_service import FinancialDataService


class InvestmentAdvisorService:
    """
    Main investment advisor service that integrates all components.
    Provides comprehensive investment advice using RAG (Retrieval Augmented Generation).
    """
    
    def __init__(self, 
                 collection_name: str = "investment_knowledge",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 bm25_weight: float = 0.4,
                 dense_weight: float = 0.6,
                 enable_gpu: bool = False):
        """
        Initialize Investment Advisor Service.
        
        Args:
            collection_name (str): ChromaDB collection name
            chunk_size (int): Text chunk size for processing
            chunk_overlap (int): Overlap between chunks
            bm25_weight (float): Weight for BM25 search
            dense_weight (float): Weight for dense search
            enable_gpu (bool): Whether to enable GPU for LLM
        """
        self.collection_name = collection_name
        self.enable_gpu = enable_gpu
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing Investment Advisor Service components...")
        
        # YouTube subtitle extractor
        self.subtitle_extractor = YouTubeSubtitleExtractor()
        
        # Text chunker
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector database
        self.vector_db = EnsembleVectorDatabase(
            collection_name=collection_name,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight
        )
        
        # LLM service (initialize with caution for GPU)
        self.llm_service = None
        self._initialize_llm_service()
        
        # Financial data service
        self.financial_service = None
        self._initialize_financial_service()
        
        self.logger.info("Investment Advisor Service initialized successfully")
    
    def _initialize_llm_service(self):
        """Initialize LLM service with error handling."""
        try:
            if self.enable_gpu:
                self.llm_service = GPTOSSService()
                self.logger.info("LLM service initialized with GPU support")
            else:
                # For CPU or mock testing
                self.logger.warning("GPU disabled - LLM service will use mock responses")
                self.llm_service = None
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            self.llm_service = None
    
    def _initialize_financial_service(self):
        """Initialize Financial Data service with error handling."""
        try:
            self.financial_service = FinancialDataService(mock_mode=True)
            self.logger.info("Financial Data Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Financial Data Service: {e}")
            self.financial_service = None
    
    def add_youtube_knowledge(self, 
                            youtube_url: str,
                            language_preference: List[str] = ['ko', 'en']) -> Dict[str, Any]:
        """
        Add knowledge from YouTube video to the database.
        
        Args:
            youtube_url (str): YouTube video URL
            language_preference (List[str]): Preferred languages for subtitles
            
        Returns:
            Dict[str, Any]: Processing result with statistics
        """
        try:
            self.logger.info(f"Processing YouTube video: {youtube_url}")
            
            # Extract subtitles
            subtitle_data = self.subtitle_extractor.extract_subtitles(
                youtube_url, 
                language_preference
            )
            
            if not subtitle_data:
                return {
                    "success": False,
                    "error": "Failed to extract subtitles from YouTube video",
                    "url": youtube_url
                }
            
            # Chunk the subtitle text
            chunks = self.text_chunker.create_semantic_chunks(
                subtitle_data['subtitle_text'],
                language='korean' if subtitle_data['language'] == 'ko' else 'english'
            )
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create text chunks",
                    "url": youtube_url
                }
            
            # Prepare documents and metadata for vector database
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": "youtube",
                    "video_id": subtitle_data['video_id'],
                    "video_url": subtitle_data['video_url'],
                    "language": subtitle_data['language'],
                    "chunk_id": i,
                    "chunk_token_count": chunk['token_count'],
                    "chunk_sentence_count": chunk['sentence_count'],
                    "total_video_duration": subtitle_data['total_duration'],
                    "timestamp": datetime.now().isoformat(),
                    "content_type": "investment_strategy"
                }
                metadatas.append(metadata)
            
            # Add to vector database
            success = self.vector_db.add_documents(documents, metadatas)
            
            if success:
                stats = self.text_chunker.get_chunk_statistics(chunks)
                return {
                    "success": True,
                    "video_info": {
                        "video_id": subtitle_data['video_id'],
                        "video_url": subtitle_data['video_url'],
                        "language": subtitle_data['language'],
                        "duration": subtitle_data['total_duration'],
                        "word_count": subtitle_data['word_count']
                    },
                    "processing_stats": stats,
                    "chunks_added": len(chunks),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to add chunks to vector database",
                    "url": youtube_url
                }
                
        except Exception as e:
            self.logger.error(f"Error processing YouTube video: {e}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "url": youtube_url
            }
    
    def search_knowledge(self, 
                        query: str, 
                        top_k: int = 5,
                        search_type: str = "ensemble") -> List[Dict]:
        """
        Search knowledge base for relevant information.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            search_type (str): Type of search ("ensemble", "dense", "bm25")
            
        Returns:
            List[Dict]: Search results
        """
        try:
            if search_type == "ensemble":
                results = self.vector_db.search_ensemble(query, top_k)
            elif search_type == "dense":
                results = self.vector_db.search_dense(query, top_k)
            elif search_type == "bm25":
                results = self.vector_db.search_bm25(query, top_k)
            else:
                self.logger.warning(f"Unknown search type: {search_type}, using ensemble")
                results = self.vector_db.search_ensemble(query, top_k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def generate_investment_advice(self, 
                                 stock_symbol: str,
                                 quantity: int,
                                 price: float,
                                 strategy: str,
                                 reasoning_effort: str = "high") -> Dict[str, Any]:
        """
        Generate comprehensive investment advice using RAG.
        
        Args:
            stock_symbol (str): Stock symbol or company name
            quantity (int): Number of shares to buy
            price (float): Price per share
            strategy (str): Investment strategy query or YouTube URL
            reasoning_effort (str): LLM reasoning effort level
            
        Returns:
            Dict[str, Any]: Comprehensive investment advice
        """
        try:
            self.logger.info(f"Generating investment advice for {stock_symbol}")
            
            # Get real-time financial data
            financial_context = ""
            if self.financial_service:
                try:
                    # Get investment context from financial service
                    investment_context = self.financial_service.get_investment_context(stock_symbol)
                    if investment_context:
                        stock_analysis = investment_context.get("stock_analysis", {})
                        market_sentiment = investment_context.get("market_sentiment", {})
                        investment_signals = investment_context.get("investment_signals", {})
                        risk_assessment = investment_context.get("risk_assessment", {})
                        
                        financial_context = f"""
실시간 금융 데이터 분석:
- 현재가: {stock_analysis.get('current_price', 'N/A')}원
- 등락률: {stock_analysis.get('change_rate', 'N/A')}%
- 거래량: {stock_analysis.get('volume', 'N/A')}주
- RSI: {stock_analysis.get('rsi', 'N/A')}
- 5일 이동평균: {stock_analysis.get('moving_average_5', 'N/A')}원
- 20일 이동평균: {stock_analysis.get('moving_average_20', 'N/A')}원

시장 심리:
- KOSPI 동향: {market_sentiment.get('kospi_trend', 'N/A')}
- KOSDAQ 동향: {market_sentiment.get('kosdaq_trend', 'N/A')}
- 시장 변동성: {market_sentiment.get('market_volatility', 'N/A')}

투자 신호:
- 기술적 신호: {investment_signals.get('technical_signal', 'N/A')}
- 시장 신호: {investment_signals.get('market_signal', 'N/A')}
- 종합 신호: {investment_signals.get('overall_signal', 'N/A')}
- 신뢰도: {investment_signals.get('confidence', 'N/A')}

리스크 평가:
- 전체 리스크: {risk_assessment.get('risk_level', 'N/A')}
- 변동성 리스크: {risk_assessment.get('volatility_risk', 'N/A')}
- 시장 리스크: {risk_assessment.get('market_risk', 'N/A')}
"""
                        self.logger.info("Real-time financial data integrated successfully")
                    else:
                        financial_context = "실시간 금융 데이터를 가져올 수 없습니다."
                        
                except Exception as e:
                    self.logger.error(f"Error getting financial data: {e}")
                    financial_context = f"금융 데이터 조회 중 오류 발생: {str(e)}"
            else:
                financial_context = "금융 데이터 서비스가 사용 불가능합니다."
            
            # Process strategy input
            strategy_info = ""
            
            # Check if strategy is a YouTube URL
            if "youtube.com" in strategy or "youtu.be" in strategy:
                self.logger.info("Strategy input detected as YouTube URL, processing...")
                youtube_result = self.add_youtube_knowledge(strategy)
                
                if youtube_result["success"]:
                    # Search for relevant information
                    search_query = f"{stock_symbol} 투자 전략 분석"
                    search_results = self.search_knowledge(search_query, top_k=5)
                    
                    strategy_info = "\n".join([result['document'] for result in search_results])
                    
                    if not strategy_info:
                        strategy_info = f"YouTube 동영상에서 투자 정보를 추출했지만 {stock_symbol}와 직접적으로 관련된 내용을 찾지 못했습니다."
                else:
                    strategy_info = f"YouTube 동영상 처리 중 오류 발생: {youtube_result.get('error', 'Unknown error')}"
            else:
                # Search existing knowledge base
                search_results = self.search_knowledge(strategy, top_k=5)
                
                if search_results:
                    strategy_info = "\n".join([result['document'] for result in search_results])
                else:
                    strategy_info = strategy  # Use the strategy as-is if no search results
            
            # Generate advice using LLM
            if self.llm_service:
                # Combine strategy info with financial data
                enhanced_strategy_info = f"""
{strategy_info}

{financial_context}
"""
                advice_result = self.llm_service.generate_investment_advice(
                    stock_symbol=stock_symbol,
                    quantity=quantity,
                    price=price,
                    strategy_info=enhanced_strategy_info,
                    reasoning_effort=reasoning_effort
                )
            else:
                # Fallback mock advice with financial data
                advice_result = self._generate_mock_advice(
                    stock_symbol, quantity, price, f"{strategy_info}\n\n{financial_context}"
                )
            
            # Add knowledge base search results to response
            advice_result["knowledge_search"] = {
                "query": strategy if not ("youtube.com" in strategy or "youtu.be" in strategy) else f"{stock_symbol} 투자 전략",
                "results_found": len(search_results) if 'search_results' in locals() else 0,
                "search_type": "ensemble"
            }
            
            # Add service metadata
            advice_result["service_info"] = {
                "service_version": "1.0.0",
                "components_used": {
                    "youtube_extraction": "youtube.com" in strategy or "youtu.be" in strategy,
                    "knowledge_search": True,
                    "llm_generation": self.llm_service is not None,
                    "ensemble_search": True,
                    "financial_data": self.financial_service is not None,
                    "real_time_analysis": bool(financial_context and "실시간 금융 데이터 분석" in financial_context)
                },
                "processing_timestamp": datetime.now().isoformat()
            }
            
            return advice_result
            
        except Exception as e:
            self.logger.error(f"Error generating investment advice: {e}")
            return {
                "error": f"투자 조언 생성 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "service_info": {
                    "error_occurred": True,
                    "llm_available": self.llm_service is not None
                }
            }
    
    def _generate_mock_advice(self, 
                            stock_symbol: str, 
                            quantity: int, 
                            price: float, 
                            strategy_info: str) -> Dict[str, Any]:
        """
        Generate mock investment advice when LLM is not available.
        
        Args:
            stock_symbol (str): Stock symbol
            quantity (int): Number of shares
            price (float): Price per share
            strategy_info (str): Strategy information
            
        Returns:
            Dict[str, Any]: Mock advice result
        """
        total_investment = quantity * price
        
        mock_advice = f"""
**{stock_symbol} 투자 분석 보고서 (Mock Analysis)**

**투자 개요**
- 종목: {stock_symbol}
- 투자 수량: {quantity:,}주
- 주당 가격: {price:,}원
- 총 투자금액: {total_investment:,}원

**분석 근거**
{strategy_info[:500]}...

**투자 권고사항**
1. **기업 분석**: 현재 제공된 정보를 바탕으로 추가 분석이 필요합니다.
2. **시장 전망**: 최신 시장 동향을 고려한 투자 전략 수립을 권장합니다.
3. **리스크 관리**: 분산투자와 손절매 전략을 함께 고려하시기 바랍니다.
4. **투자 기간**: 중장기 관점에서의 투자 접근을 권장합니다.

**주의사항**
- 이 분석은 모의 분석 결과입니다.
- 실제 투자 결정 시에는 전문가의 조언을 받으시기 바랍니다.
- 시장 상황 변화에 따른 지속적인 모니터링이 필요합니다.

**면책 조항**
본 분석은 참고용으로만 사용하시고, 투자 결정에 대한 책임은 투자자 본인에게 있습니다.
        """
        
        return {
            "advice": mock_advice,
            "investment_summary": {
                "stock_symbol": stock_symbol,
                "quantity": quantity,
                "price_per_share": price,
                "total_investment": total_investment,
                "currency": "KRW"
            },
            "generation_params": {
                "reasoning_effort": "mock",
                "llm_service": "mock_service"
            },
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_id": "mock_advisor",
                "version": "demo"
            },
            "warning": "This is a mock analysis. LLM service is not available."
        }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dict[str, Any]: Knowledge base statistics
        """
        try:
            vector_stats = self.vector_db.get_collection_stats()
            
            # Add service-specific stats
            stats = {
                "vector_database": vector_stats,
                "service_info": {
                    "collection_name": self.collection_name,
                    "components_initialized": {
                        "subtitle_extractor": self.subtitle_extractor is not None,
                        "text_chunker": self.text_chunker is not None,
                        "vector_database": self.vector_db is not None,
                        "llm_service": self.llm_service is not None
                    },
                    "search_weights": {
                        "bm25_weight": self.vector_db.bm25_weight,
                        "dense_weight": self.vector_db.dense_weight
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge base stats: {e}")
            return {
                "error": f"통계 조회 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            Dict[str, Any]: Clear operation result
        """
        try:
            success = self.vector_db.clear_collection()
            
            return {
                "success": success,
                "message": "지식 베이스가 성공적으로 초기화되었습니다." if success else "지식 베이스 초기화에 실패했습니다.",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error clearing knowledge base: {e}")
            return {
                "success": False,
                "error": f"지식 베이스 초기화 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


def main():
    """Example usage of InvestmentAdvisorService"""
    print("=== Investment Advisor Service Test ===")
    
    # Initialize service (without GPU for testing)
    advisor = InvestmentAdvisorService(enable_gpu=False)
    
    # Test 1: Add YouTube knowledge (with mock URL)
    print("\n1. Testing YouTube Knowledge Addition...")
    youtube_url = "https://www.youtube.com/watch?v=test123"
    
    # For testing, we'll skip actual YouTube processing
    print(f"Mock: Would process YouTube video: {youtube_url}")
    
    # Test 2: Generate investment advice
    print("\n2. Testing Investment Advice Generation...")
    advice_result = advisor.generate_investment_advice(
        stock_symbol="삼성전자 (005930)",
        quantity=10,
        price=70000,
        strategy="삼성전자 반도체 사업 분석과 AI 칩 시장 전망에 대한 투자 전략"
    )
    
    print("Investment Advice Generated:")
    print(f"- Stock: {advice_result['investment_summary']['stock_symbol']}")
    print(f"- Total Investment: {advice_result['investment_summary']['total_investment']:,} KRW")
    print(f"- Generated at: {advice_result['timestamp']}")
    
    if 'advice' in advice_result:
        print(f"\nAdvice Preview:\n{advice_result['advice'][:300]}...")
    
    # Test 3: Knowledge base statistics
    print("\n3. Testing Knowledge Base Statistics...")
    stats = advisor.get_knowledge_base_stats()
    print("Knowledge Base Stats:")
    for key, value in stats.items():
        if key != 'vector_database':  # Skip detailed vector DB stats for brevity
            print(f"- {key}: {value}")
    
    # Test 4: Search knowledge base
    print("\n4. Testing Knowledge Search...")
    search_results = advisor.search_knowledge("삼성전자 투자 전략", top_k=3)
    print(f"Search Results: {len(search_results)} found")
    
    for i, result in enumerate(search_results[:2], 1):
        print(f"{i}. Score: {result.get('score', 0):.3f}")
        print(f"   Text: {result['document'][:100]}...")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
