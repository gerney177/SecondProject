import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import traceback

# Import the main service
from investment_advisor_service import InvestmentAdvisorService
from financial_data_service import FinancialDataService


class InvestmentAdvisorAPI:
    """
    Flask-based REST API server for Investment Advisor Service.
    Provides endpoints for investment advice, knowledge management, and service statistics.
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False,
                 enable_gpu: bool = False):
        """
        Initialize the Investment Advisor API server.
        
        Args:
            host (str): Host address to bind the server
            port (int): Port number for the server
            debug (bool): Enable Flask debug mode
            enable_gpu (bool): Enable GPU for LLM processing
        """
        self.host = host
        self.port = port
        self.debug = debug
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['JSON_AS_ASCII'] = False  # Support Korean characters
        
        # Enable CORS for all routes
        CORS(self.app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Initialize the investment advisor service
        try:
            self.advisor_service = InvestmentAdvisorService(enable_gpu=enable_gpu)
            self.logger.info("Investment Advisor Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Investment Advisor Service: {e}")
            self.advisor_service = None
        
        # Initialize the financial data service
        try:
            self.financial_service = FinancialDataService(mock_mode=True)
            self.logger.info("Financial Data Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Financial Data Service: {e}")
            self.financial_service = None
        
        # Register routes
        self._register_routes()
        
        # Register error handlers
        self._register_error_handlers()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return self._create_response({
                "status": "healthy",
                "service": "Investment Advisor API",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "advisor_service": self.advisor_service is not None,
                    "financial_service": self.financial_service is not None,
                    "api_server": True
                }
            })
        
        @self.app.route('/api/get-advice', methods=['POST'])
        def get_investment_advice():
            """
            Get investment advice based on provided parameters.
            
            Expected JSON payload:
            {
                "stock_symbol": "삼성전자 (005930)",
                "quantity": 10,
                "price": 70000,
                "strategy": "YouTube URL or investment strategy text",
                "reasoning_effort": "high" (optional)
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                
                # Validate required fields
                required_fields = ["stock_symbol", "quantity", "price", "strategy"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self._create_error_response(
                        f"Missing required fields: {', '.join(missing_fields)}", 400
                    )
                
                # Validate data types
                try:
                    stock_symbol = str(data["stock_symbol"])
                    quantity = int(data["quantity"])
                    price = float(data["price"])
                    strategy = str(data["strategy"])
                    reasoning_effort = data.get("reasoning_effort", "high")
                except (ValueError, TypeError) as e:
                    return self._create_error_response(
                        f"Invalid data type in request: {str(e)}", 400
                    )
                
                # Validate ranges
                if quantity <= 0:
                    return self._create_error_response("Quantity must be positive", 400)
                
                if price <= 0:
                    return self._create_error_response("Price must be positive", 400)
                
                # Generate investment advice
                self.logger.info(f"Generating advice for {stock_symbol}, {quantity} shares at {price}")
                
                advice_result = self.advisor_service.generate_investment_advice(
                    stock_symbol=stock_symbol,
                    quantity=quantity,
                    price=price,
                    strategy=strategy,
                    reasoning_effort=reasoning_effort
                )
                
                return self._create_response(advice_result)
                
            except Exception as e:
                self.logger.error(f"Error in get_investment_advice: {e}")
                self.logger.error(traceback.format_exc())
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/add-youtube-knowledge', methods=['POST'])
        def add_youtube_knowledge():
            """
            Add knowledge from YouTube video to the knowledge base.
            
            Expected JSON payload:
            {
                "youtube_url": "https://www.youtube.com/watch?v=...",
                "language_preference": ["ko", "en"] (optional)
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                
                if "youtube_url" not in data:
                    return self._create_error_response("Missing 'youtube_url' field", 400)
                
                youtube_url = str(data["youtube_url"])
                language_preference = data.get("language_preference", ["ko", "en"])
                
                # Validate YouTube URL
                if not ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
                    return self._create_error_response("Invalid YouTube URL", 400)
                
                self.logger.info(f"Adding YouTube knowledge: {youtube_url}")
                
                result = self.advisor_service.add_youtube_knowledge(
                    youtube_url=youtube_url,
                    language_preference=language_preference
                )
                
                if result["success"]:
                    return self._create_response(result)
                else:
                    return self._create_error_response(
                        result.get("error", "Failed to process YouTube video"), 400
                    )
                
            except Exception as e:
                self.logger.error(f"Error in add_youtube_knowledge: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/search-knowledge', methods=['POST'])
        def search_knowledge():
            """
            Search the knowledge base.
            
            Expected JSON payload:
            {
                "query": "search query",
                "top_k": 5 (optional),
                "search_type": "ensemble" (optional: "ensemble", "dense", "bm25")
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                
                if "query" not in data:
                    return self._create_error_response("Missing 'query' field", 400)
                
                query = str(data["query"])
                top_k = int(data.get("top_k", 5))
                search_type = data.get("search_type", "ensemble")
                
                # Validate parameters
                if top_k <= 0 or top_k > 50:
                    return self._create_error_response("top_k must be between 1 and 50", 400)
                
                if search_type not in ["ensemble", "dense", "bm25"]:
                    return self._create_error_response(
                        "search_type must be 'ensemble', 'dense', or 'bm25'", 400
                    )
                
                self.logger.info(f"Searching knowledge: '{query}' (top_k={top_k}, type={search_type})")
                
                results = self.advisor_service.search_knowledge(
                    query=query,
                    top_k=top_k,
                    search_type=search_type
                )
                
                return self._create_response({
                    "query": query,
                    "search_type": search_type,
                    "results_count": len(results),
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in search_knowledge: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/knowledge-stats', methods=['GET'])
        def get_knowledge_stats():
            """Get knowledge base statistics."""
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                stats = self.advisor_service.get_knowledge_base_stats()
                return self._create_response(stats)
                
            except Exception as e:
                self.logger.error(f"Error in get_knowledge_stats: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/clear-knowledge', methods=['POST'])
        def clear_knowledge():
            """Clear all knowledge from the knowledge base."""
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                # Optional: Add authentication check here
                # if not self._is_authorized(request):
                #     return self._create_error_response("Unauthorized", 401)
                
                self.logger.info("Clearing knowledge base")
                
                result = self.advisor_service.clear_knowledge_base()
                
                if result["success"]:
                    return self._create_response(result)
                else:
                    return self._create_error_response(
                        result.get("error", "Failed to clear knowledge base"), 500
                    )
                
            except Exception as e:
                self.logger.error(f"Error in clear_knowledge: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/service-info', methods=['GET'])
        def get_service_info():
            """Get information about the service and its components."""
            try:
                info = {
                    "service": "Investment Advisor API",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "api_server": {
                            "status": "running",
                            "host": self.host,
                            "port": self.port,
                            "debug": self.debug
                        },
                        "advisor_service": {
                            "available": self.advisor_service is not None,
                            "components": {}
                        }
                    },
                    "endpoints": [
                        "GET /api/health",
                        "POST /api/get-advice",
                        "POST /api/add-youtube-knowledge",
                        "POST /api/search-knowledge",
                        "GET /api/knowledge-stats",
                        "POST /api/clear-knowledge",
                        "GET /api/service-info",
                        "GET /api/stock/price",
                        "GET /api/stock/analysis",
                        "GET /api/stock/chart",
                        "GET /api/market/sentiment",
                        "POST /api/investment/context",
                        "POST /api/stocks/multiple-analysis"
                    ]
                }
                
                if self.advisor_service:
                    stats = self.advisor_service.get_knowledge_base_stats()
                    info["components"]["advisor_service"]["components"] = stats.get("service_info", {})
                
                return self._create_response(info)
                
            except Exception as e:
                self.logger.error(f"Error in get_service_info: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        # ===== 금융 데이터 API 엔드포인트 =====
        
        @self.app.route('/api/stock/price', methods=['GET'])
        def get_stock_price():
            """
            실시간 주가 조회
            
            Query Parameters:
            - symbol: 종목코드 또는 종목명 (예: "005930" 또는 "삼성전자")
            """
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                symbol = request.args.get('symbol')
                if not symbol:
                    return self._create_error_response(
                        "Missing required parameter: symbol", 400
                    )
                
                price_data = self.financial_service.get_real_time_price(symbol)
                if not price_data:
                    return self._create_error_response(
                        f"Stock not found or data unavailable: {symbol}", 404
                    )
                
                return self._create_response({
                    "stock_price": price_data,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_stock_price: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/stock/analysis', methods=['GET'])
        def get_stock_analysis():
            """
            종목 분석 조회
            
            Query Parameters:
            - symbol: 종목코드 또는 종목명
            """
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                symbol = request.args.get('symbol')
                if not symbol:
                    return self._create_error_response(
                        "Missing required parameter: symbol", 400
                    )
                
                analysis = self.financial_service.get_stock_analysis(symbol)
                if not analysis:
                    return self._create_error_response(
                        f"Analysis not available for: {symbol}", 404
                    )
                
                # Convert dataclass to dict
                from dataclasses import asdict
                analysis_dict = asdict(analysis)
                
                return self._create_response({
                    "stock_analysis": analysis_dict,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_stock_analysis: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/stock/chart', methods=['GET'])
        def get_stock_chart():
            """
            차트 데이터 조회
            
            Query Parameters:
            - symbol: 종목코드 또는 종목명
            - period: 조회 기간 (일수, 기본값: 30)
            """
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                symbol = request.args.get('symbol')
                if not symbol:
                    return self._create_error_response(
                        "Missing required parameter: symbol", 400
                    )
                
                period = int(request.args.get('period', 30))
                if period < 1 or period > 365:
                    return self._create_error_response(
                        "Period must be between 1 and 365 days", 400
                    )
                
                chart_data = self.financial_service.get_chart_analysis(symbol, period)
                if not chart_data:
                    return self._create_error_response(
                        f"Chart data not available for: {symbol}", 404
                    )
                
                return self._create_response({
                    "chart_analysis": chart_data,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_stock_chart: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/market/sentiment', methods=['GET'])
        def get_market_sentiment():
            """시장 심리 분석 조회"""
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                sentiment = self.financial_service.get_market_sentiment()
                if not sentiment:
                    return self._create_error_response(
                        "Market sentiment data not available", 503
                    )
                
                # Convert dataclass to dict
                from dataclasses import asdict
                sentiment_dict = asdict(sentiment)
                
                return self._create_response({
                    "market_sentiment": sentiment_dict,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_market_sentiment: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/investment/context', methods=['POST'])
        def get_investment_context():
            """
            투자 컨텍스트 조회
            
            Expected JSON payload:
            {
                "symbol": "005930"
            }
            """
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                data = request.get_json()
                if not data or 'symbol' not in data:
                    return self._create_error_response(
                        "Missing required field: symbol", 400
                    )
                
                symbol = data['symbol']
                context = self.financial_service.get_investment_context(symbol)
                if not context:
                    return self._create_error_response(
                        f"Investment context not available for: {symbol}", 404
                    )
                
                return self._create_response({
                    "investment_context": context,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_investment_context: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/stocks/multiple-analysis', methods=['POST'])
        def get_multiple_stocks_analysis():
            """
            다중 종목 분석
            
            Expected JSON payload:
            {
                "symbols": ["005930", "000660", "035420"]
            }
            """
            try:
                if not self.financial_service:
                    return self._create_error_response(
                        "Financial Data Service not available", 503
                    )
                
                data = request.get_json()
                if not data or 'symbols' not in data:
                    return self._create_error_response(
                        "Missing required field: symbols", 400
                    )
                
                symbols = data['symbols']
                if not isinstance(symbols, list) or len(symbols) == 0:
                    return self._create_error_response(
                        "symbols must be a non-empty list", 400
                    )
                
                if len(symbols) > 10:
                    return self._create_error_response(
                        "Maximum 10 symbols allowed", 400
                    )
                
                analysis = self.financial_service.get_multiple_stocks_analysis(symbols)
                
                return self._create_response({
                    "multiple_analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in get_multiple_stocks_analysis: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
    
    def _register_error_handlers(self):
        """Register error handlers for the Flask app."""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return self._create_error_response("Endpoint not found", 404)
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return self._create_error_response("Method not allowed", 405)
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return self._create_error_response("Internal server error", 500)
    
    def _create_response(self, data: Dict[str, Any], status_code: int = 200) -> Any:
        """
        Create a standardized API response.
        
        Args:
            data (Dict[str, Any]): Response data
            status_code (int): HTTP status code
            
        Returns:
            Flask response object
        """
        response = make_response(jsonify(data), status_code)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    
    def _create_error_response(self, message: str, status_code: int) -> Any:
        """
        Create a standardized error response.
        
        Args:
            message (str): Error message
            status_code (int): HTTP status code
            
        Returns:
            Flask response object
        """
        error_data = {
            "error": True,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        return self._create_response(error_data, status_code)
    
    def run(self):
        """Start the Flask development server."""
        try:
            self.logger.info(f"Starting Investment Advisor API server on {self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise


def main():
    """Main function to start the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investment Advisor API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--enable-gpu', action='store_true', help='Enable GPU for LLM')
    
    args = parser.parse_args()
    
    # Create and start the API server
    api_server = InvestmentAdvisorAPI(
        host=args.host,
        port=args.port,
        debug=args.debug,
        enable_gpu=args.enable_gpu
    )
    
    print(f"""
=== Investment Advisor API Server ===
Server starting on: http://{args.host}:{args.port}
Debug mode: {args.debug}
GPU enabled: {args.enable_gpu}

Available endpoints:
- GET  /api/health              - Health check
- POST /api/get-advice          - Get investment advice
- POST /api/add-youtube-knowledge - Add YouTube knowledge
- POST /api/search-knowledge    - Search knowledge base
- GET  /api/knowledge-stats     - Get knowledge statistics
- POST /api/clear-knowledge     - Clear knowledge base
- GET  /api/service-info        - Get service information

Example usage:
curl -X POST http://localhost:5000/api/get-advice \\
  -H "Content-Type: application/json" \\
  -d '{
    "stock_symbol": "005930",
    "quantity": 10,
    "price": 70000,
    "strategy": "YouTube link or investment strategy"
  }'

Press Ctrl+C to stop the server.
    """)
    
    try:
        api_server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()
