import os
import torch
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from datetime import datetime
import json


class GPTOSSService:
    """
    LLM service using GPT-OSS model for investment advice generation.
    Provides text generation capabilities with investment-focused prompts.
    """
    
    def __init__(self, 
                 model_id: str = "openai/gpt-oss-20b",
                 device: str = "auto",
                 torch_dtype: str = "auto",
                 max_new_tokens: int = 1000,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        """
        Initialize GPT-OSS service.
        
        Args:
            model_id (str): Model identifier for GPT-OSS
            device (str): Device to run model on
            torch_dtype (str): Torch data type for model
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Temperature for generation
            top_p (float): Top-p for nucleus sampling
        """
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Investment advice templates
        self.advice_templates = {
            "stock_analysis": """당신은 전문적인 투자 분석가입니다. 다음 정보를 바탕으로 상세한 투자 분석을 제공해주세요:

주식 종목: {stock_symbol}
구매 수량: {quantity}주
구매 가격: {price:,}원
전략 정보: {strategy_info}

다음 관점에서 분석해주세요:
1. 기업 분석 (재무상태, 사업모델, 경쟁력)
2. 시장 분석 (업계 전망, 경쟁사 비교)
3. 투자 리스크 (단기/장기 위험요소)
4. 투자 권고 (목표가, 보유기간, 포트폴리오 비중)

분석 근거와 함께 구체적인 투자 조언을 제공해주세요.""",

            "market_outlook": """전문 투자 분석가로서 현재 시장 상황과 투자 전략에 대해 분석해주세요:

시장 정보: {market_info}
관련 뉴스/정보: {news_info}

다음 사항을 포함하여 분석해주세요:
1. 현재 시장 상황 진단
2. 주요 투자 기회와 위험요소
3. 섹터별 투자 전략
4. 포트폴리오 조정 방안

데이터와 근거를 바탕으로 실용적인 투자 조언을 제공해주세요.""",

            "risk_assessment": """투자 위험 평가 전문가로서 다음 투자에 대한 종합적인 리스크 분석을 제공해주세요:

투자 정보: {investment_info}
시장 맥락: {market_context}

분석 항목:
1. 시장 리스크 (변동성, 유동성, 시스템 리스크)
2. 기업 고유 리스크 (재무, 운영, 규제 리스크)
3. 거시경제 리스크 (금리, 환율, 인플레이션)
4. 리스크 관리 방안

위험도 등급과 함께 구체적인 대응 전략을 제시해주세요."""
        }
    
    def _initialize_model(self):
        """Initialize GPT-OSS model and tokenizer."""
        try:
            self.logger.info(f"Loading GPT-OSS model: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.device if self.device == "auto" else torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.logger.info("Model loaded successfully")
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize GPT-OSS model: {e}")
    
    def generate_text(self, 
                     messages: List[Dict[str, str]], 
                     reasoning_effort: str = "medium") -> str:
        """
        Generate text using GPT-OSS model.
        
        Args:
            messages (List[Dict[str, str]]): Chat messages in format [{"role": "user", "content": "..."}]
            reasoning_effort (str): Reasoning effort level ("low", "medium", "high")
            
        Returns:
            str: Generated text
        """
        try:
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=reasoning_effort
            )
            
            # Move to model device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = generated[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"텍스트 생성 중 오류가 발생했습니다: {e}"
    
    def generate_investment_advice(self, 
                                 stock_symbol: str,
                                 quantity: int,
                                 price: float,
                                 strategy_info: str,
                                 reasoning_effort: str = "high") -> Dict[str, Any]:
        """
        Generate comprehensive investment advice.
        
        Args:
            stock_symbol (str): Stock symbol or company name
            quantity (int): Number of shares
            price (float): Price per share
            strategy_info (str): Additional strategy information from knowledge base
            reasoning_effort (str): Reasoning effort level
            
        Returns:
            Dict[str, Any]: Investment advice with metadata
        """
        try:
            # Format the prompt
            prompt = self.advice_templates["stock_analysis"].format(
                stock_symbol=stock_symbol,
                quantity=quantity,
                price=price,
                strategy_info=strategy_info
            )
            
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": "당신은 20년 경력의 전문 투자 분석가입니다. 데이터와 시장 분석을 바탕으로 정확하고 실용적인 투자 조언을 제공합니다."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Generate advice
            advice = self.generate_text(messages, reasoning_effort)
            
            # Calculate basic metrics
            total_investment = quantity * price
            
            return {
                "advice": advice,
                "investment_summary": {
                    "stock_symbol": stock_symbol,
                    "quantity": quantity,
                    "price_per_share": price,
                    "total_investment": total_investment,
                    "currency": "KRW"
                },
                "generation_params": {
                    "reasoning_effort": reasoning_effort,
                    "temperature": self.temperature,
                    "max_tokens": self.max_new_tokens
                },
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_id": self.model_id,
                    "version": "gpt-oss"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating investment advice: {e}")
            return {
                "error": f"투자 조언 생성 중 오류가 발생했습니다: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_market_analysis(self, 
                               market_info: str,
                               news_info: str = "",
                               reasoning_effort: str = "high") -> Dict[str, Any]:
        """
        Generate market analysis and outlook.
        
        Args:
            market_info (str): Market information from knowledge base
            news_info (str): Recent news or additional context
            reasoning_effort (str): Reasoning effort level
            
        Returns:
            Dict[str, Any]: Market analysis with metadata
        """
        try:
            prompt = self.advice_templates["market_outlook"].format(
                market_info=market_info,
                news_info=news_info if news_info else "최신 시장 동향 분석 필요"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "당신은 글로벌 금융시장 전문가입니다. 거시경제 지표와 시장 데이터를 종합하여 투자자들에게 유용한 시장 분석을 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            analysis = self.generate_text(messages, reasoning_effort)
            
            return {
                "market_analysis": analysis,
                "analysis_scope": {
                    "market_info_provided": bool(market_info),
                    "news_info_provided": bool(news_info),
                    "analysis_type": "market_outlook"
                },
                "generation_params": {
                    "reasoning_effort": reasoning_effort,
                    "temperature": self.temperature
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating market analysis: {e}")
            return {
                "error": f"시장 분석 생성 중 오류가 발생했습니다: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def assess_investment_risk(self, 
                             investment_info: str,
                             market_context: str = "",
                             reasoning_effort: str = "high") -> Dict[str, Any]:
        """
        Generate investment risk assessment.
        
        Args:
            investment_info (str): Investment details
            market_context (str): Market context information
            reasoning_effort (str): Reasoning effort level
            
        Returns:
            Dict[str, Any]: Risk assessment with metadata
        """
        try:
            prompt = self.advice_templates["risk_assessment"].format(
                investment_info=investment_info,
                market_context=market_context if market_context else "현재 시장 상황 고려 필요"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "당신은 투자 리스크 관리 전문가입니다. 다양한 위험요소를 체계적으로 분석하고 실용적인 리스크 관리 방안을 제시합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            risk_analysis = self.generate_text(messages, reasoning_effort)
            
            return {
                "risk_assessment": risk_analysis,
                "assessment_scope": {
                    "investment_info_provided": bool(investment_info),
                    "market_context_provided": bool(market_context),
                    "assessment_type": "comprehensive_risk"
                },
                "generation_params": {
                    "reasoning_effort": reasoning_effort,
                    "temperature": self.temperature
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {e}")
            return {
                "error": f"리스크 분석 생성 중 오류가 발생했습니다: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            "model_id": self.model_id,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.model.device) if self.model else None,
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }
        
        if self.model:
            try:
                info["model_params"] = sum(p.numel() for p in self.model.parameters())
                info["model_dtype"] = str(next(self.model.parameters()).dtype)
            except:
                pass
        
        return info


def main():
    """Example usage of GPTOSSService"""
    try:
        print("=== GPT-OSS Service Test ===")
        
        # Note: This requires GPU and proper model access
        print("Initializing GPT-OSS service...")
        
        # For testing purposes, we'll create a mock version
        # In real usage, uncomment the line below:
        # llm_service = GPTOSSService()
        
        print("⚠️  GPT-OSS requires GPU and model access from Hugging Face")
        print("This is a demonstration of the service interface.")
        
        # Mock service for demonstration
        class MockGPTOSSService:
            def __init__(self):
                self.model_id = "openai/gpt-oss-20b"
                self.logger = logging.getLogger(__name__)
            
            def generate_investment_advice(self, stock_symbol, quantity, price, strategy_info, reasoning_effort="high"):
                return {
                    "advice": f"""
**{stock_symbol} 투자 분석 보고서**

1. **기업 분석**
   - 재무 건전성: 양호한 현금흐름과 부채비율 관리
   - 사업 모델: 지속가능한 수익 구조
   - 시장 경쟁력: 업계 내 선도적 위치

2. **시장 분석**
   - 업계 전망: 긍정적 성장 전망
   - 경쟁사 대비 우위: 기술력과 브랜드 가치

3. **투자 리스크**
   - 단기: 시장 변동성에 따른 주가 등락
   - 장기: 산업 구조 변화 대응 필요

4. **투자 권고**
   - 목표 수익률: 15-20% (12개월)
   - 권장 보유 기간: 중장기 (1-2년)
   - 포트폴리오 비중: 5-10%

**종합 평가**: 현재 가격 {price:,}원에서 {quantity}주 투자는 적정한 수준으로 판단됩니다.
                    """,
                    "investment_summary": {
                        "stock_symbol": stock_symbol,
                        "quantity": quantity,
                        "price_per_share": price,
                        "total_investment": quantity * price,
                        "currency": "KRW"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            def get_model_info(self):
                return {
                    "model_id": self.model_id,
                    "status": "mock_service",
                    "note": "Real service requires GPU and model access"
                }
        
        # Use mock service for demonstration
        llm_service = MockGPTOSSService()
        
        # Test investment advice generation
        print("\n=== Testing Investment Advice Generation ===")
        advice_result = llm_service.generate_investment_advice(
            stock_symbol="삼성전자 (005930)",
            quantity=10,
            price=70000,
            strategy_info="YouTube 동영상에서 추출한 투자 전략 정보: 반도체 업계 성장성, AI 칩 수요 증가 전망"
        )
        
        print("투자 조언 결과:")
        print(f"종목: {advice_result['investment_summary']['stock_symbol']}")
        print(f"총 투자금액: {advice_result['investment_summary']['total_investment']:,}원")
        print(f"생성 시간: {advice_result['timestamp']}")
        print(f"\n조언 내용:\n{advice_result['advice']}")
        
        # Model information
        model_info = llm_service.get_model_info()
        print(f"\n=== Model Information ===")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
