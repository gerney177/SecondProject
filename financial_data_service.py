#!/usr/bin/env python3
"""
Financial Data Service
금융 데이터 서비스

KIS Developers API를 활용하여 실시간 금융 데이터를 수집하고 
투자 조언 서비스에 필요한 데이터를 제공하는 서비스입니다.

주요 기능:
- 실시간 주가 데이터 수집
- 차트 데이터 분석
- 시장 지수 모니터링
- 종목 정보 검색
- 투자 분석용 데이터 가공
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from kis_api_client import KISAPIClient, create_kis_client


@dataclass
class StockAnalysis:
    """주식 분석 결과 데이터 클래스"""
    stock_code: str
    stock_name: str
    current_price: float
    change_rate: float
    volume: int
    market_cap: Optional[float] = None
    
    # 기술적 분석 지표
    rsi: Optional[float] = None
    moving_average_5: Optional[float] = None
    moving_average_20: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    
    # 시장 비교
    market_correlation: Optional[float] = None
    sector_performance: Optional[str] = None
    
    # 분석 시점
    analysis_timestamp: str = ""


@dataclass
class MarketSentiment:
    """시장 심리 분석 데이터 클래스"""
    kospi_trend: str  # "상승", "하락", "보합"
    kosdaq_trend: str
    market_volatility: float  # 변동성 지수
    fear_greed_index: Optional[float] = None  # 공포탐욕지수 (0-100)
    
    # 섹터별 동향
    sector_trends: Dict[str, str] = None
    
    analysis_timestamp: str = ""


class FinancialDataService:
    """
    금융 데이터 서비스 클래스
    
    KIS API를 통해 금융 데이터를 수집하고 분석하여
    투자 조언 서비스에 필요한 정보를 제공합니다.
    """
    
    def __init__(self, mock_mode: bool = True):
        """
        금융 데이터 서비스 초기화
        
        Args:
            mock_mode (bool): Mock 모드 사용 여부
        """
        self.mock_mode = mock_mode
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # KIS API 클라이언트 초기화
        self.kis_client = create_kis_client(mock_mode=mock_mode)
        
        # 캐시 설정 (데이터 중복 요청 방지)
        self.cache = {}
        self.cache_ttl = 300  # 5분 캐시
        
        # 주요 종목 코드 매핑
        self.major_stocks = {
            "삼성전자": "005930",
            "SK하이닉스": "000660",
            "NAVER": "035420",
            "카카오": "035720",
            "LG화학": "051910",
            "현대차": "005380",
            "POSCO홀딩스": "005490",
            "KB금융": "105560"
        }
        
        self.logger.info(f"금융 데이터 서비스 초기화 완료 (Mock 모드: {mock_mode})")
    
    def _get_cache_key(self, method: str, *args) -> str:
        """캐시 키 생성"""
        return f"{method}_{hash(str(args))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 검사"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get("timestamp", 0)
        return time.time() - cached_time < self.cache_ttl
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """캐시 데이터 저장"""
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def _get_cache(self, cache_key: str) -> Any:
        """캐시 데이터 조회"""
        return self.cache[cache_key]["data"]
    
    def get_stock_code(self, stock_identifier: str) -> Optional[str]:
        """
        종목명 또는 코드를 표준 종목코드로 변환
        
        Args:
            stock_identifier (str): 종목명 또는 종목코드
        
        Returns:
            Optional[str]: 표준 종목코드 또는 None
        """
        # 이미 종목코드 형식인 경우
        if stock_identifier.isdigit() and len(stock_identifier) == 6:
            return stock_identifier
        
        # 주요 종목명 매핑에서 검색
        if stock_identifier in self.major_stocks:
            return self.major_stocks[stock_identifier]
        
        # KIS API로 종목 검색
        search_results = self.kis_client.search_stock_info(stock_identifier)
        if search_results and len(search_results) > 0:
            return search_results[0]["stock_code"]
        
        return None
    
    def get_real_time_price(self, stock_identifier: str) -> Optional[Dict[str, Any]]:
        """
        실시간 주가 정보 조회
        
        Args:
            stock_identifier (str): 종목명 또는 종목코드
        
        Returns:
            Optional[Dict[str, Any]]: 실시간 주가 정보
        """
        try:
            # 종목코드 변환
            stock_code = self.get_stock_code(stock_identifier)
            if not stock_code:
                self.logger.error(f"종목을 찾을 수 없습니다: {stock_identifier}")
                return None
            
            # 캐시 확인
            cache_key = self._get_cache_key("real_time_price", stock_code)
            if self._is_cache_valid(cache_key):
                return self._get_cache(cache_key)
            
            # KIS API로 현재가 조회
            price_data = self.kis_client.get_current_price(stock_code)
            if price_data:
                # 캐시에 저장
                self._set_cache(cache_key, price_data)
                return price_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"실시간 주가 조회 중 오류 발생: {str(e)}")
            return None
    
    def get_chart_analysis(self, stock_identifier: str, period: int = 30) -> Optional[Dict[str, Any]]:
        """
        차트 데이터 분석
        
        Args:
            stock_identifier (str): 종목명 또는 종목코드
            period (int): 분석 기간 (일수)
        
        Returns:
            Optional[Dict[str, Any]]: 차트 분석 결과
        """
        try:
            # 종목코드 변환
            stock_code = self.get_stock_code(stock_identifier)
            if not stock_code:
                return None
            
            # 차트 데이터 조회
            chart_data = self.kis_client.get_daily_chart(stock_code, period)
            if not chart_data:
                return None
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(chart_data)
            df['close_price'] = df['close_price'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 기술적 분석 지표 계산
            analysis = self._calculate_technical_indicators(df)
            
            return {
                "stock_code": stock_code,
                "period": period,
                "chart_data": chart_data,
                "technical_analysis": analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"차트 분석 중 오류 발생: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기술적 분석 지표 계산"""
        try:
            indicators = {}
            
            if len(df) < 20:
                return indicators
            
            prices = df['close_price'].values
            
            # 이동평균선
            indicators['ma_5'] = float(np.mean(prices[-5:]))
            indicators['ma_20'] = float(np.mean(prices[-20:]))
            
            # RSI 계산 (14일)
            if len(prices) >= 14:
                indicators['rsi'] = self._calculate_rsi(prices, 14)
            
            # 볼린저 밴드 (20일, 2σ)
            if len(prices) >= 20:
                ma_20 = np.mean(prices[-20:])
                std_20 = np.std(prices[-20:])
                indicators['bollinger_upper'] = float(ma_20 + 2 * std_20)
                indicators['bollinger_lower'] = float(ma_20 - 2 * std_20)
            
            # 변동성 (20일 표준편차)
            if len(prices) >= 20:
                indicators['volatility'] = float(np.std(prices[-20:]))
            
            # 거래량 분석
            volumes = df['volume'].values
            if len(volumes) >= 5:
                indicators['avg_volume_5'] = float(np.mean(volumes[-5:]))
                indicators['volume_ratio'] = float(volumes[-1] / np.mean(volumes[-5:]))
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI (Relative Strength Index) 계산"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception:
            return 50.0  # 중립값 반환
    
    def get_market_sentiment(self) -> Optional[MarketSentiment]:
        """
        시장 심리 분석
        
        Returns:
            Optional[MarketSentiment]: 시장 심리 분석 결과
        """
        try:
            # 캐시 확인
            cache_key = self._get_cache_key("market_sentiment")
            if self._is_cache_valid(cache_key):
                cached_data = self._get_cache(cache_key)
                return MarketSentiment(**cached_data)
            
            # 시장 지수 조회
            indices = self.kis_client.get_market_indices()
            if not indices:
                return None
            
            # KOSPI, KOSDAQ 동향 분석
            kospi_data = indices.get("KOSPI", {})
            kosdaq_data = indices.get("KOSDAQ", {})
            
            kospi_change = float(kospi_data.get("change_rate", "0"))
            kosdaq_change = float(kosdaq_data.get("change_rate", "0"))
            
            # 트렌드 결정
            kospi_trend = self._determine_trend(kospi_change)
            kosdaq_trend = self._determine_trend(kosdaq_change)
            
            # 시장 변동성 계산 (간단한 버전)
            volatility = abs(kospi_change) + abs(kosdaq_change)
            
            sentiment = MarketSentiment(
                kospi_trend=kospi_trend,
                kosdaq_trend=kosdaq_trend,
                market_volatility=volatility,
                analysis_timestamp=datetime.now().isoformat()
            )
            
            # 캐시에 저장
            self._set_cache(cache_key, asdict(sentiment))
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"시장 심리 분석 중 오류 발생: {str(e)}")
            return None
    
    def _determine_trend(self, change_rate: float) -> str:
        """등락률을 바탕으로 트렌드 결정"""
        if change_rate > 0.5:
            return "상승"
        elif change_rate < -0.5:
            return "하락"
        else:
            return "보합"
    
    def get_stock_analysis(self, stock_identifier: str) -> Optional[StockAnalysis]:
        """
        종목 종합 분석
        
        Args:
            stock_identifier (str): 종목명 또는 종목코드
        
        Returns:
            Optional[StockAnalysis]: 종목 분석 결과
        """
        try:
            # 실시간 주가 정보
            price_data = self.get_real_time_price(stock_identifier)
            if not price_data:
                return None
            
            # 차트 분석
            chart_analysis = self.get_chart_analysis(stock_identifier, 30)
            
            # 시장 심리
            market_sentiment = self.get_market_sentiment()
            
            # 분석 결과 생성
            analysis = StockAnalysis(
                stock_code=price_data["stock_code"],
                stock_name=price_data["stock_name"],
                current_price=float(price_data["current_price"]),
                change_rate=float(price_data["change_rate"]),
                volume=int(price_data["volume"]),
                analysis_timestamp=datetime.now().isoformat()
            )
            
            # 기술적 분석 지표 추가
            if chart_analysis:
                tech_indicators = chart_analysis.get("technical_analysis", {})
                analysis.rsi = tech_indicators.get("rsi")
                analysis.moving_average_5 = tech_indicators.get("ma_5")
                analysis.moving_average_20 = tech_indicators.get("ma_20")
                analysis.bollinger_upper = tech_indicators.get("bollinger_upper")
                analysis.bollinger_lower = tech_indicators.get("bollinger_lower")
            
            # 시장 상관관계 분석 (간단한 버전)
            if market_sentiment:
                kospi_change = 0.85 if market_sentiment.kospi_trend == "상승" else -0.32
                stock_change = analysis.change_rate
                
                # 상관관계 계산 (간단한 방향성 비교)
                if (kospi_change > 0 and stock_change > 0) or (kospi_change < 0 and stock_change < 0):
                    analysis.market_correlation = 0.7  # 양의 상관관계
                else:
                    analysis.market_correlation = -0.3  # 음의 상관관계
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"종목 분석 중 오류 발생: {str(e)}")
            return None
    
    def get_investment_context(self, stock_identifier: str) -> Optional[Dict[str, Any]]:
        """
        투자 조언을 위한 컨텍스트 데이터 생성
        
        Args:
            stock_identifier (str): 종목명 또는 종목코드
        
        Returns:
            Optional[Dict[str, Any]]: 투자 컨텍스트 데이터
        """
        try:
            # 종목 분석
            stock_analysis = self.get_stock_analysis(stock_identifier)
            if not stock_analysis:
                return None
            
            # 시장 심리
            market_sentiment = self.get_market_sentiment()
            
            # 차트 분석
            chart_analysis = self.get_chart_analysis(stock_identifier, 30)
            
            # 투자 컨텍스트 생성
            context = {
                "stock_analysis": asdict(stock_analysis),
                "market_sentiment": asdict(market_sentiment) if market_sentiment else {},
                "chart_analysis": chart_analysis or {},
                "investment_signals": self._generate_investment_signals(stock_analysis, market_sentiment),
                "risk_assessment": self._assess_risk(stock_analysis, market_sentiment),
                "context_timestamp": datetime.now().isoformat()
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"투자 컨텍스트 생성 중 오류 발생: {str(e)}")
            return None
    
    def _generate_investment_signals(self, stock_analysis: StockAnalysis, market_sentiment: Optional[MarketSentiment]) -> Dict[str, Any]:
        """투자 신호 생성"""
        signals = {
            "technical_signal": "중립",
            "market_signal": "중립",
            "overall_signal": "중립",
            "confidence": 0.5
        }
        
        try:
            # 기술적 분석 신호
            if stock_analysis.rsi:
                if stock_analysis.rsi < 30:
                    signals["technical_signal"] = "매수"
                elif stock_analysis.rsi > 70:
                    signals["technical_signal"] = "매도"
            
            # 시장 신호
            if market_sentiment:
                if market_sentiment.kospi_trend == "상승" and market_sentiment.kosdaq_trend == "상승":
                    signals["market_signal"] = "긍정적"
                elif market_sentiment.kospi_trend == "하락" and market_sentiment.kosdaq_trend == "하락":
                    signals["market_signal"] = "부정적"
            
            # 종합 신호 (간단한 로직)
            positive_signals = sum([
                1 if signals["technical_signal"] == "매수" else 0,
                1 if signals["market_signal"] == "긍정적" else 0,
                1 if stock_analysis.change_rate > 0 else 0
            ])
            
            if positive_signals >= 2:
                signals["overall_signal"] = "긍정적"
                signals["confidence"] = 0.7
            elif positive_signals == 0:
                signals["overall_signal"] = "부정적"
                signals["confidence"] = 0.7
            
        except Exception as e:
            self.logger.error(f"투자 신호 생성 중 오류 발생: {str(e)}")
        
        return signals
    
    def _assess_risk(self, stock_analysis: StockAnalysis, market_sentiment: Optional[MarketSentiment]) -> Dict[str, Any]:
        """리스크 평가"""
        risk_assessment = {
            "risk_level": "중간",
            "volatility_risk": "중간",
            "market_risk": "중간",
            "liquidity_risk": "낮음"
        }
        
        try:
            # 변동성 리스크
            if abs(stock_analysis.change_rate) > 3:
                risk_assessment["volatility_risk"] = "높음"
            elif abs(stock_analysis.change_rate) < 1:
                risk_assessment["volatility_risk"] = "낮음"
            
            # 시장 리스크
            if market_sentiment and market_sentiment.market_volatility > 2:
                risk_assessment["market_risk"] = "높음"
            
            # 종합 리스크 레벨
            high_risk_count = sum([
                1 if risk_assessment["volatility_risk"] == "높음" else 0,
                1 if risk_assessment["market_risk"] == "높음" else 0
            ])
            
            if high_risk_count >= 2:
                risk_assessment["risk_level"] = "높음"
            elif high_risk_count == 0:
                risk_assessment["risk_level"] = "낮음"
            
        except Exception as e:
            self.logger.error(f"리스크 평가 중 오류 발생: {str(e)}")
        
        return risk_assessment
    
    def get_multiple_stocks_analysis(self, stock_identifiers: List[str]) -> Dict[str, Any]:
        """
        여러 종목 동시 분석
        
        Args:
            stock_identifiers (List[str]): 종목 리스트
        
        Returns:
            Dict[str, Any]: 다중 종목 분석 결과
        """
        results = {}
        
        for stock_id in stock_identifiers:
            try:
                analysis = self.get_stock_analysis(stock_id)
                if analysis:
                    results[stock_id] = asdict(analysis)
                else:
                    results[stock_id] = {"error": "분석 실패"}
            except Exception as e:
                results[stock_id] = {"error": str(e)}
        
        return {
            "analyses": results,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_stocks": len(stock_identifiers),
            "successful_analyses": len([r for r in results.values() if "error" not in r])
        }


def main():
    """테스트 및 데모 함수"""
    print("🚀 금융 데이터 서비스 테스트")
    print("=" * 50)
    
    # Mock 모드로 서비스 생성
    service = FinancialDataService(mock_mode=True)
    
    # 1. 실시간 주가 조회
    print("\n📊 삼성전자 실시간 주가:")
    price_data = service.get_real_time_price("삼성전자")
    if price_data:
        print(f"종목명: {price_data['stock_name']}")
        print(f"현재가: {price_data['current_price']}원")
        print(f"등락률: {price_data['change_rate']}%")
    
    # 2. 종목 분석
    print("\n📈 삼성전자 종목 분석:")
    analysis = service.get_stock_analysis("005930")
    if analysis:
        print(f"RSI: {analysis.rsi}")
        print(f"5일 이평: {analysis.moving_average_5}")
        print(f"시장 상관관계: {analysis.market_correlation}")
    
    # 3. 시장 심리 분석
    print("\n📊 시장 심리 분석:")
    sentiment = service.get_market_sentiment()
    if sentiment:
        print(f"KOSPI 동향: {sentiment.kospi_trend}")
        print(f"KOSDAQ 동향: {sentiment.kosdaq_trend}")
        print(f"시장 변동성: {sentiment.market_volatility}")
    
    # 4. 투자 컨텍스트
    print("\n💡 투자 컨텍스트:")
    context = service.get_investment_context("삼성전자")
    if context:
        signals = context["investment_signals"]
        print(f"기술적 신호: {signals['technical_signal']}")
        print(f"시장 신호: {signals['market_signal']}")
        print(f"종합 신호: {signals['overall_signal']}")
        
        risk = context["risk_assessment"]
        print(f"리스크 레벨: {risk['risk_level']}")
    
    # 5. 다중 종목 분석
    print("\n📋 다중 종목 분석:")
    multi_analysis = service.get_multiple_stocks_analysis(["삼성전자", "SK하이닉스"])
    print(f"분석 성공: {multi_analysis['successful_analyses']}/{multi_analysis['total_stocks']}")
    
    print("\n✅ 금융 데이터 서비스 테스트 완료!")


if __name__ == "__main__":
    main()

