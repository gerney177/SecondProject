#!/usr/bin/env python3
"""
KIS Developers API Client for Korean Investment & Securities
한국투자증권 KIS Developers API 클라이언트

이 모듈은 KIS Developers API를 통해 실시간 주식 데이터, 차트 데이터, 
시장 정보 등을 수집하는 기능을 제공합니다.

주요 기능:
- 실시간 주가 조회 > 빼기
- 일봉/분봉 차트 데이터 조회 > 분봉 빼기
- 종목 정보 조회
- 시장 지수 조회 > 빼기
- 뉴스 및 공시 정보 조회 > 빼기
"""

import os
import json
import time
import hashlib
import hmac
import base64
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from urllib.parse import urlencode


@dataclass
class KISConfig:
    """KIS API 설정 클래스"""
    app_key: str
    app_secret: str
    base_url: str = "https://openapi.koreainvestment.com:9443"
    mock_mode: bool = False  # 테스트용 Mock 모드


class KISAPIClient:
    """
    KIS Developers API 클라이언트
    
    한국투자증권의 Open API를 통해 금융 데이터를 수집하는 클라이언트입니다.
    실시간 주가, 차트 데이터, 종목 정보 등을 제공합니다.
    """
    
    def __init__(self, config: KISConfig):
        """
        KIS API 클라이언트 초기화
        
        Args:
            config (KISConfig): KIS API 설정 정보
        """
        self.config = config
        self.access_token = None
        self.token_expires_at = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # API 엔드포인트 정의
        self.endpoints = {
            "token": "/oauth2/tokenP",
            "current_price": "/uapi/domestic-stock/v1/quotations/inquire-price",
            "daily_chart": "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            "minute_chart": "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            "stock_info": "/uapi/domestic-stock/v1/quotations/search-stock-info",
            "market_index": "/uapi/domestic-stock/v1/quotations/inquire-index-price",
            "news": "/uapi/domestic-stock/v1/quotations/inquire-daily-news"
        }
        
        # Mock 데이터 (테스트용)
        self.mock_data = self._initialize_mock_data()
        
        if not self.config.mock_mode:
            self._authenticate()
    
    def _initialize_mock_data(self) -> Dict[str, Any]:
        """테스트용 Mock 데이터 초기화"""
        return {
            "current_price": {
                "005930": {  # 삼성전자
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "current_price": "71000",
                    "change_rate": "1.43",
                    "change_amount": "1000",
                    "volume": "15234567",
                    "market_cap": "425000000000000",
                    "high_price": "71500",
                    "low_price": "70000",
                    "open_price": "70500"
                },
                "000660": {  # SK하이닉스
                    "stock_code": "000660",
                    "stock_name": "SK하이닉스",
                    "current_price": "128000",
                    "change_rate": "2.40",
                    "change_amount": "3000",
                    "volume": "8765432",
                    "market_cap": "93000000000000",
                    "high_price": "129000",
                    "low_price": "125000",
                    "open_price": "126000"
                }
            },
            "market_indices": {
                "KOSPI": {
                    "index_name": "KOSPI",
                    "current_value": "2485.67",
                    "change_rate": "0.85",
                    "change_amount": "20.95"
                },
                "KOSDAQ": {
                    "index_name": "KOSDAQ",
                    "current_value": "745.23",
                    "change_rate": "-0.32",
                    "change_amount": "-2.41"
                }
            }
        }
    
    def _authenticate(self) -> bool:
        """
        KIS API 인증 토큰 발급
        
        Returns:
            bool: 인증 성공 여부
        """
        try:
            url = f"{self.config.base_url}{self.endpoints['token']}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get("access_token")
                expires_in = result.get("expires_in", 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.logger.info("KIS API 인증 성공")
                return True
            else:
                self.logger.error(f"KIS API 인증 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"KIS API 인증 중 오류 발생: {str(e)}")
            return False
    
    def _is_token_valid(self) -> bool:
        """토큰 유효성 검사"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now() < self.token_expires_at
    
    def _get_headers(self, tr_id: str) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        if not self._is_token_valid():
            self._authenticate()
        
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id
        }
    
    def get_current_price(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        실시간 주가 조회
        
        Args:
            stock_code (str): 종목코드 (예: "005930")
        
        Returns:
            Optional[Dict[str, Any]]: 주가 정보 또는 None
        """
        if self.config.mock_mode:
            return self.mock_data["current_price"].get(stock_code)
        
        try:
            url = f"{self.config.base_url}{self.endpoints['current_price']}"
            headers = self._get_headers("FHKST01010100")
            
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    output = result.get("output", {})
                    return {
                        "stock_code": stock_code,
                        "stock_name": output.get("hts_kor_isnm", ""),
                        "current_price": output.get("stck_prpr", "0"),
                        "change_rate": output.get("prdy_ctrt", "0"),
                        "change_amount": output.get("prdy_vrss", "0"),
                        "volume": output.get("acml_vol", "0"),
                        "high_price": output.get("stck_hgpr", "0"),
                        "low_price": output.get("stck_lwpr", "0"),
                        "open_price": output.get("stck_oprc", "0"),
                        "timestamp": datetime.now().isoformat()
                    }
            
            self.logger.error(f"주가 조회 실패: {response.status_code}")
            return None
            
        except Exception as e:
            self.logger.error(f"주가 조회 중 오류 발생: {str(e)}")
            return None
    
    def get_daily_chart(self, stock_code: str, period: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        일봉 차트 데이터 조회
        
        Args:
            stock_code (str): 종목코드
            period (int): 조회 기간 (일수)
        
        Returns:
            Optional[List[Dict[str, Any]]]: 차트 데이터 리스트 또는 None
        """
        if self.config.mock_mode:
            # Mock 데이터 생성
            chart_data = []
            base_price = 70000
            for i in range(period):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                price_change = (i % 5 - 2) * 1000  # 간단한 가격 변동
                chart_data.append({
                    "date": date,
                    "open_price": str(base_price + price_change),
                    "high_price": str(base_price + price_change + 500),
                    "low_price": str(base_price + price_change - 500),
                    "close_price": str(base_price + price_change + 200),
                    "volume": str(10000000 + i * 100000)
                })
            return chart_data[::-1]  # 날짜 순으로 정렬
        
        try:
            url = f"{self.config.base_url}{self.endpoints['daily_chart']}"
            headers = self._get_headers("FHKST03010100")
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=period)).strftime("%Y%m%d")
            
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
                "fid_input_date_1": start_date,
                "fid_input_date_2": end_date,
                "fid_period_div_code": "D"
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    output_list = result.get("output2", [])
                    chart_data = []
                    
                    for item in output_list:
                        chart_data.append({
                            "date": item.get("stck_bsop_date", ""),
                            "open_price": item.get("stck_oprc", "0"),
                            "high_price": item.get("stck_hgpr", "0"),
                            "low_price": item.get("stck_lwpr", "0"),
                            "close_price": item.get("stck_clpr", "0"),
                            "volume": item.get("acml_vol", "0")
                        })
                    
                    return chart_data
            
            self.logger.error(f"차트 데이터 조회 실패: {response.status_code}")
            return None
            
        except Exception as e:
            self.logger.error(f"차트 데이터 조회 중 오류 발생: {str(e)}")
            return None
    
    def get_market_indices(self) -> Optional[Dict[str, Any]]:
        """
        주요 시장 지수 조회 (KOSPI, KOSDAQ)
        
        Returns:
            Optional[Dict[str, Any]]: 시장 지수 정보 또는 None
        """
        if self.config.mock_mode:
            return self.mock_data["market_indices"]
        
        try:
            indices = {}
            
            # KOSPI 조회
            kospi_data = self._get_index_data("0001")  # KOSPI 코드
            if kospi_data:
                indices["KOSPI"] = kospi_data
            
            # KOSDAQ 조회
            kosdaq_data = self._get_index_data("1001")  # KOSDAQ 코드
            if kosdaq_data:
                indices["KOSDAQ"] = kosdaq_data
            
            return indices if indices else None
            
        except Exception as e:
            self.logger.error(f"시장 지수 조회 중 오류 발생: {str(e)}")
            return None
    
    def _get_index_data(self, index_code: str) -> Optional[Dict[str, Any]]:
        """개별 지수 데이터 조회"""
        try:
            url = f"{self.config.base_url}{self.endpoints['market_index']}"
            headers = self._get_headers("FHPUP02100000")
            
            params = {
                "fid_cond_mrkt_div_code": "U",
                "fid_input_iscd": index_code
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    output = result.get("output", {})
                    return {
                        "index_name": output.get("bstp_nmix_prpr", ""),
                        "current_value": output.get("bstp_nmix_prpr", "0"),
                        "change_rate": output.get("bstp_nmix_prdy_ctrt", "0"),
                        "change_amount": output.get("bstp_nmix_prdy_vrss", "0"),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"지수 데이터 조회 중 오류 발생: {str(e)}")
            return None
    
    def search_stock_info(self, keyword: str) -> Optional[List[Dict[str, Any]]]:
        """
        종목 검색
        
        Args:
            keyword (str): 검색 키워드 (종목명 또는 코드)
        
        Returns:
            Optional[List[Dict[str, Any]]]: 검색 결과 리스트 또는 None
        """
        if self.config.mock_mode:
            # Mock 검색 결과
            mock_results = []
            if "삼성" in keyword or "005930" in keyword:
                mock_results.append({
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "market_type": "KOSPI"
                })
            if "SK" in keyword or "000660" in keyword:
                mock_results.append({
                    "stock_code": "000660",
                    "stock_name": "SK하이닉스",
                    "market_type": "KOSPI"
                })
            return mock_results
        
        try:
            url = f"{self.config.base_url}{self.endpoints['stock_info']}"
            headers = self._get_headers("CTPF1002R")
            
            params = {
                "user_id": "",
                "seq": "1",
                "gubun": "1",
                "text": keyword
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    output_list = result.get("output", [])
                    search_results = []
                    
                    for item in output_list:
                        search_results.append({
                            "stock_code": item.get("pdno", ""),
                            "stock_name": item.get("prdt_name", ""),
                            "market_type": item.get("prdt_type_cd", "")
                        })
                    
                    return search_results
            
            self.logger.error(f"종목 검색 실패: {response.status_code}")
            return None
            
        except Exception as e:
            self.logger.error(f"종목 검색 중 오류 발생: {str(e)}")
            return None
    
    def get_stock_analysis_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        종목 분석용 종합 데이터 조회
        
        Args:
            stock_code (str): 종목코드
        
        Returns:
            Optional[Dict[str, Any]]: 종합 분석 데이터 또는 None
        """
        try:
            # 현재가 정보
            current_data = self.get_current_price(stock_code)
            if not current_data:
                return None
            
            # 차트 데이터 (최근 30일)
            chart_data = self.get_daily_chart(stock_code, 30)
            
            # 시장 지수 정보
            market_indices = self.get_market_indices()
            
            return {
                "stock_info": current_data,
                "chart_data": chart_data or [],
                "market_indices": market_indices or {},
                "analysis_timestamp": datetime.now().isoformat(),
                "data_source": "KIS_API" if not self.config.mock_mode else "MOCK_DATA"
            }
            
        except Exception as e:
            self.logger.error(f"종합 데이터 조회 중 오류 발생: {str(e)}")
            return None


def create_kis_client(app_key: str = None, app_secret: str = None, mock_mode: bool = True) -> KISAPIClient:
    """
    KIS API 클라이언트 생성 헬퍼 함수
    
    Args:
        app_key (str): KIS API 앱 키 (환경변수에서 자동 로드)
        app_secret (str): KIS API 앱 시크릿 (환경변수에서 자동 로드)
        mock_mode (bool): Mock 모드 사용 여부 (기본값: True)
    
    Returns:
        KISAPIClient: KIS API 클라이언트 인스턴스
    """
    # 환경변수에서 API 키 로드
    if not app_key:
        app_key = os.getenv("KIS_APP_KEY", "")
    if not app_secret:
        app_secret = os.getenv("KIS_APP_SECRET", "")
    
    # API 키가 없으면 Mock 모드로 설정
    if not app_key or not app_secret:
        mock_mode = True
        app_key = "mock_key"
        app_secret = "mock_secret"
    
    config = KISConfig(
        app_key=app_key,
        app_secret=app_secret,
        mock_mode=mock_mode
    )
    
    return KISAPIClient(config)


def main():
    """테스트 및 데모 함수"""
    print("🚀 KIS Developers API 클라이언트 테스트")
    print("=" * 50)
    
    # Mock 모드로 클라이언트 생성
    client = create_kis_client(mock_mode=True)
    
    # 1. 삼성전자 현재가 조회
    print("\n📊 삼성전자 현재가 조회:")
    samsung_price = client.get_current_price("005930")
    if samsung_price:
        print(f"종목명: {samsung_price['stock_name']}")
        print(f"현재가: {samsung_price['current_price']}원")
        print(f"등락률: {samsung_price['change_rate']}%")
    
    # 2. 차트 데이터 조회
    print("\n📈 차트 데이터 조회 (최근 5일):")
    chart_data = client.get_daily_chart("005930", 5)
    if chart_data:
        for data in chart_data[-3:]:  # 최근 3일만 출력
            print(f"날짜: {data['date']}, 종가: {data['close_price']}원")
    
    # 3. 시장 지수 조회
    print("\n📊 시장 지수 조회:")
    indices = client.get_market_indices()
    if indices:
        for index_name, index_data in indices.items():
            print(f"{index_name}: {index_data['current_value']} ({index_data['change_rate']}%)")
    
    # 4. 종목 검색
    print("\n🔍 종목 검색 (삼성):")
    search_results = client.search_stock_info("삼성")
    if search_results:
        for result in search_results:
            print(f"코드: {result['stock_code']}, 종목명: {result['stock_name']}")
    
    # 5. 종합 분석 데이터
    print("\n📋 종합 분석 데이터:")
    analysis_data = client.get_stock_analysis_data("005930")
    if analysis_data:
        print(f"데이터 소스: {analysis_data['data_source']}")
        print(f"분석 시점: {analysis_data['analysis_timestamp']}")
        print(f"차트 데이터 개수: {len(analysis_data['chart_data'])}개")
    
    print("\n✅ KIS API 클라이언트 테스트 완료!")


if __name__ == "__main__":
    main()

